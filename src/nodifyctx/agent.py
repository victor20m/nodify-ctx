from __future__ import annotations

from dataclasses import dataclass, field
import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, TypedDict
from uuid import uuid4

from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from openai import APIConnectionError, APITimeoutError

from .graph_builder import EmbeddingsServiceError, GraphStoreConfig, KnowledgeGraphBuilder
from .tools import TOOL_FUNCTIONS, build_langchain_tools, close_runtime, semantic_search


_env_path = find_dotenv()
if _env_path:
    load_dotenv(_env_path)
else:
    load_dotenv()

try:
    print(f"Loaded .env from: {_env_path}")
except Exception:
    pass


class AgentState(TypedDict):
    messages: list[BaseMessage]
    search_results: list[dict[str, str]]
    traversal_count: int
    iteration_count: int


class ModelServiceError(RuntimeError):
    """Raised when the configured chat model service cannot complete a request."""


def _env(name: str, default: str | None = None, *, legacy_name: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is not None:
        return value
    if legacy_name is not None:
        legacy_value = os.getenv(legacy_name)
        if legacy_value is not None:
            return legacy_value
    return default


def _env_int(name: str, default: int) -> int:
    raw_value = _env(name, legacy_name=name.replace("NODIFYCTX_", "REPOCONTEXT_"))
    if raw_value is None or not raw_value.strip():
        return default

    value = int(raw_value)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _env_non_negative_int(name: str, default: int) -> int:
    raw_value = _env(name, legacy_name=name.replace("NODIFYCTX_", "REPOCONTEXT_"))
    if raw_value is None or not raw_value.strip():
        return default

    value = int(raw_value)
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return value


def _env_recursion_limit(name: str, default: int) -> int:
    raw_value = _env(name, legacy_name=name.replace("NODIFYCTX_", "REPOCONTEXT_"))
    if raw_value is None or not raw_value.strip():
        return default

    normalized = raw_value.strip().lower()
    if normalized in {"none", "unlimited", "no-limit", "no-limits", "inf", "infinite", "0"}:
        return 1_000_000

    value = int(normalized)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer, or one of: none, unlimited, 0.")
    return value


def _env_float(name: str, default: float) -> float:
    raw_value = _env(name, legacy_name=name.replace("NODIFYCTX_", "REPOCONTEXT_"))
    if raw_value is None or not raw_value.strip():
        return default

    value = float(raw_value)
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0.")
    return value


def _normalize_provider(provider: str | None) -> str:
    normalized = (provider or "").strip().lower().replace("_", "-")
    aliases = {
        "anthropic": "anthropic",
        "claude": "anthropic",
        "deepseek": "deepseek",
        "lm-studio": "lmstudio",
        "lmstudio": "lmstudio",
        "openai": "openai",
        "openai-compatible": "openai-compatible",
    }
    if not normalized:
        return ""
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            "Unsupported chat provider. Expected one of: anthropic, claude, deepseek, lmstudio, openai, openai-compatible."
        ) from exc


def _infer_chat_provider(base_url: str | None, model_name: str | None) -> str:
    normalized_base_url = (base_url or "").strip().lower()
    normalized_model_name = (model_name or "").strip().lower()

    if "anthropic.com" in normalized_base_url or normalized_model_name.startswith("claude"):
        return "anthropic"
    if "deepseek.com" in normalized_base_url or normalized_model_name.startswith("deepseek"):
        return "deepseek"
    if "api.openai.com" in normalized_base_url or normalized_model_name.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    if any(host in normalized_base_url for host in ("127.0.0.1:1234", "localhost:1234", "lmstudio")):
        return "lmstudio"
    if normalized_base_url:
        return "openai-compatible"
    return "lmstudio"


def _resolve_chat_provider(provider: str | None, base_url: str | None, model_name: str | None) -> str:
    normalized_provider = _normalize_provider(provider)
    if normalized_provider:
        return normalized_provider
    return _infer_chat_provider(base_url, model_name)


def _default_chat_base_url(provider: str) -> str | None:
    return {
        "deepseek": "https://api.deepseek.com/v1",
        "lmstudio": "http://127.0.0.1:1234/v1",
        "openai": "https://api.openai.com/v1",
    }.get(provider)


def _provider_api_key_env_names(provider: str) -> list[tuple[str, str | None]]:
    # Deprecated: provider-specific API key env names removed. Keep for compatibility.
    return []


def _resolve_provider_api_key(provider: str, explicit_value: str | None) -> str | None:
    if explicit_value is not None and explicit_value.strip():
        return explicit_value.strip()

    shared_value = _env("NODIFYCTX_MODEL_API_KEY", legacy_name="REPOCONTEXT_MODEL_API_KEY")
    if shared_value is not None and shared_value.strip():
        return shared_value.strip()

    legacy_value = _env("NODIFYCTX_LMSTUDIO_API_KEY", "lm-studio", legacy_name="REPOCONTEXT_LMSTUDIO_API_KEY")
    if legacy_value is not None and legacy_value.strip():
        return legacy_value.strip()

    if provider in {"lmstudio", "openai-compatible"}:
        return "lm-studio"
    return None


@dataclass(slots=True)
class ChatBackend:
    label: str
    provider: str
    model_name: str
    base_url: str | None
    model: Any
    tool_model: Any


@dataclass(slots=True)
class AgentRuntimeConfig:
    chat_model: str
    chat_provider: str | None = field(default_factory=lambda: _env("NODIFYCTX_CHAT_PROVIDER", legacy_name="REPOCONTEXT_CHAT_PROVIDER"))
    chat_base_url: str | None = field(
        default_factory=lambda: os.getenv(
            "NODIFYCTX_CHAT_BASE_URL",
            _env("REPOCONTEXT_CHAT_BASE_URL", legacy_name="REPOCONTEXT_MODEL_BASE_URL")
            or _env("NODIFYCTX_MODEL_BASE_URL", legacy_name="REPOCONTEXT_MODEL_BASE_URL"),
        )
    )
    embeddings_base_url: str = field(
        default_factory=lambda: os.getenv(
            "NODIFYCTX_EMBEDDINGS_BASE_URL",
            _env("REPOCONTEXT_EMBEDDINGS_BASE_URL", legacy_name="NODIFYCTX_LMSTUDIO_BASE_URL")
            or _env("NODIFYCTX_LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1", legacy_name="REPOCONTEXT_LMSTUDIO_BASE_URL"),
        )
    )
    model_api_key: str | None = field(default_factory=lambda: _env("NODIFYCTX_MODEL_API_KEY", legacy_name="REPOCONTEXT_MODEL_API_KEY"))
    model_timeout_seconds: float = field(default_factory=lambda: _env_float("NODIFYCTX_MODEL_TIMEOUT_SECONDS", 30.0))
    model_max_retries: int = field(default_factory=lambda: _env_non_negative_int("NODIFYCTX_MODEL_MAX_RETRIES", 2))
    temperature: float = 0.0
    max_tool_iterations: int = field(default_factory=lambda: _env_int("NODIFYCTX_MAX_TOOL_ITERATIONS", 10))
    graph_recursion_limit: int = field(default_factory=lambda: _env_recursion_limit("NODIFYCTX_GRAPH_RECURSION_LIMIT", 100))
    fallback_chat_model: str | None = field(default_factory=lambda: _env("NODIFYCTX_FALLBACK_CHAT_MODEL", legacy_name="REPOCONTEXT_FALLBACK_CHAT_MODEL"))
    fallback_chat_provider: str | None = field(default_factory=lambda: _env("NODIFYCTX_FALLBACK_CHAT_PROVIDER", legacy_name="REPOCONTEXT_FALLBACK_CHAT_PROVIDER"))
    fallback_chat_base_url: str | None = field(default_factory=lambda: _env("NODIFYCTX_FALLBACK_CHAT_BASE_URL", legacy_name="REPOCONTEXT_FALLBACK_CHAT_BASE_URL"))
    fallback_model_api_key: str | None = field(default_factory=lambda: _env("NODIFYCTX_FALLBACK_MODEL_API_KEY", legacy_name="REPOCONTEXT_FALLBACK_MODEL_API_KEY"))

    def __post_init__(self) -> None:
        self.chat_provider = _resolve_chat_provider(self.chat_provider, self.chat_base_url, self.chat_model)
        self.chat_base_url = (self.chat_base_url or _default_chat_base_url(self.chat_provider) or None)
        self.model_api_key = _resolve_provider_api_key(self.chat_provider, self.model_api_key)

        if self.fallback_chat_model is not None and not self.fallback_chat_model.strip():
            self.fallback_chat_model = None
        if self.fallback_chat_model is not None:
            self.fallback_chat_model = self.fallback_chat_model.strip()
            self.fallback_chat_provider = _resolve_chat_provider(
                self.fallback_chat_provider or self.chat_provider,
                self.fallback_chat_base_url,
                self.fallback_chat_model,
            )
            self.fallback_chat_base_url = self.fallback_chat_base_url or _default_chat_base_url(self.fallback_chat_provider)
            self.fallback_model_api_key = _resolve_provider_api_key(
                self.fallback_chat_provider,
                self.fallback_model_api_key or self.model_api_key,
            )
        else:
            self.fallback_chat_provider = None
            self.fallback_chat_base_url = None
            self.fallback_model_api_key = None


class NodifyCtxAgent:
    """LangGraph-powered CLI agent for graph-guided code exploration."""

    SYSTEM_PROMPT = (
        "You are NodifyCtx, a local code exploration agent.\n"
        "Follow these rules strictly:\n"
        "1. Never inspect code immediately.\n"
        "2. Always begin from semantic_search results.\n"
        "3. Build a mental map with get_callers and get_dependencies before inspect_code.\n"
        "4. Only inspect code when the correct node is isolated.\n"
        "5. Keep final answers concise and grounded in tool outputs.\n"
        "6. Never output <think>, <tool_call>, <function=...>, or XML-like tool markup.\n"
        "7. If you need a tool, use the model's native tool call interface only.\n"
        "8. If a tool returns an error or ambiguity, continue by choosing another valid tool instead of stopping."
    )

    def __init__(self, config: AgentRuntimeConfig) -> None:
        self.config = config
        self.tool_functions = {tool_function.__name__: tool_function for tool_function in TOOL_FUNCTIONS}
        self.tools = build_langchain_tools()
        self.model_backends = self._build_chat_backends()
        self.model = self.model_backends[0].model
        self.tool_model = self.model_backends[0].tool_model
        self.graph = self._build_graph()

    def run(self, query: str) -> str:
        if not query.strip():
            raise ValueError("query must not be empty.")

        initial_state: AgentState = {
            "messages": [
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=query.strip()),
            ],
            "search_results": [],
            "traversal_count": 0,
            "iteration_count": 0,
        }
        final_state = self.graph.invoke(
            initial_state,
            config={"recursion_limit": self.config.graph_recursion_limit},
        )
        final_answer = self._last_final_answer(final_state["messages"])
        print(f"[final]\n{final_answer}\n")
        return final_answer

    def _build_graph(self) -> Any:
        graph = StateGraph(AgentState)
        graph.add_node("semantic_seed", self._semantic_seed)
        graph.add_node("reason", self._reason)
        graph.add_node("execute_tools", self._execute_tools)
        graph.add_edge(START, "semantic_seed")
        graph.add_edge("semantic_seed", "reason")
        graph.add_conditional_edges("reason", self._route_after_reason)
        graph.add_edge("execute_tools", "reason")
        return graph.compile()

    def _semantic_seed(self, state: AgentState) -> AgentState:
        query = self._latest_user_query(state["messages"])
        results = semantic_search(query)
        self._print_step("semantic_search", results)

        seed_message = SystemMessage(
            content=(
                "Initial semantic_search results are below. Use them as your starting map.\n"
                f"{json.dumps(results, ensure_ascii=True)}\n"
                "You must call get_callers or get_dependencies before inspect_code."
            )
        )
        return {
            **state,
            "messages": state["messages"] + [seed_message],
            "search_results": results,
        }

    def _reason(self, state: AgentState) -> AgentState:
        if state["iteration_count"] >= self.config.max_tool_iterations:
            response = self._invoke_chat_model(
                state["messages"]
                + [
                    SystemMessage(
                        content=(
                            "Tool budget reached. Do not call more tools. "
                            "Provide the best final answer now in plain text using only the existing tool outputs."
                        )
                    )
                ],
                use_tools=False,
            )
            response = self._finalize_budget_response(state, response)
        else:
            response = self._invoke_chat_model(state["messages"], use_tools=True)
            response = self._coerce_tool_calls(response)

        visible_content = self._visible_message_content(response.content)
        if visible_content:
            print(f"[agent]\n{visible_content}\n")

        return {
            **state,
            "messages": state["messages"] + [response],
        }

    def _route_after_reason(self, state: AgentState) -> str:
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage):
            if last_message.tool_calls:
                return "execute_tools"
            if self._parse_tool_calls_from_text(str(last_message.content or "")):
                return "execute_tools"
        return END

    def _execute_tools(self, state: AgentState) -> AgentState:
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            raise RuntimeError("Expected the last graph message to be an AI tool-selection message.")

        messages = list(state["messages"])
        traversal_count = state["traversal_count"]
        tool_calls = last_message.tool_calls or self._parse_tool_calls_from_text(str(last_message.content or ""))

        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})

            if tool_name == "inspect_code" and traversal_count == 0:
                result: Any = {
                    "error": "inspect_code is blocked until get_callers or get_dependencies has been used.",
                }
            else:
                try:
                    tool_function = self.tool_functions[tool_name]
                except KeyError as exc:
                    raise RuntimeError(f"Unknown tool requested by the model: {tool_name}") from exc

                result = tool_function(**tool_args)
                if tool_name in {"get_callers", "get_dependencies"}:
                    traversal_count += 1

            self._print_step(tool_name, result)
            messages.append(
                ToolMessage(
                    content=self._serialize_tool_result(result),
                    tool_call_id=tool_call["id"],
                )
            )

        return {
            **state,
            "messages": messages,
            "traversal_count": traversal_count,
            "iteration_count": state["iteration_count"] + 1,
        }

    @staticmethod
    def _latest_user_query(messages: list[BaseMessage]) -> str:
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                return str(message.content)
        raise RuntimeError("No user query was found in the agent state.")

    @staticmethod
    def _last_final_answer(messages: list[BaseMessage]) -> str:
        for message in reversed(messages):
            if isinstance(message, AIMessage) and message.content and not message.tool_calls:
                return NodifyCtxAgent._visible_message_content(message.content)
        raise RuntimeError("The agent did not produce a final answer.")

    @staticmethod
    def _visible_message_content(content: Any) -> str:
        text = str(content or "")
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
        return text

    @staticmethod
    def _contains_raw_tool_markup(content: Any) -> bool:
        text = str(content or "")
        return "<tool_call>" in text or "<function=" in text

    def _coerce_tool_calls(self, response: AIMessage) -> AIMessage:
        if response.tool_calls:
            return response

        fallback_tool_calls = self._parse_tool_calls_from_text(str(response.content or ""))
        if not fallback_tool_calls:
            return response

        return AIMessage(
            content=response.content,
            additional_kwargs=response.additional_kwargs,
            response_metadata=response.response_metadata,
            id=response.id,
            tool_calls=fallback_tool_calls,
        )

    def _parse_tool_calls_from_text(self, text: str) -> list[dict[str, Any]]:
        pattern = re.compile(
            r"<tool_call>\s*<function=(?P<name>[a-zA-Z_][\w]*)>\s*(?P<body>.*?)\s*</function>\s*</tool_call>",
            flags=re.DOTALL | re.IGNORECASE,
        )
        parameter_pattern = re.compile(
            r"<parameter=(?P<key>[a-zA-Z_][\w]*)>\s*(?P<value>.*?)\s*</parameter>",
            flags=re.DOTALL | re.IGNORECASE,
        )

        tool_calls: list[dict[str, Any]] = []
        for match in pattern.finditer(text):
            tool_name = match.group("name")
            if tool_name not in self.tool_functions:
                continue

            raw_body = match.group("body")
            args: dict[str, Any] = {}
            for parameter_match in parameter_pattern.finditer(raw_body):
                key = parameter_match.group("key")
                value = parameter_match.group("value").strip()
                args[key] = value

            tool_calls.append(
                {
                    "name": tool_name,
                    "args": args,
                    "id": f"fallback_{uuid4().hex}",
                    "type": "tool_call",
                }
            )

        return tool_calls

    def _finalize_budget_response(self, state: AgentState, response: AIMessage) -> AIMessage:
        visible_content = self._visible_message_content(response.content)
        if visible_content and not self._contains_raw_tool_markup(response.content):
            return response

        retry_response = self._invoke_chat_model(
            state["messages"]
            + [
                SystemMessage(
                    content=(
                        "Answer now in plain text only. Do not request tools. Do not emit XML, <tool_call>, "
                        "or reasoning tags. Use only the evidence already gathered."
                    )
                )
            ],
            use_tools=False,
        )
        retry_visible_content = self._visible_message_content(retry_response.content)
        if retry_visible_content and not self._contains_raw_tool_markup(retry_response.content):
            return retry_response

        return AIMessage(
            content=(
                "Unable to complete the analysis within the configured tool budget. "
                "Increase NODIFYCTX_MAX_TOOL_ITERATIONS, increase NODIFYCTX_GRAPH_RECURSION_LIMIT, "
                "or ask a narrower question."
            )
        )

    @staticmethod
    def _serialize_tool_result(result: Any) -> str:
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=True)

    @staticmethod
    def _print_step(name: str, payload: Any) -> None:
        print(f"[tool] {name}")
        if isinstance(payload, str):
            print(f"{payload}\n")
            return
        print(f"{json.dumps(payload, indent=2, ensure_ascii=True)}\n")

    def _build_chat_backends(self) -> list[ChatBackend]:
        backends = [
            self._build_chat_backend(
                provider=self.config.chat_provider,
                model_name=self.config.chat_model,
                base_url=self.config.chat_base_url,
                api_key=self.config.model_api_key,
            )
        ]

        if self.config.fallback_chat_model:
            fallback_backend = self._build_chat_backend(
                provider=self.config.fallback_chat_provider or self.config.chat_provider,
                model_name=self.config.fallback_chat_model,
                base_url=self.config.fallback_chat_base_url,
                api_key=self.config.fallback_model_api_key,
            )
            if fallback_backend.label != backends[0].label:
                backends.append(fallback_backend)

        return backends

    def _build_chat_backend(self, *, provider: str, model_name: str, base_url: str | None, api_key: str | None) -> ChatBackend:
        if provider == "anthropic":
            try:
                from langchain_anthropic import ChatAnthropic
            except ImportError as exc:
                raise RuntimeError(
                    "Claude support requires langchain-anthropic. Reinstall dependencies after updating requirements."
                ) from exc

            if api_key is None or not api_key.strip():
                raise ValueError(
                    "An API key is required for Anthropic chat models. Set NODIFYCTX_ANTHROPIC_API_KEY or NODIFYCTX_MODEL_API_KEY."
                )

            model = ChatAnthropic(
                model=model_name,
                api_key=api_key,
                timeout=self.config.model_timeout_seconds,
                temperature=self.config.temperature,
                max_retries=self.config.model_max_retries,
            )
            resolved_base_url = None
        else:
            resolved_base_url = base_url or _default_chat_base_url(provider)
            if resolved_base_url is None or not resolved_base_url.strip():
                raise ValueError(
                    f"A chat base URL is required for provider '{provider}'. Set NODIFYCTX_CHAT_BASE_URL or pass --chat-base-url."
                )
            if api_key is None or not api_key.strip():
                provider_hint = provider.upper().replace("-", "_")
                raise ValueError(
                    f"An API key is required for provider '{provider}'. Set NODIFYCTX_{provider_hint}_API_KEY or NODIFYCTX_MODEL_API_KEY."
                )

            model = ChatOpenAI(
                model=model_name,
                base_url=resolved_base_url,
                api_key=api_key,
                timeout=self.config.model_timeout_seconds,
                temperature=self.config.temperature,
                max_retries=self.config.model_max_retries,
            )

        return ChatBackend(
            label=f"{provider}:{model_name}",
            provider=provider,
            model_name=model_name,
            base_url=resolved_base_url,
            model=model,
            tool_model=model.bind_tools(self.tools),
        )

    def _invoke_chat_model(self, messages: list[BaseMessage], *, use_tools: bool) -> AIMessage:
        failures: list[tuple[ChatBackend, Exception]] = []
        for index, backend in enumerate(self.model_backends):
            runnable = backend.tool_model if use_tools else backend.model
            try:
                return runnable.invoke(messages)
            except Exception as exc:
                failures.append((backend, exc))
                if index < len(self.model_backends) - 1:
                    print(f"[model] {backend.label} failed: {self._summarize_model_exception(exc)}")

        last_backend, last_exception = failures[-1]
        raise ModelServiceError(self._format_model_failure_message(failures)) from last_exception

    @staticmethod
    def _summarize_model_exception(exc: Exception) -> str:
        message = str(exc).strip()
        if message:
            return message
        return exc.__class__.__name__

    def _format_model_failure_message(self, failures: list[tuple[ChatBackend, Exception]]) -> str:
        summaries = []
        for backend, exc in failures:
            endpoint = backend.base_url or backend.provider
            summaries.append(f"{backend.label} at {endpoint}: {self._summarize_model_exception(exc)}")

        last_exception = failures[-1][1]
        if isinstance(last_exception, (APITimeoutError, APIConnectionError)) or "timed out" in str(last_exception).lower():
            guidance = (
                "Increase NODIFYCTX_MODEL_TIMEOUT_SECONDS, raise NODIFYCTX_MODEL_MAX_RETRIES, "
                "or configure NODIFYCTX_FALLBACK_CHAT_MODEL to fail over to another provider."
            )
        else:
            guidance = "Check the configured provider, model name, API key, and base URL."

        return "All configured chat models failed. " + "; ".join(summaries) + f". {guidance}"


def parse_args() -> argparse.Namespace:
    cli = argparse.ArgumentParser(description="Run the NodifyCtx local agent.")
    cli.add_argument("repository", nargs="?", default=".")
    cli.add_argument("--question", help="Ask one question non-interactively and exit.")
    cli.add_argument("--chat-model", default=_env("NODIFYCTX_CHAT_MODEL", legacy_name="REPOCONTEXT_CHAT_MODEL"))
    cli.add_argument(
        "--chat-provider",
        default=_env("NODIFYCTX_CHAT_PROVIDER", legacy_name="REPOCONTEXT_CHAT_PROVIDER"),
        help="Chat provider: lmstudio, openai-compatible, openai, deepseek, anthropic, or claude.",
    )
    cli.add_argument(
        "--embedding-model",
        default=_env("NODIFYCTX_EMBEDDING_MODEL", "text-embedding-nomic-embed-text-v1.5", legacy_name="REPOCONTEXT_EMBEDDING_MODEL"),
    )
    cli.add_argument(
        "--chat-base-url",
        default=_env("NODIFYCTX_CHAT_BASE_URL", legacy_name="REPOCONTEXT_CHAT_BASE_URL"),
        help="Base URL for chat/completions API (overrides NODIFYCTX_CHAT_BASE_URL).",
    )
    cli.add_argument(
        "--embeddings-base-url",
        default=_env("NODIFYCTX_EMBEDDINGS_BASE_URL", legacy_name="REPOCONTEXT_EMBEDDINGS_BASE_URL"),
        help="Base URL for embeddings API (overrides NODIFYCTX_EMBEDDINGS_BASE_URL).",
    )
    cli.add_argument(
        "--model-base-url",
        default=_env("NODIFYCTX_MODEL_BASE_URL", legacy_name="REPOCONTEXT_MODEL_BASE_URL"),
        help="Optional single base URL to use for both chat and embeddings (convenience).",
    )
    cli.add_argument(
        "--model-api-key",
        default=_env("NODIFYCTX_MODEL_API_KEY", legacy_name="REPOCONTEXT_MODEL_API_KEY")
        or _env("NODIFYCTX_LMSTUDIO_API_KEY", "lm-studio", legacy_name="REPOCONTEXT_LMSTUDIO_API_KEY"),
        help="API key/token for the model service (NODIFYCTX_MODEL_API_KEY).",
    )
    cli.add_argument(
        "--model-timeout-seconds",
        type=float,
        default=_env_float("NODIFYCTX_MODEL_TIMEOUT_SECONDS", 30.0),
        help="Timeout in seconds for model service requests used during indexing and agent execution.",
    )
    cli.add_argument(
        "--model-max-retries",
        type=int,
        default=_env_non_negative_int("NODIFYCTX_MODEL_MAX_RETRIES", 2),
        help="Maximum retries per model request before failing over or exiting.",
    )
    cli.add_argument(
        "--fallback-chat-model",
        default=_env("NODIFYCTX_FALLBACK_CHAT_MODEL", legacy_name="REPOCONTEXT_FALLBACK_CHAT_MODEL"),
        help="Optional fallback chat model to use when the primary provider fails.",
    )
    cli.add_argument(
        "--fallback-chat-provider",
        default=_env("NODIFYCTX_FALLBACK_CHAT_PROVIDER", legacy_name="REPOCONTEXT_FALLBACK_CHAT_PROVIDER"),
        help="Optional fallback provider: lmstudio, openai-compatible, openai, deepseek, anthropic, or claude.",
    )
    cli.add_argument(
        "--fallback-chat-base-url",
        default=_env("NODIFYCTX_FALLBACK_CHAT_BASE_URL", legacy_name="REPOCONTEXT_FALLBACK_CHAT_BASE_URL"),
        help="Optional fallback base URL for OpenAI-compatible, OpenAI, DeepSeek, or LM Studio chat APIs.",
    )
    cli.add_argument(
        "--fallback-model-api-key",
        default=_env("NODIFYCTX_FALLBACK_MODEL_API_KEY", legacy_name="REPOCONTEXT_FALLBACK_MODEL_API_KEY"),
        help="Optional fallback API key. Defaults to the primary model API key if omitted.",
    )
    cli.add_argument("--neo4j-uri", default=_env("NODIFYCTX_NEO4J_URI", "bolt://localhost:7687", legacy_name="REPOCONTEXT_NEO4J_URI"))
    cli.add_argument("--neo4j-username", default=_env("NODIFYCTX_NEO4J_USERNAME", "neo4j", legacy_name="REPOCONTEXT_NEO4J_USERNAME"))
    cli.add_argument("--neo4j-password", default=_env("NODIFYCTX_NEO4J_PASSWORD", "neo4j", legacy_name="REPOCONTEXT_NEO4J_PASSWORD"))
    cli.add_argument("--qdrant-path", default=_env("NODIFYCTX_QDRANT_PATH", ".nodifyctx/qdrant", legacy_name="REPOCONTEXT_QDRANT_PATH"))
    cli.add_argument("--qdrant-url", default=_env("NODIFYCTX_QDRANT_URL", legacy_name="REPOCONTEXT_QDRANT_URL"))
    cli.add_argument(
        "--collection-id",
        default=_env("NODIFYCTX_COLLECTION_ID", legacy_name="REPOCONTEXT_COLLECTION_ID"),
        help=(
            "Optional repository scope ID. Defaults to a stable value derived from the repository path, "
            "so different local repos keep separate graph/vector contexts."
        ),
    )
    cli.add_argument(
        "--qdrant-collection",
        default=_env("NODIFYCTX_QDRANT_COLLECTION", legacy_name="REPOCONTEXT_QDRANT_COLLECTION"),
        help="Optional full Qdrant collection name override. Usually --collection-id is the better choice.",
    )
    cli.add_argument(
        "--lmstudio-base-url",
        default=_env("NODIFYCTX_LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1", legacy_name="REPOCONTEXT_LMSTUDIO_BASE_URL"),
        help="Legacy LM Studio base URL (kept for backward compatibility).",
    )
    cli.add_argument(
        "--lmstudio-api-key",
        default=_env("NODIFYCTX_LMSTUDIO_API_KEY", "lm-studio", legacy_name="REPOCONTEXT_LMSTUDIO_API_KEY"),
        help="Legacy LM Studio API key (kept for backward compatibility).",
    )
    cli.add_argument("--rebuild", action="store_true", help="Clear previous graph/vector data before indexing.")
    cli.add_argument("--skip-index", action="store_true", help="Use the existing index without re-parsing the repo.")
    return cli.parse_args()


def configure_environment(args: argparse.Namespace) -> None:
    repository_path = str(Path(args.repository).resolve())
    requested_chat_provider = _resolve_chat_provider(
        getattr(args, "chat_provider", None),
        getattr(args, "chat_base_url", None) or getattr(args, "model_base_url", None),
        args.chat_model,
    )

    os.environ["NODIFYCTX_REPOSITORY_PATH"] = repository_path
    os.environ["NODIFYCTX_EMBEDDING_MODEL"] = args.embedding_model
    os.environ["NODIFYCTX_MODEL_TIMEOUT_SECONDS"] = str(args.model_timeout_seconds)
    os.environ["NODIFYCTX_MODEL_MAX_RETRIES"] = str(getattr(args, "model_max_retries", 2))
    os.environ["NODIFYCTX_NEO4J_URI"] = args.neo4j_uri
    os.environ["NODIFYCTX_NEO4J_USERNAME"] = args.neo4j_username
    os.environ["NODIFYCTX_NEO4J_PASSWORD"] = args.neo4j_password
    os.environ["NODIFYCTX_QDRANT_PATH"] = args.qdrant_path
    if args.qdrant_url:
        os.environ["NODIFYCTX_QDRANT_URL"] = args.qdrant_url
    else:
        os.environ.pop("NODIFYCTX_QDRANT_URL", None)
    if args.collection_id:
        os.environ["NODIFYCTX_COLLECTION_ID"] = args.collection_id
    else:
        os.environ.pop("NODIFYCTX_COLLECTION_ID", None)
    if args.qdrant_collection:
        os.environ["NODIFYCTX_QDRANT_COLLECTION"] = args.qdrant_collection
    else:
        os.environ.pop("NODIFYCTX_QDRANT_COLLECTION", None)
    if getattr(args, "chat_provider", None):
        os.environ["NODIFYCTX_CHAT_PROVIDER"] = requested_chat_provider
    else:
        os.environ.pop("NODIFYCTX_CHAT_PROVIDER", None)
    if getattr(args, "model_base_url", None):
        chat_url = embeddings_url = args.model_base_url
    else:
        chat_url = args.chat_base_url or _env("NODIFYCTX_CHAT_BASE_URL", legacy_name="REPOCONTEXT_CHAT_BASE_URL") or _default_chat_base_url(requested_chat_provider)
        embeddings_url = args.embeddings_base_url or args.lmstudio_base_url or _env("NODIFYCTX_EMBEDDINGS_BASE_URL", legacy_name="REPOCONTEXT_EMBEDDINGS_BASE_URL") or _env("NODIFYCTX_LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1", legacy_name="REPOCONTEXT_LMSTUDIO_BASE_URL")

    if chat_url:
        os.environ["NODIFYCTX_CHAT_BASE_URL"] = chat_url
    else:
        os.environ.pop("NODIFYCTX_CHAT_BASE_URL", None)
    os.environ["NODIFYCTX_EMBEDDINGS_BASE_URL"] = embeddings_url
    if getattr(args, "model_base_url", None):
        os.environ["NODIFYCTX_MODEL_BASE_URL"] = args.model_base_url
    else:
        if chat_url and chat_url == embeddings_url:
            os.environ["NODIFYCTX_MODEL_BASE_URL"] = chat_url
        else:
            os.environ.pop("NODIFYCTX_MODEL_BASE_URL", None)

    model_api_key = _resolve_provider_api_key(
        requested_chat_provider,
        args.model_api_key or args.lmstudio_api_key,
    )
    os.environ["NODIFYCTX_MODEL_API_KEY"] = model_api_key
    lmstudio_base_url = getattr(args, "lmstudio_base_url", None) or (chat_url if requested_chat_provider in {"lmstudio", "openai-compatible"} else embeddings_url)
    if lmstudio_base_url:
        os.environ["NODIFYCTX_LMSTUDIO_BASE_URL"] = lmstudio_base_url
    else:
        os.environ.pop("NODIFYCTX_LMSTUDIO_BASE_URL", None)
    os.environ["NODIFYCTX_LMSTUDIO_API_KEY"] = model_api_key
    if args.chat_model:
        os.environ["NODIFYCTX_CHAT_MODEL"] = args.chat_model
    fallback_chat_model = getattr(args, "fallback_chat_model", None)
    if fallback_chat_model:
        os.environ["NODIFYCTX_FALLBACK_CHAT_MODEL"] = fallback_chat_model
    else:
        os.environ.pop("NODIFYCTX_FALLBACK_CHAT_MODEL", None)
    fallback_chat_provider = getattr(args, "fallback_chat_provider", None)
    if fallback_chat_provider:
        os.environ["NODIFYCTX_FALLBACK_CHAT_PROVIDER"] = _resolve_chat_provider(
            fallback_chat_provider,
            getattr(args, "fallback_chat_base_url", None),
            fallback_chat_model,
        )
    else:
        os.environ.pop("NODIFYCTX_FALLBACK_CHAT_PROVIDER", None)
    fallback_chat_base_url = getattr(args, "fallback_chat_base_url", None)
    if fallback_chat_base_url:
        os.environ["NODIFYCTX_FALLBACK_CHAT_BASE_URL"] = fallback_chat_base_url
    else:
        os.environ.pop("NODIFYCTX_FALLBACK_CHAT_BASE_URL", None)
    fallback_model_api_key = getattr(args, "fallback_model_api_key", None)
    if fallback_model_api_key:
        os.environ["NODIFYCTX_FALLBACK_MODEL_API_KEY"] = fallback_model_api_key
    else:
        os.environ.pop("NODIFYCTX_FALLBACK_MODEL_API_KEY", None)

    legacy_pairs = {
        "REPOCONTEXT_REPOSITORY_PATH": "NODIFYCTX_REPOSITORY_PATH",
        "REPOCONTEXT_EMBEDDING_MODEL": "NODIFYCTX_EMBEDDING_MODEL",
        "REPOCONTEXT_NEO4J_URI": "NODIFYCTX_NEO4J_URI",
        "REPOCONTEXT_NEO4J_USERNAME": "NODIFYCTX_NEO4J_USERNAME",
        "REPOCONTEXT_NEO4J_PASSWORD": "NODIFYCTX_NEO4J_PASSWORD",
        "REPOCONTEXT_QDRANT_PATH": "NODIFYCTX_QDRANT_PATH",
        "REPOCONTEXT_CHAT_BASE_URL": "NODIFYCTX_CHAT_BASE_URL",
        "REPOCONTEXT_EMBEDDINGS_BASE_URL": "NODIFYCTX_EMBEDDINGS_BASE_URL",
        "REPOCONTEXT_MODEL_API_KEY": "NODIFYCTX_MODEL_API_KEY",
        "REPOCONTEXT_MODEL_TIMEOUT_SECONDS": "NODIFYCTX_MODEL_TIMEOUT_SECONDS",
        "REPOCONTEXT_MODEL_MAX_RETRIES": "NODIFYCTX_MODEL_MAX_RETRIES",
        "REPOCONTEXT_LMSTUDIO_BASE_URL": "NODIFYCTX_LMSTUDIO_BASE_URL",
        "REPOCONTEXT_LMSTUDIO_API_KEY": "NODIFYCTX_LMSTUDIO_API_KEY",
        "REPOCONTEXT_CHAT_MODEL": "NODIFYCTX_CHAT_MODEL",
        "REPOCONTEXT_CHAT_PROVIDER": "NODIFYCTX_CHAT_PROVIDER",
    }
    optional_legacy_pairs = {
        "REPOCONTEXT_MODEL_BASE_URL": "NODIFYCTX_MODEL_BASE_URL",
        "REPOCONTEXT_QDRANT_URL": "NODIFYCTX_QDRANT_URL",
        "REPOCONTEXT_COLLECTION_ID": "NODIFYCTX_COLLECTION_ID",
        "REPOCONTEXT_QDRANT_COLLECTION": "NODIFYCTX_QDRANT_COLLECTION",
        "REPOCONTEXT_FALLBACK_CHAT_MODEL": "NODIFYCTX_FALLBACK_CHAT_MODEL",
        "REPOCONTEXT_FALLBACK_CHAT_PROVIDER": "NODIFYCTX_FALLBACK_CHAT_PROVIDER",
        "REPOCONTEXT_FALLBACK_CHAT_BASE_URL": "NODIFYCTX_FALLBACK_CHAT_BASE_URL",
        "REPOCONTEXT_FALLBACK_MODEL_API_KEY": "NODIFYCTX_FALLBACK_MODEL_API_KEY",
    }
    for legacy_name, current_name in legacy_pairs.items():
        os.environ[legacy_name] = os.environ[current_name]
    for legacy_name, current_name in optional_legacy_pairs.items():
        current_value = os.environ.get(current_name)
        if current_value is None:
            os.environ.pop(legacy_name, None)
        else:
            os.environ[legacy_name] = current_value


def build_index_if_requested(args: argparse.Namespace) -> None:
    if args.skip_index:
        return

    repository_path = Path(args.repository).resolve()
    with KnowledgeGraphBuilder(
        GraphStoreConfig(
            repository_path=str(repository_path),
            collection_id=args.collection_id,
            qdrant_collection=args.qdrant_collection,
        )
    ) as builder:
        summary = builder.index_repository(repository_path, rebuild=args.rebuild)
    print(
        "[index]\n"
        f"Indexed {summary['entities_indexed']} entities and {summary['relationships_indexed']} relationships "
        f"from {summary['repository_path']} into scope '{summary['collection_id']}' "
        f"(Qdrant: {summary['qdrant_collection']}).\n"
    )


def main() -> None:
    args = parse_args()
    configure_environment(args)

    if not args.chat_model:
        raise ValueError("Pass --chat-model or set NODIFYCTX_CHAT_MODEL before starting the agent.")

    try:
        build_index_if_requested(args)
    except EmbeddingsServiceError as exc:
        raise SystemExit(str(exc)) from exc

    try:
        agent = NodifyCtxAgent(AgentRuntimeConfig(chat_model=args.chat_model))
    except (ModelServiceError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    try:
        if args.question:
            try:
                agent.run(args.question)
            except ModelServiceError as exc:
                raise SystemExit(str(exc)) from exc
            return

        while True:
            try:
                query = input(">> ").strip()
            except EOFError:
                print()
                break

            if query.lower() in {"exit", "quit"}:
                break
            if not query:
                continue

            try:
                agent.run(query)
            except ModelServiceError as exc:
                print(str(exc))
    finally:
        close_runtime()
