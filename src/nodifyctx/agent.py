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


@dataclass(slots=True)
class AgentRuntimeConfig:
    chat_model: str
    chat_base_url: str = field(
        default_factory=lambda: os.getenv(
            "NODIFYCTX_CHAT_BASE_URL",
            _env("REPOCONTEXT_CHAT_BASE_URL", legacy_name="NODIFYCTX_LMSTUDIO_BASE_URL")
            or _env("NODIFYCTX_LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1", legacy_name="REPOCONTEXT_LMSTUDIO_BASE_URL"),
        )
    )
    embeddings_base_url: str = field(
        default_factory=lambda: os.getenv(
            "NODIFYCTX_EMBEDDINGS_BASE_URL",
            _env("REPOCONTEXT_EMBEDDINGS_BASE_URL", legacy_name="NODIFYCTX_LMSTUDIO_BASE_URL")
            or _env("NODIFYCTX_LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1", legacy_name="REPOCONTEXT_LMSTUDIO_BASE_URL"),
        )
    )
    model_api_key: str = field(
        default_factory=lambda: os.getenv(
            "NODIFYCTX_MODEL_API_KEY",
            _env("REPOCONTEXT_MODEL_API_KEY", legacy_name="NODIFYCTX_LMSTUDIO_API_KEY")
            or _env("NODIFYCTX_LMSTUDIO_API_KEY", "lm-studio", legacy_name="REPOCONTEXT_LMSTUDIO_API_KEY"),
        )
    )
    model_timeout_seconds: float = field(default_factory=lambda: _env_float("NODIFYCTX_MODEL_TIMEOUT_SECONDS", 10.0))
    temperature: float = 0.0
    max_tool_iterations: int = field(default_factory=lambda: _env_int("NODIFYCTX_MAX_TOOL_ITERATIONS", 10))
    graph_recursion_limit: int = field(default_factory=lambda: _env_recursion_limit("NODIFYCTX_GRAPH_RECURSION_LIMIT", 100))


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
        self.model = ChatOpenAI(
            model=self.config.chat_model,
            base_url=self.config.chat_base_url,
            api_key=self.config.model_api_key,
            timeout=self.config.model_timeout_seconds,
            temperature=self.config.temperature,
        )
        self.tool_model = self.model.bind_tools(self.tools)
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
            response = self.model.invoke(
                state["messages"]
                + [
                    SystemMessage(
                        content=(
                            "Tool budget reached. Do not call more tools. "
                            "Provide the best final answer now in plain text using only the existing tool outputs."
                        )
                    )
                ]
            )
            response = self._finalize_budget_response(state, response)
        else:
            response = self.tool_model.invoke(state["messages"])
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

        retry_response = self.model.invoke(
            state["messages"]
            + [
                SystemMessage(
                    content=(
                        "Answer now in plain text only. Do not request tools. Do not emit XML, <tool_call>, "
                        "or reasoning tags. Use only the evidence already gathered."
                    )
                )
            ]
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


def parse_args() -> argparse.Namespace:
    cli = argparse.ArgumentParser(description="Run the NodifyCtx local agent.")
    cli.add_argument("repository", nargs="?", default=".")
    cli.add_argument("--question", help="Ask one question non-interactively and exit.")
    cli.add_argument("--chat-model", default=_env("NODIFYCTX_CHAT_MODEL", legacy_name="REPOCONTEXT_CHAT_MODEL"))
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
        default=_env_float("NODIFYCTX_MODEL_TIMEOUT_SECONDS", 10.0),
        help="Timeout in seconds for model service requests used during indexing and agent execution.",
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

    os.environ["NODIFYCTX_REPOSITORY_PATH"] = repository_path
    os.environ["NODIFYCTX_EMBEDDING_MODEL"] = args.embedding_model
    os.environ["NODIFYCTX_MODEL_TIMEOUT_SECONDS"] = str(args.model_timeout_seconds)
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
    if getattr(args, "model_base_url", None):
        chat_url = embeddings_url = args.model_base_url
    else:
        chat_url = args.chat_base_url or args.lmstudio_base_url or _env("NODIFYCTX_CHAT_BASE_URL", legacy_name="REPOCONTEXT_CHAT_BASE_URL") or _env("NODIFYCTX_LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1", legacy_name="REPOCONTEXT_LMSTUDIO_BASE_URL")
        embeddings_url = args.embeddings_base_url or args.lmstudio_base_url or _env("NODIFYCTX_EMBEDDINGS_BASE_URL", legacy_name="REPOCONTEXT_EMBEDDINGS_BASE_URL") or _env("NODIFYCTX_LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1", legacy_name="REPOCONTEXT_LMSTUDIO_BASE_URL")

    os.environ["NODIFYCTX_CHAT_BASE_URL"] = chat_url
    os.environ["NODIFYCTX_EMBEDDINGS_BASE_URL"] = embeddings_url
    if getattr(args, "model_base_url", None):
        os.environ["NODIFYCTX_MODEL_BASE_URL"] = args.model_base_url
    else:
        if chat_url == embeddings_url:
            os.environ["NODIFYCTX_MODEL_BASE_URL"] = chat_url
        else:
            os.environ.pop("NODIFYCTX_MODEL_BASE_URL", None)

    model_api_key = args.model_api_key or args.lmstudio_api_key or _env("NODIFYCTX_MODEL_API_KEY", legacy_name="REPOCONTEXT_MODEL_API_KEY") or _env("NODIFYCTX_LMSTUDIO_API_KEY", "lm-studio", legacy_name="REPOCONTEXT_LMSTUDIO_API_KEY")
    os.environ["NODIFYCTX_MODEL_API_KEY"] = model_api_key
    os.environ["NODIFYCTX_LMSTUDIO_BASE_URL"] = chat_url
    os.environ["NODIFYCTX_LMSTUDIO_API_KEY"] = model_api_key
    if args.chat_model:
        os.environ["NODIFYCTX_CHAT_MODEL"] = args.chat_model

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
        "REPOCONTEXT_LMSTUDIO_BASE_URL": "NODIFYCTX_LMSTUDIO_BASE_URL",
        "REPOCONTEXT_LMSTUDIO_API_KEY": "NODIFYCTX_LMSTUDIO_API_KEY",
        "REPOCONTEXT_CHAT_MODEL": "NODIFYCTX_CHAT_MODEL",
    }
    optional_legacy_pairs = {
        "REPOCONTEXT_MODEL_BASE_URL": "NODIFYCTX_MODEL_BASE_URL",
        "REPOCONTEXT_QDRANT_URL": "NODIFYCTX_QDRANT_URL",
        "REPOCONTEXT_COLLECTION_ID": "NODIFYCTX_COLLECTION_ID",
        "REPOCONTEXT_QDRANT_COLLECTION": "NODIFYCTX_QDRANT_COLLECTION",
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

    agent = NodifyCtxAgent(AgentRuntimeConfig(chat_model=args.chat_model))

    try:
        if args.question:
            agent.run(args.question)
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

            agent.run(query)
    finally:
        close_runtime()
