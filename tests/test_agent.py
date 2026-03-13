import argparse
import sys
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END

from nodifyctx import agent as agent_mod


def _agent_instance():
    instance = agent_mod.NodifyCtxAgent.__new__(agent_mod.NodifyCtxAgent)
    instance.tool_functions = {
        "get_dependencies": lambda node_name: [f"dep:{node_name}"],
        "inspect_code": lambda node_name: {"name": node_name, "file": "a.py", "code": "pass"},
    }
    instance.model = SimpleNamespace(invoke=lambda messages: AIMessage(content="retry answer"))
    return instance


def test_env_parsers_validate_values(monkeypatch):
    monkeypatch.setenv("NODIFYCTX_MAX_TOOL_ITERATIONS", "3")
    monkeypatch.setenv("NODIFYCTX_GRAPH_RECURSION_LIMIT", "unlimited")

    assert agent_mod._env_int("NODIFYCTX_MAX_TOOL_ITERATIONS", 10) == 3
    assert agent_mod._env_recursion_limit("NODIFYCTX_GRAPH_RECURSION_LIMIT", 100) == 1_000_000

    monkeypatch.setenv("NODIFYCTX_MAX_TOOL_ITERATIONS", "0")
    with pytest.raises(ValueError):
        agent_mod._env_int("NODIFYCTX_MAX_TOOL_ITERATIONS", 10)


def test_message_helpers_extract_and_clean_content():
    messages = [SystemMessage(content="sys"), HumanMessage(content="Where is handler?")]

    assert agent_mod.NodifyCtxAgent._latest_user_query(messages) == "Where is handler?"
    assert agent_mod.NodifyCtxAgent._visible_message_content("<think>x</think> visible") == "visible"
    assert agent_mod.NodifyCtxAgent._contains_raw_tool_markup("<tool_call><function=x>") is True


def test_parse_and_coerce_tool_calls_from_text():
    instance = _agent_instance()
    text = "<tool_call><function=get_dependencies><parameter=node_name>alpha</parameter></function></tool_call>"

    parsed = instance._parse_tool_calls_from_text(text)
    coerced = instance._coerce_tool_calls(AIMessage(content=text))

    assert parsed[0]["name"] == "get_dependencies"
    assert parsed[0]["args"] == {"node_name": "alpha"}
    assert coerced.tool_calls[0]["name"] == "get_dependencies"


def test_route_after_reason_uses_tool_calls_or_ends():
    instance = _agent_instance()

    assert instance._route_after_reason({"messages": [AIMessage(content="done")], "search_results": [], "traversal_count": 0, "iteration_count": 0}) == END
    assert instance._route_after_reason({"messages": [AIMessage(content="", tool_calls=[{"name": "get_dependencies", "args": {}, "id": "1", "type": "tool_call"}])], "search_results": [], "traversal_count": 0, "iteration_count": 0}) == "execute_tools"


def test_execute_tools_blocks_inspect_until_traversal_and_records_tool_message():
    instance = _agent_instance()
    state = {
        "messages": [AIMessage(content="", tool_calls=[{"name": "inspect_code", "args": {"node_name": "alpha"}, "id": "1", "type": "tool_call"}])],
        "search_results": [],
        "traversal_count": 0,
        "iteration_count": 0,
    }

    result = instance._execute_tools(state)

    assert result["traversal_count"] == 0
    assert result["iteration_count"] == 1
    assert isinstance(result["messages"][-1], ToolMessage)
    assert "blocked" in result["messages"][-1].content


def test_execute_tools_runs_dependencies_and_increments_traversal():
    instance = _agent_instance()
    state = {
        "messages": [AIMessage(content="", tool_calls=[{"name": "get_dependencies", "args": {"node_name": "alpha"}, "id": "1", "type": "tool_call"}])],
        "search_results": [],
        "traversal_count": 0,
        "iteration_count": 2,
    }

    result = instance._execute_tools(state)

    assert result["traversal_count"] == 1
    assert result["iteration_count"] == 3
    assert result["messages"][-1].content == '["dep:alpha"]'


def test_finalize_budget_response_uses_retry_or_fallback():
    instance = _agent_instance()
    state = {"messages": [SystemMessage(content="sys")], "search_results": [], "traversal_count": 0, "iteration_count": 10}

    plain = AIMessage(content="final answer")
    assert instance._finalize_budget_response(state, plain) is plain

    retry = instance._finalize_budget_response(state, AIMessage(content="<tool_call>bad</tool_call>"))
    assert retry.content == "retry answer"

    instance.model = SimpleNamespace(invoke=lambda messages: AIMessage(content="<tool_call>still bad</tool_call>"))
    fallback = instance._finalize_budget_response(state, AIMessage(content="<tool_call>bad</tool_call>"))
    assert "Unable to complete the analysis" in fallback.content


def test_build_index_if_requested_respects_skip_and_indexes(monkeypatch, capsys, tmp_path):
    calls = []

    class FakeBuilder:
        def __init__(self, config):
            calls.append(("init", config.repository_path, config.collection_id, config.qdrant_collection))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def index_repository(self, repository_path, rebuild=False):
            calls.append(("index", str(repository_path), rebuild))
            return {
                "entities_indexed": 1,
                "relationships_indexed": 2,
                "repository_path": str(repository_path),
                "collection_id": "scope-a",
                "qdrant_collection": "collection-a",
            }

    monkeypatch.setattr(agent_mod, "KnowledgeGraphBuilder", FakeBuilder)
    args = argparse.Namespace(repository=str(tmp_path), skip_index=True, collection_id=None, qdrant_collection=None, rebuild=False)
    agent_mod.build_index_if_requested(args)
    assert calls == []

    args.skip_index = False
    args.collection_id = "scope-a"
    args.qdrant_collection = "collection-a"
    agent_mod.build_index_if_requested(args)
    output = capsys.readouterr().out
    assert calls[0][0] == "init"
    assert calls[1] == ("index", str(tmp_path), False)
    assert "Indexed 1 entities and 2 relationships" in output


def test_semantic_seed_adds_system_message(monkeypatch):
    instance = _agent_instance()
    monkeypatch.setattr(agent_mod, "semantic_search", lambda query: [{"name": "node", "type": "file", "file": "a.py"}])
    instance._print_step = lambda name, payload: None

    result = instance._semantic_seed(
        {"messages": [HumanMessage(content="find node")], "search_results": [], "traversal_count": 0, "iteration_count": 0}
    )

    assert result["search_results"] == [{"name": "node", "type": "file", "file": "a.py"}]
    assert "Initial semantic_search results are below" in result["messages"][-1].content


def test_reason_uses_tool_model_or_budget_finalize():
    instance = _agent_instance()
    instance.config = SimpleNamespace(max_tool_iterations=5)
    instance.tool_model = SimpleNamespace(invoke=lambda messages: AIMessage(content="tool answer"))
    instance._coerce_tool_calls = lambda response: response

    state = {"messages": [SystemMessage(content="sys")], "search_results": [], "traversal_count": 0, "iteration_count": 0}
    result = instance._reason(state)
    assert result["messages"][-1].content == "tool answer"

    instance.config = SimpleNamespace(max_tool_iterations=0)
    instance.model = SimpleNamespace(invoke=lambda messages: AIMessage(content="budget raw"))
    instance._finalize_budget_response = lambda state, response: AIMessage(content="final budget answer")
    result = instance._reason(state)
    assert result["messages"][-1].content == "final budget answer"


def test_run_uses_graph_invoke_and_returns_answer():
    instance = _agent_instance()
    instance.config = SimpleNamespace(graph_recursion_limit=7)
    instance.graph = SimpleNamespace(invoke=lambda initial_state, config: {"messages": [AIMessage(content="done")]} )

    assert instance.run("question") == "done"

    with pytest.raises(ValueError):
        instance.run("   ")


def test_parse_args_reads_cli_flags(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "nodifyctx",
        "repo",
        "--question",
        "What is this?",
        "--skip-index",
        "--chat-model",
        "demo-model",
    ])

    args = agent_mod.parse_args()

    assert args.repository == "repo"
    assert args.question == "What is this?"
    assert args.skip_index is True
    assert args.chat_model == "demo-model"


def test_main_runs_one_question_and_closes_runtime(monkeypatch, tmp_path):
    called = []

    args = argparse.Namespace(
        repository=str(tmp_path),
        question="What is this?",
        chat_model="demo-model",
        embedding_model="embed-model",
        chat_base_url="http://chat",
        embeddings_base_url="http://embed",
        model_base_url=None,
        model_api_key="key",
        model_timeout_seconds=5.0,
        neo4j_uri="bolt://db",
        neo4j_username="neo4j",
        neo4j_password="secret",
        qdrant_path=".nodifyctx/qdrant",
        qdrant_url=None,
        collection_id=None,
        qdrant_collection=None,
        lmstudio_base_url="http://chat",
        lmstudio_api_key="key",
        rebuild=False,
        skip_index=True,
    )

    class FakeAgent:
        def __init__(self, config):
            called.append(("init", config.chat_model))

        def run(self, query):
            called.append(("run", query))

    monkeypatch.setattr(agent_mod, "parse_args", lambda: args)
    monkeypatch.setattr(agent_mod, "configure_environment", lambda parsed_args: called.append(("configure", parsed_args.repository)))
    monkeypatch.setattr(agent_mod, "build_index_if_requested", lambda parsed_args: called.append(("index", parsed_args.skip_index)))
    monkeypatch.setattr(agent_mod, "NodifyCtxAgent", FakeAgent)
    monkeypatch.setattr(agent_mod, "close_runtime", lambda: called.append(("close", None)))

    agent_mod.main()

    assert called == [
        ("configure", str(tmp_path)),
        ("index", True),
        ("init", "demo-model"),
        ("run", "What is this?"),
        ("close", None),
    ]


def test_main_exits_cleanly_when_embeddings_service_is_unreachable(monkeypatch, tmp_path):
    args = argparse.Namespace(
        repository=str(tmp_path),
        question=None,
        chat_model="demo-model",
        embedding_model="embed-model",
        chat_base_url="http://chat",
        embeddings_base_url="http://embed",
        model_base_url=None,
        model_api_key="key",
        model_timeout_seconds=5.0,
        neo4j_uri="bolt://db",
        neo4j_username="neo4j",
        neo4j_password="secret",
        qdrant_path=".nodifyctx/qdrant",
        qdrant_url=None,
        collection_id=None,
        qdrant_collection=None,
        lmstudio_base_url="http://chat",
        lmstudio_api_key="key",
        rebuild=False,
        skip_index=False,
    )

    monkeypatch.setattr(agent_mod, "parse_args", lambda: args)
    monkeypatch.setattr(agent_mod, "configure_environment", lambda parsed_args: None)
    monkeypatch.setattr(
        agent_mod,
        "build_index_if_requested",
        lambda parsed_args: (_ for _ in ()).throw(agent_mod.EmbeddingsServiceError("embedding endpoint unreachable")),
    )

    with pytest.raises(SystemExit, match="embedding endpoint unreachable"):
        agent_mod.main()