from types import SimpleNamespace

import pytest

from nodifyctx import tools


class FakeSession:
    def __init__(self, records):
        self.records = records
        self.calls = []

    def run(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return self.records

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class FakeNeo4j:
    def __init__(self, records):
        self._session = FakeSession(records)
        self.closed = False

    def session(self):
        return self._session

    def close(self):
        self.closed = True


class FakeQdrant:
    def __init__(self, *, exists=True, points=None):
        self.exists = exists
        self.points = points or []
        self.queries = []
        self.closed = False

    def collection_exists(self, name):
        self.queries.append(("collection_exists", name))
        return self.exists

    def query_points(self, **kwargs):
        self.queries.append(("query_points", kwargs))
        return SimpleNamespace(points=self.points)

    def close(self):
        self.closed = True


class FakeBuilder:
    def __init__(self, *, exists=True, points=None, records=None):
        self.config = SimpleNamespace(collection_id="scope-a", qdrant_collection="collection-a")
        self.qdrant = FakeQdrant(exists=exists, points=points)
        self.neo4j = FakeNeo4j(records or [])
        self.closed = False
        self.embedded = []

    def embed_text(self, text):
        self.embedded.append(text)
        return [0.1, 0.2]

    def close(self):
        self.closed = True
        self.qdrant.close()
        self.neo4j.close()


def test_semantic_search_validates_query():
    with pytest.raises(ValueError):
        tools.semantic_search("  ")


def test_semantic_search_requires_existing_collection(monkeypatch):
    monkeypatch.setattr(tools, "get_builder", lambda: FakeBuilder(exists=False))

    with pytest.raises(RuntimeError):
        tools.semantic_search("find handler")


def test_semantic_search_returns_payload_projection(monkeypatch):
    builder = FakeBuilder(
        points=[
            SimpleNamespace(payload={"name": "Greeter", "type": "class", "file_path": "pkg/service.py"}),
            SimpleNamespace(payload={"name": "helper", "type": "function", "file_path": "pkg/service.py"}),
        ]
    )
    monkeypatch.setattr(tools, "get_builder", lambda: builder)

    result = tools.semantic_search("greeter")

    assert result == [
        {"name": "Greeter", "type": "class", "file": "pkg/service.py"},
        {"name": "helper", "type": "function", "file": "pkg/service.py"},
    ]
    assert builder.embedded == ["greeter"]


@pytest.mark.parametrize(("function", "records"), [
    (tools.get_callers, [{"name": "alpha"}, {"name": "beta"}]),
    (tools.get_dependencies, [{"name": "gamma"}, {"name": "delta"}]),
])
def test_graph_lookup_tools_return_names(monkeypatch, function, records):
    monkeypatch.setattr(tools, "get_builder", lambda: FakeBuilder(records=records))

    assert function("node") == [record["name"] for record in records]


def test_inspect_code_returns_error_for_missing_node(monkeypatch):
    monkeypatch.setattr(tools, "get_builder", lambda: FakeBuilder(records=[]))

    assert tools.inspect_code("missing") == {
        "name": "missing",
        "error": "No indexed node named 'missing' was found.",
    }


def test_inspect_code_returns_ambiguity(monkeypatch):
    monkeypatch.setattr(
        tools,
        "get_builder",
        lambda: FakeBuilder(records=[{"file": "a.py", "name": "node", "code": "x"}, {"file": "b.py", "name": "node", "code": "y"}]),
    )

    result = tools.inspect_code("node")

    assert result["name"] == "node"
    assert "Multiple nodes share this name" in result["error"]
    assert result["matches"] == ["a.py", "b.py"]


def test_inspect_code_returns_single_match(monkeypatch):
    monkeypatch.setattr(
        tools,
        "get_builder",
        lambda: FakeBuilder(records=[{"file": "a.py", "name": "node", "code": "print('x')"}]),
    )

    assert tools.inspect_code("node") == {
        "name": "node",
        "file": "a.py",
        "code": "print('x')",
    }


def test_build_langchain_tools_matches_tool_functions():
    tools_list = tools.build_langchain_tools()

    assert len(tools_list) == len(tools.TOOL_FUNCTIONS)
    assert [tool.name for tool in tools_list] == [function.__name__ for function in tools.TOOL_FUNCTIONS]


def test_close_runtime_closes_cached_builder(monkeypatch):
    created = []

    class CachedBuilder(FakeBuilder):
        def __init__(self):
            super().__init__()
            created.append(self)

    tools.get_builder.cache_clear()
    monkeypatch.setattr(tools, "KnowledgeGraphBuilder", CachedBuilder)

    builder = tools.get_builder()
    tools.close_runtime()

    assert builder is created[0]
    assert created[0].closed is True
    assert tools.get_builder.cache_info().currsize == 0