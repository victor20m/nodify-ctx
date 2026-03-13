from pathlib import Path
from types import SimpleNamespace

import pytest

from nodifyctx import graph_builder as gb


class RecordingSession:
    def __init__(self):
        self.calls = []

    def run(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class RecordingNeo4j:
    def __init__(self):
        self.session_instance = RecordingSession()
        self.closed = False

    def session(self):
        return self.session_instance

    def close(self):
        self.closed = True


class RecordingQdrant:
    def __init__(self, *, exists=False):
        self.exists = exists
        self.created = []
        self.deleted = []
        self.upserts = []
        self.closed = False

    def collection_exists(self, name):
        return self.exists

    def create_collection(self, **kwargs):
        self.created.append(kwargs)
        self.exists = True

    def delete_collection(self, name):
        self.deleted.append(name)

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)

    def close(self):
        self.closed = True


def _builder(config=None):
    builder = gb.KnowledgeGraphBuilder.__new__(gb.KnowledgeGraphBuilder)
    builder.config = config or gb.GraphStoreConfig(repository_path=".", collection_id="scope-a", qdrant_collection="collection-a")
    builder.qdrant = RecordingQdrant()
    builder.neo4j = RecordingNeo4j()
    builder.embedding_client = SimpleNamespace()
    return builder


def test_collection_helpers_normalize_values(tmp_path: Path):
    collection_id = gb.build_collection_id(tmp_path / "My Repo")

    assert gb.sanitize_collection_id(" My Scope ") == "my_scope"
    assert collection_id.startswith("my_repo_")
    assert gb.build_qdrant_collection_name("scope") == "nodify_ctx_nodes__scope"


def test_sanitize_collection_id_requires_alphanumeric():
    with pytest.raises(ValueError):
        gb.sanitize_collection_id("___")


def test_prepare_qdrant_path_creates_directory(tmp_path: Path):
    target = tmp_path / "data" / "qdrant"
    builder = _builder(SimpleNamespace(qdrant_url=None, qdrant_path=str(target)))

    gb.KnowledgeGraphBuilder._prepare_qdrant_path(builder)

    assert target.exists()


def test_prepare_qdrant_path_skips_for_url_and_memory(tmp_path: Path):
    url_builder = _builder(SimpleNamespace(qdrant_url="http://qdrant", qdrant_path=str(tmp_path / "ignored")))
    memory_builder = _builder(SimpleNamespace(qdrant_url=None, qdrant_path=":memory:"))

    gb.KnowledgeGraphBuilder._prepare_qdrant_path(url_builder)
    gb.KnowledgeGraphBuilder._prepare_qdrant_path(memory_builder)

    assert not (tmp_path / "ignored").exists()


def test_build_qdrant_client_selects_url_or_path(monkeypatch):
    captured = []

    class FakeClient:
        def __init__(self, **kwargs):
            captured.append(kwargs)

    monkeypatch.setattr(gb, "QdrantClient", FakeClient)

    url_builder = _builder(SimpleNamespace(qdrant_url="http://qdrant", qdrant_path="ignored"))
    path_builder = _builder(SimpleNamespace(qdrant_url=None, qdrant_path="local/path"))

    gb.KnowledgeGraphBuilder._build_qdrant_client(url_builder)
    gb.KnowledgeGraphBuilder._build_qdrant_client(path_builder)

    assert captured == [{"url": "http://qdrant"}, {"path": "local/path"}]


def test_graph_schema_and_mutation_helpers_issue_queries():
    builder = _builder()

    gb.KnowledgeGraphBuilder._ensure_graph_schema(builder)
    gb.KnowledgeGraphBuilder._clear_graph(builder)
    gb.KnowledgeGraphBuilder._upsert_entity_node(
        builder,
        builder.neo4j.session_instance,
        {
            "id": "1",
            "name": "node",
            "type": "file",
            "file_path": "pkg/service.py",
            "qualified_name": "pkg.service",
            "start_line": 1,
            "end_line": 3,
            "raw_code": "code",
        },
    )
    gb.KnowledgeGraphBuilder._upsert_relationship(
        builder,
        builder.neo4j.session_instance,
        {"type": "CONTAINS", "source_id": "1", "target_id": "2"},
    )

    queries = [query for query, _ in builder.neo4j.session_instance.calls]
    assert any("DROP CONSTRAINT repo_context_node_id IF EXISTS" in query for query in queries)
    assert any("CREATE CONSTRAINT nodify_ctx_node_repo_id" in query for query in queries)
    assert any("DROP INDEX repo_context_node_name IF EXISTS" in query for query in queries)
    assert any("DETACH DELETE node" in query for query in queries)
    assert any("MERGE (node:CodeEntity {repo_id: $repo_id, node_id: $node_id})" in query for query in queries)
    assert any("SET node:File" in query for query in queries)
    assert any("MERGE (source)-[edge:CONTAINS]->(target)" in query for query in queries)


def test_collection_helpers_create_and_recreate_collection():
    builder = _builder()

    gb.KnowledgeGraphBuilder._ensure_collection(builder, vector_size=2)
    assert builder.qdrant.created[0]["collection_name"] == "collection-a"

    builder.qdrant.exists = True
    gb.KnowledgeGraphBuilder._recreate_collection(builder)
    assert builder.qdrant.deleted == ["collection-a"]


def test_ingest_indexes_entities_and_relationships(monkeypatch):
    builder = _builder()
    builder.config = SimpleNamespace(collection_id="scope-a", qdrant_collection="collection-a")
    builder.embed_text = lambda text: [float(len(text))]

    summary = gb.KnowledgeGraphBuilder.ingest(
        builder,
        {
            "entities": [
                {"id": "1", "name": "root", "type": "folder", "file_path": ".", "qualified_name": "repo", "start_line": 0, "end_line": 0, "raw_code": "Folder root"},
                {"id": "2", "name": "file.py", "type": "file", "file_path": "file.py", "qualified_name": "file.py", "start_line": 1, "end_line": 1, "raw_code": "print('x')"},
            ],
            "relationships": [{"type": "CONTAINS", "source_id": "1", "target_id": "2"}],
        },
        rebuild=True,
    )

    assert summary == {
        "entities_indexed": 2,
        "relationships_indexed": 1,
        "collection_id": "scope-a",
        "qdrant_collection": "collection-a",
    }
    assert len(builder.qdrant.upserts) == 1
    assert len(builder.qdrant.upserts[0]["points"]) == 2


def test_ingest_uses_metadata_fallback_for_empty_raw_code():
    builder = _builder()
    builder.config = SimpleNamespace(collection_id="scope-a", qdrant_collection="collection-a")
    embedded_inputs = []
    builder.embed_text = lambda text: embedded_inputs.append(text) or [1.0, 2.0]

    gb.KnowledgeGraphBuilder.ingest(
        builder,
        {
            "entities": [
                {"id": "1", "name": "empty.py", "type": "file", "file_path": "empty.py", "qualified_name": "empty.py", "start_line": 0, "end_line": 0, "raw_code": "   "},
            ],
            "relationships": [],
        },
        rebuild=False,
    )

    assert embedded_inputs == ["file empty.py in empty.py"]


def test_embed_text_validates_and_reads_embedding_payload():
    builder = _builder(SimpleNamespace(embedding_model="embed-model"))
    builder.embedding_client = SimpleNamespace(
        embeddings=SimpleNamespace(create=lambda **kwargs: SimpleNamespace(data=[SimpleNamespace(embedding=[0.3, 0.4])]))
    )

    with pytest.raises(ValueError):
        gb.KnowledgeGraphBuilder.embed_text(builder, "   ")

    assert gb.KnowledgeGraphBuilder.embed_text(builder, "hello") == [0.3, 0.4]


def test_embed_text_wraps_embeddings_service_failures(monkeypatch):
    class FakeTimeoutError(Exception):
        pass

    monkeypatch.setattr(gb, "APITimeoutError", FakeTimeoutError)
    builder = _builder(
        SimpleNamespace(
            embeddings_base_url="http://10.5.0.2:1234/v1",
            embedding_model="embed-model",
        )
    )
    builder.embedding_client = SimpleNamespace(
        embeddings=SimpleNamespace(create=lambda **kwargs: (_ for _ in ()).throw(FakeTimeoutError("timed out")))
    )

    with pytest.raises(gb.EmbeddingsServiceError) as exc_info:
        gb.KnowledgeGraphBuilder.embed_text(builder, "hello")

    assert "Neo4j connectivity was verified successfully" in str(exc_info.value)
    assert "NODIFYCTX_EMBEDDINGS_BASE_URL" in str(exc_info.value)


def test_embedding_text_for_entity_prefers_raw_code_and_falls_back_to_metadata():
    assert gb.KnowledgeGraphBuilder._embedding_text_for_entity({"raw_code": "print('x')", "type": "file"}) == "print('x')"
    assert gb.KnowledgeGraphBuilder._embedding_text_for_entity(
        {"raw_code": "   ", "type": "file", "qualified_name": "pkg.empty", "file_path": "pkg/empty.py"}
    ) == "file pkg.empty in pkg/empty.py"


def test_close_and_context_manager_close_clients():
    builder = _builder()

    gb.KnowledgeGraphBuilder.close(builder)
    assert builder.qdrant.closed is True
    assert builder.neo4j.closed is True


def test_neo4j_label_rejects_unknown_entity_type():
    with pytest.raises(ValueError):
        gb.KnowledgeGraphBuilder._neo4j_label("unknown")