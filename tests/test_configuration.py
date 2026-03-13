import argparse
import os
from pathlib import Path

from nodifyctx.agent import AgentRuntimeConfig, configure_environment
from nodifyctx.graph_builder import DEFAULT_QDRANT_COLLECTION_PREFIX, GraphStoreConfig, build_qdrant_collection_name


def test_graph_store_config_uses_nodifyctx_defaults(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("NODIFYCTX_QDRANT_PATH", raising=False)
    monkeypatch.delenv("REPOCONTEXT_QDRANT_PATH", raising=False)
    monkeypatch.setenv("NODIFYCTX_REPOSITORY_PATH", str(tmp_path))
    monkeypatch.setenv("NODIFYCTX_COLLECTION_ID", "My Scope")

    config = GraphStoreConfig()

    assert config.repository_path == str(tmp_path.resolve())
    assert config.collection_id == "my_scope"
    assert config.qdrant_path == ".nodifyctx/qdrant"
    assert DEFAULT_QDRANT_COLLECTION_PREFIX == "nodify_ctx_nodes"
    assert build_qdrant_collection_name(config.collection_id) == "nodify_ctx_nodes__my_scope"


def test_agent_runtime_config_accepts_legacy_environment(monkeypatch) -> None:
    monkeypatch.delenv("NODIFYCTX_CHAT_BASE_URL", raising=False)
    monkeypatch.delenv("NODIFYCTX_MODEL_API_KEY", raising=False)
    monkeypatch.delenv("NODIFYCTX_MODEL_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setenv("REPOCONTEXT_CHAT_BASE_URL", "http://legacy-chat")
    monkeypatch.setenv("REPOCONTEXT_MODEL_API_KEY", "legacy-key")
    monkeypatch.setenv("REPOCONTEXT_MODEL_TIMEOUT_SECONDS", "3.5")

    config = AgentRuntimeConfig(chat_model="local-model")

    assert config.chat_base_url == "http://legacy-chat"
    assert config.model_api_key == "legacy-key"
    assert config.model_timeout_seconds == 3.5


def test_configure_environment_sets_new_and_legacy_names(tmp_path: Path, monkeypatch) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    args = argparse.Namespace(
        repository=str(repository),
        question=None,
        chat_model="chat-model",
        embedding_model="embed-model",
        chat_base_url="http://chat",
        embeddings_base_url="http://embed",
        model_base_url=None,
        model_api_key="model-key",
        model_timeout_seconds=7.0,
        neo4j_uri="bolt://db",
        neo4j_username="neo",
        neo4j_password="secret",
        qdrant_path=".nodifyctx/qdrant",
        qdrant_url="http://qdrant",
        collection_id="scope-a",
        qdrant_collection="nodify_ctx_nodes__scope-a",
        lmstudio_base_url="http://lmstudio",
        lmstudio_api_key="lm-key",
        rebuild=False,
        skip_index=True,
    )

    for name in list(os.environ):
        if name.startswith("NODIFYCTX_") or name.startswith("REPOCONTEXT_"):
            monkeypatch.delenv(name, raising=False)

    configure_environment(args)

    expected_repository = str(repository.resolve())
    assert os.environ["NODIFYCTX_REPOSITORY_PATH"] == expected_repository
    assert os.environ["NODIFYCTX_CHAT_MODEL"] == "chat-model"
    assert os.environ["NODIFYCTX_MODEL_TIMEOUT_SECONDS"] == "7.0"
    assert os.environ["NODIFYCTX_QDRANT_COLLECTION"] == "nodify_ctx_nodes__scope-a"
    assert os.environ["REPOCONTEXT_REPOSITORY_PATH"] == expected_repository
    assert os.environ["REPOCONTEXT_CHAT_MODEL"] == "chat-model"
    assert os.environ["REPOCONTEXT_MODEL_TIMEOUT_SECONDS"] == "7.0"
    assert os.environ["REPOCONTEXT_QDRANT_COLLECTION"] == "nodify_ctx_nodes__scope-a"