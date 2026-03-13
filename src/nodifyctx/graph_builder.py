from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import re
from typing import Any

from neo4j import GraphDatabase
from openai import APIConnectionError, APITimeoutError, OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from .parser import CodeParser


DEFAULT_QDRANT_COLLECTION_PREFIX = "nodify_ctx_nodes"
_COLLECTION_SEGMENT_PATTERN = re.compile(r"[^a-z0-9]+")


class EmbeddingsServiceError(RuntimeError):
    """Raised when the configured embeddings endpoint cannot be reached."""


def _env(name: str, default: str | None = None, *, legacy_name: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is not None:
        return value
    if legacy_name is not None:
        legacy_value = os.getenv(legacy_name)
        if legacy_value is not None:
            return legacy_value
    return default


def _env_float(name: str, default: float, *, legacy_name: str | None = None) -> float:
    raw_value = _env(name, legacy_name=legacy_name)
    if raw_value is None or not raw_value.strip():
        return default

    value = float(raw_value)
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0.")
    return value


def _normalize_repository_path(repository_path: str | Path) -> str:
    return str(Path(repository_path).resolve()).replace("\\", "/").lower()


def sanitize_collection_id(value: str) -> str:
    normalized = _COLLECTION_SEGMENT_PATTERN.sub("_", value.strip().lower()).strip("_")
    if not normalized:
        raise ValueError("Collection ID must contain at least one letter or number.")
    return normalized[:40]


def build_collection_id(repository_path: str | Path) -> str:
    repository_root = Path(repository_path).resolve()
    slug = sanitize_collection_id(repository_root.name or "repo")
    digest = hashlib.sha1(_normalize_repository_path(repository_root).encode("utf8")).hexdigest()[:10]
    return f"{slug}_{digest}"


def build_qdrant_collection_name(collection_id: str) -> str:
    return f"{DEFAULT_QDRANT_COLLECTION_PREFIX}__{collection_id}"


@dataclass(slots=True)
class GraphStoreConfig:
    repository_path: str = field(default_factory=lambda: _env("NODIFYCTX_REPOSITORY_PATH", ".", legacy_name="REPOCONTEXT_REPOSITORY_PATH") or ".")
    collection_id: str | None = field(default_factory=lambda: _env("NODIFYCTX_COLLECTION_ID", legacy_name="REPOCONTEXT_COLLECTION_ID") or None)
    neo4j_uri: str = field(default_factory=lambda: _env("NODIFYCTX_NEO4J_URI", "bolt://localhost:7687", legacy_name="REPOCONTEXT_NEO4J_URI") or "bolt://localhost:7687")
    neo4j_username: str = field(default_factory=lambda: _env("NODIFYCTX_NEO4J_USERNAME", "neo4j", legacy_name="REPOCONTEXT_NEO4J_USERNAME") or "neo4j")
    neo4j_password: str = field(default_factory=lambda: _env("NODIFYCTX_NEO4J_PASSWORD", "neo4j", legacy_name="REPOCONTEXT_NEO4J_PASSWORD") or "neo4j")
    qdrant_url: str | None = field(default_factory=lambda: _env("NODIFYCTX_QDRANT_URL", legacy_name="REPOCONTEXT_QDRANT_URL") or None)
    qdrant_path: str = field(default_factory=lambda: _env("NODIFYCTX_QDRANT_PATH", ".nodifyctx/qdrant", legacy_name="REPOCONTEXT_QDRANT_PATH") or ".nodifyctx/qdrant")
    qdrant_collection: str | None = field(default_factory=lambda: _env("NODIFYCTX_QDRANT_COLLECTION", legacy_name="REPOCONTEXT_QDRANT_COLLECTION") or None)
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
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "NODIFYCTX_EMBEDDING_MODEL",
            _env("REPOCONTEXT_EMBEDDING_MODEL") or "text-embedding-nomic-embed-text-v1.5",
        )
    )
    model_timeout_seconds: float = field(
        default_factory=lambda: _env_float(
            "NODIFYCTX_MODEL_TIMEOUT_SECONDS",
            10.0,
            legacy_name="REPOCONTEXT_MODEL_TIMEOUT_SECONDS",
        )
    )

    def __post_init__(self) -> None:
        self.repository_path = str(Path(self.repository_path).resolve())
        if self.collection_id and self.collection_id.strip():
            self.collection_id = sanitize_collection_id(self.collection_id)
        else:
            self.collection_id = build_collection_id(self.repository_path)

        if self.qdrant_collection and self.qdrant_collection.strip():
            self.qdrant_collection = self.qdrant_collection.strip()
        else:
            self.qdrant_collection = build_qdrant_collection_name(self.collection_id)


class KnowledgeGraphBuilder:
    """Ingest parsed code entities into Neo4j and Qdrant with LM Studio embeddings."""

    def __init__(self, config: GraphStoreConfig | None = None) -> None:
        self.config = config or GraphStoreConfig()
        self._prepare_qdrant_path()
        self.qdrant = self._build_qdrant_client()
        self.neo4j = GraphDatabase.driver(
            self.config.neo4j_uri,
            auth=(self.config.neo4j_username, self.config.neo4j_password),
        )
        self.neo4j.verify_connectivity()
        self.embedding_client = OpenAI(
            base_url=self.config.embeddings_base_url,
            api_key=self.config.model_api_key,
            timeout=self.config.model_timeout_seconds,
        )
        self._ensure_graph_schema()

    def index_repository(self, repository_path: str | Path, *, rebuild: bool = False) -> dict[str, Any]:
        parsed_output = CodeParser(repository_path).parse()
        summary = self.ingest(parsed_output, rebuild=rebuild)
        summary["repository_path"] = str(Path(repository_path).resolve())
        return summary

    def ingest(self, parsed_output: dict[str, list[dict[str, Any]]], *, rebuild: bool = False) -> dict[str, Any]:
        entities = parsed_output.get("entities", [])
        relationships = parsed_output.get("relationships", [])
        if not entities:
            raise ValueError("Parsed output did not contain any entities to ingest.")

        if rebuild:
            self._clear_graph()
            self._recreate_collection()

        first_embedding = self.embed_text(self._embedding_text_for_entity(entities[0]))
        self._ensure_collection(vector_size=len(first_embedding))

        points: list[PointStruct] = []
        with self.neo4j.session() as session:
            for index, entity in enumerate(entities):
                embedding = first_embedding if index == 0 else self.embed_text(self._embedding_text_for_entity(entity))
                self._upsert_entity_node(session, entity)
                points.append(
                    PointStruct(
                        id=entity["id"],
                        vector=embedding,
                        payload={
                            "node_id": entity["id"],
                            "name": entity["name"],
                            "type": entity["type"],
                            "file_path": entity["file_path"],
                            "qualified_name": entity["qualified_name"],
                        },
                    )
                )

            for relationship in relationships:
                self._upsert_relationship(session, relationship)

        self.qdrant.upsert(
            collection_name=self.config.qdrant_collection,
            wait=True,
            points=points,
        )

        return {
            "entities_indexed": len(entities),
            "relationships_indexed": len(relationships),
            "collection_id": self.config.collection_id,
            "qdrant_collection": self.config.qdrant_collection,
        }

    def embed_text(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Embedding input must not be empty.")
        try:
            response = self.embedding_client.embeddings.create(
                model=self.config.embedding_model,
                input=[text],
            )
        except (APITimeoutError, APIConnectionError) as exc:
            raise EmbeddingsServiceError(
                "Unable to reach the embeddings service at "
                f"'{self.config.embeddings_base_url}'. Neo4j connectivity was verified successfully, "
                "so this is not a Neo4j failure. Check that your OpenAI-compatible server is running, "
                f"that NODIFYCTX_EMBEDDINGS_BASE_URL points to the right host, and that the embedding model "
                f"'{self.config.embedding_model}' is available. If this repository is already indexed, rerun "
                "with --skip-index."
            ) from exc
        return list(response.data[0].embedding)

    @staticmethod
    def _embedding_text_for_entity(entity: dict[str, Any]) -> str:
        raw_code = str(entity.get("raw_code", ""))
        if raw_code.strip():
            return raw_code

        entity_type = str(entity.get("type", "entity"))
        qualified_name = str(entity.get("qualified_name") or entity.get("name") or "unknown")
        file_path = str(entity.get("file_path") or ".")
        return f"{entity_type} {qualified_name} in {file_path}"

    def close(self) -> None:
        self.qdrant.close()
        self.neo4j.close()

    def __enter__(self) -> "KnowledgeGraphBuilder":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _prepare_qdrant_path(self) -> None:
        if self.config.qdrant_url:
            return
        if self.config.qdrant_path == ":memory:":
            return
        Path(self.config.qdrant_path).mkdir(parents=True, exist_ok=True)

    def _build_qdrant_client(self) -> QdrantClient:
        if self.config.qdrant_url:
            return QdrantClient(url=self.config.qdrant_url)
        return QdrantClient(path=self.config.qdrant_path)

    def _ensure_graph_schema(self) -> None:
        with self.neo4j.session() as session:
            session.run(
                """
                DROP CONSTRAINT nodify_ctx_node_id IF EXISTS
                """
            )
            session.run(
                """
                DROP CONSTRAINT repo_context_node_id IF EXISTS
                """
            )
            session.run(
                """
                CREATE CONSTRAINT nodify_ctx_node_repo_id IF NOT EXISTS
                FOR (node:CodeEntity)
                REQUIRE (node.repo_id, node.node_id) IS UNIQUE
                """
            )
            session.run(
                """
                DROP INDEX repo_context_node_name IF EXISTS
                """
            )
            session.run(
                """
                CREATE INDEX nodify_ctx_node_repo_name IF NOT EXISTS
                FOR (node:CodeEntity)
                ON (node.repo_id, node.name)
                """
            )

    def _clear_graph(self) -> None:
        with self.neo4j.session() as session:
            session.run(
                """
                MATCH (node:CodeEntity {repo_id: $repo_id})
                DETACH DELETE node
                """,
                repo_id=self.config.collection_id,
            )

    def _recreate_collection(self) -> None:
        if self.qdrant.collection_exists(self.config.qdrant_collection):
            self.qdrant.delete_collection(self.config.qdrant_collection)

    def _ensure_collection(self, *, vector_size: int) -> None:
        if self.qdrant.collection_exists(self.config.qdrant_collection):
            return
        self.qdrant.create_collection(
            collection_name=self.config.qdrant_collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    def _upsert_entity_node(self, session: Any, entity: dict[str, Any]) -> None:
        label = self._neo4j_label(entity["type"])
        session.run(
            f"""
            MERGE (node:CodeEntity {{repo_id: $repo_id, node_id: $node_id}})
            SET node:{label}
            SET node.repo_id = $repo_id,
                node.name = $name,
                node.type = $type,
                node.file = $file_path,
                node.file_path = $file_path,
                node.qualified_name = $qualified_name,
                node.start_line = $start_line,
                node.end_line = $end_line,
                node.raw_code = $raw_code
            """,
            repo_id=self.config.collection_id,
            node_id=entity["id"],
            name=entity["name"],
            type=entity["type"],
            file_path=entity["file_path"],
            qualified_name=entity["qualified_name"],
            start_line=entity["start_line"],
            end_line=entity["end_line"],
            raw_code=entity["raw_code"],
        )

    def _upsert_relationship(self, session: Any, relationship: dict[str, Any]) -> None:
        session.run(
            f"""
            MATCH (source:CodeEntity {{repo_id: $repo_id, node_id: $source_id}})
            MATCH (target:CodeEntity {{repo_id: $repo_id, node_id: $target_id}})
            MERGE (source)-[edge:{relationship['type']}]->(target)
            """,
            repo_id=self.config.collection_id,
            source_id=relationship["source_id"],
            target_id=relationship["target_id"],
        )

    @staticmethod
    def _neo4j_label(entity_type: str) -> str:
        labels = {
            "folder": "Folder",
            "file": "File",
            "function": "Function",
            "class": "Class",
        }
        try:
            return labels[entity_type]
        except KeyError as exc:
            raise ValueError(f"Unsupported entity type for Neo4j label: {entity_type}") from exc