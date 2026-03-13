from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import os

from neo4j import GraphDatabase
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from parser import CodeParser


@dataclass(slots=True)
class GraphStoreConfig:
    neo4j_uri: str = field(default_factory=lambda: os.getenv("REPOCONTEXT_NEO4J_URI", "bolt://localhost:7687"))
    neo4j_username: str = field(default_factory=lambda: os.getenv("REPOCONTEXT_NEO4J_USERNAME", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: os.getenv("REPOCONTEXT_NEO4J_PASSWORD", "neo4j"))
    qdrant_url: str | None = field(default_factory=lambda: os.getenv("REPOCONTEXT_QDRANT_URL") or None)
    qdrant_path: str = field(default_factory=lambda: os.getenv("REPOCONTEXT_QDRANT_PATH", ".repocontext\\qdrant"))
    qdrant_collection: str = field(default_factory=lambda: os.getenv("REPOCONTEXT_QDRANT_COLLECTION", "repo_context_nodes"))
    embeddings_base_url: str = field(
        default_factory=lambda: os.getenv(
            "REPOCONTEXT_EMBEDDINGS_BASE_URL",
            os.getenv("REPOCONTEXT_LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1"),
        )
    )
    model_api_key: str = field(
        default_factory=lambda: os.getenv(
            "REPOCONTEXT_MODEL_API_KEY",
            os.getenv("REPOCONTEXT_LMSTUDIO_API_KEY", "lm-studio"),
        )
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "REPOCONTEXT_EMBEDDING_MODEL",
            "text-embedding-nomic-embed-text-v1.5",
        )
    )


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

        first_embedding = self.embed_text(entities[0]["raw_code"])
        self._ensure_collection(vector_size=len(first_embedding))

        points: list[PointStruct] = []
        with self.neo4j.session() as session:
            for index, entity in enumerate(entities):
                embedding = first_embedding if index == 0 else self.embed_text(entity["raw_code"])
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
            "qdrant_collection": self.config.qdrant_collection,
        }

    def embed_text(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Embedding input must not be empty.")
        response = self.embedding_client.embeddings.create(
            model=self.config.embedding_model,
            input=[text],
        )
        return list(response.data[0].embedding)

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
                CREATE CONSTRAINT repo_context_node_id IF NOT EXISTS
                FOR (node:CodeEntity)
                REQUIRE node.node_id IS UNIQUE
                """
            )
            session.run(
                """
                CREATE INDEX repo_context_node_name IF NOT EXISTS
                FOR (node:CodeEntity)
                ON (node.name)
                """
            )

    def _clear_graph(self) -> None:
        with self.neo4j.session() as session:
            session.run("MATCH (node:CodeEntity) DETACH DELETE node")

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
            MERGE (node:CodeEntity:{label} {{node_id: $node_id}})
            SET node.name = $name,
                node.type = $type,
                node.file = $file_path,
                node.file_path = $file_path,
                node.qualified_name = $qualified_name,
                node.start_line = $start_line,
                node.end_line = $end_line,
                node.raw_code = $raw_code
            """,
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
            MATCH (source:CodeEntity {{node_id: $source_id}})
            MATCH (target:CodeEntity {{node_id: $target_id}})
            MERGE (source)-[edge:{relationship["type"]}]->(target)
            """,
            source_id=relationship["source_id"],
            target_id=relationship["target_id"],
        )

    @staticmethod
    def _neo4j_label(entity_type: str) -> str:
        labels = {
            "function": "Function",
            "class": "Class",
        }
        try:
            return labels[entity_type]
        except KeyError as exc:
            raise ValueError(f"Unsupported entity type for Neo4j label: {entity_type}") from exc


if __name__ == "__main__":
    import argparse
    import json

    cli = argparse.ArgumentParser(description="Index a repository into RepoContext stores.")
    cli.add_argument("repository", nargs="?", default=".")
    cli.add_argument("--rebuild", action="store_true")
    args = cli.parse_args()

    with KnowledgeGraphBuilder() as builder:
        summary = builder.index_repository(args.repository, rebuild=args.rebuild)
    print(json.dumps(summary, indent=2))
