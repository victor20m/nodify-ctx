from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_core.tools import StructuredTool

from graph_builder import KnowledgeGraphBuilder


@lru_cache(maxsize=1)
def get_builder() -> KnowledgeGraphBuilder:
    return KnowledgeGraphBuilder()


def semantic_search(query: str) -> list[dict[str, str]]:
    """Run semantic retrieval over indexed code entities.

    Use this first for every new user question. Pass a natural-language query that
    describes the behavior, symbol, or responsibility you want to locate.

    Args:
        query: Natural-language search text for the indexed repository.

    Returns:
        Up to three dictionaries with keys `name`, `type`, and `file`.
    """
    if not query.strip():
        raise ValueError("query must not be empty.")

    builder = get_builder()
    if not builder.qdrant.collection_exists(builder.config.qdrant_collection):
        raise RuntimeError(
            f"Qdrant collection '{builder.config.qdrant_collection}' does not exist. "
            "Index a repository before using semantic_search."
        )

    embedding = builder.embed_text(query.strip())
    response = builder.qdrant.query_points(
        collection_name=builder.config.qdrant_collection,
        query=embedding,
        limit=3,
        with_payload=True,
    )
    hits = response.points
    return [
        {
            "name": hit.payload["name"],
            "type": hit.payload["type"],
            "file": hit.payload["file_path"],
        }
        for hit in hits
    ]


def get_callers(node_name: str) -> list[str]:
    """List code nodes that call the named target.

    Use this after semantic search to understand inbound dependencies before
    deciding whether the node is the right inspection target.

    Args:
        node_name: Exact indexed node name to look up in Neo4j.

    Returns:
        A sorted list of caller names. The list may be empty when no callers exist.
    """
    if not node_name.strip():
        raise ValueError("node_name must not be empty.")

    builder = get_builder()
    with builder.neo4j.session() as session:
        records = session.run(
            """
            MATCH (caller:CodeEntity)-[:CALLS]->(target:CodeEntity)
            WHERE target.repo_id = $repo_id AND caller.repo_id = $repo_id AND target.name = $node_name
            RETURN DISTINCT caller.name AS name
            ORDER BY name
            """,
            repo_id=builder.config.collection_id,
            node_name=node_name.strip(),
        )
        return [record["name"] for record in records]


def get_dependencies(node_name: str) -> list[str]:
    """List the named node's outbound code dependencies.

    Use this after semantic search to map what a candidate node calls before
    inspecting its source code directly.

    Args:
        node_name: Exact indexed node name to look up in Neo4j.

    Returns:
        A sorted list of dependency names. The list may be empty when none exist.
    """
    if not node_name.strip():
        raise ValueError("node_name must not be empty.")

    builder = get_builder()
    with builder.neo4j.session() as session:
        records = session.run(
            """
            MATCH (source:CodeEntity)-[:CALLS]->(dependency:CodeEntity)
            WHERE source.repo_id = $repo_id AND dependency.repo_id = $repo_id AND source.name = $node_name
            RETURN DISTINCT dependency.name AS name
            ORDER BY name
            """,
            repo_id=builder.config.collection_id,
            node_name=node_name.strip(),
        )
        return [record["name"] for record in records]


def inspect_code(node_name: str) -> dict[str, Any]:
    """Fetch source code for one clearly identified node.

    Use this only after semantic search and graph traversal have isolated the
    correct node. If multiple indexed nodes share the same name, the tool returns
    an explicit ambiguity payload instead of guessing.

    Args:
        node_name: Exact indexed node name to inspect.

    Returns:
        A dictionary containing `name`, `file`, and `code`, or an error payload.
    """
    if not node_name.strip():
        raise ValueError("node_name must not be empty.")

    builder = get_builder()
    with builder.neo4j.session() as session:
        records = list(
            session.run(
                """
                MATCH (node:CodeEntity)
                WHERE node.repo_id = $repo_id AND node.name = $node_name
                RETURN node.name AS name,
                       node.file AS file,
                       node.raw_code AS code
                ORDER BY node.file, node.start_line
                LIMIT 5
                """,
                repo_id=builder.config.collection_id,
                node_name=node_name.strip(),
            )
        )

    if not records:
        return {
            "name": node_name,
            "error": f"No indexed node named '{node_name}' was found.",
        }

    if len(records) > 1:
        return {
            "name": node_name,
            "error": "Multiple nodes share this name. Narrow the target with graph traversal first.",
            "matches": [record["file"] for record in records],
        }

    record = records[0]
    return {
        "name": record["name"],
        "file": record["file"],
        "code": record["code"],
    }


TOOL_FUNCTIONS = (
    semantic_search,
    get_callers,
    get_dependencies,
    inspect_code,
)


def build_langchain_tools() -> list[StructuredTool]:
    return [StructuredTool.from_function(func=tool_function) for tool_function in TOOL_FUNCTIONS]


def close_runtime() -> None:
    if get_builder.cache_info().currsize == 0:
        return
    get_builder().close()
    get_builder.cache_clear()
