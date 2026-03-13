# NodifyCtx

Local code-graph indexing and exploration for Python, JavaScript, TypeScript, and React repositories.

NodifyCtx parses a local codebase into semantic entities, stores structure in Neo4j, stores embeddings in Qdrant, and exposes a small tool surface that an LLM can use for graph-guided code exploration.

## Supported source files

- `.py`
- `.js`
- `.jsx`
- `.ts`
- `.tsx`

The parser extracts folders, files, classes, functions, methods, arrow-function assignments, and class field handlers, then resolves `CONTAINS` and `CALLS` relationships across the indexed repository.

## Project Layout

Core implementation lives under `src/nodifyctx/`:

- `src/nodifyctx/agent.py` for the CLI agent runtime
- `src/nodifyctx/graph_builder.py` for Neo4j and Qdrant ingestion
- `src/nodifyctx/parser.py` for repository parsing and IR generation
- `src/nodifyctx/tools.py` for agent-facing retrieval and graph tools

The project also includes `pyproject.toml` for packaging and test configuration. Install the project in editable mode during local development so the `src` package is importable.

## Setup

Install Python dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

Start Neo4j:

```bash
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/change-me -e NEO4J_dbms_security_auth__minimum__password__length=1 neo4j:5
```

Start Qdrant:

```bash
docker run -d --name qdrant -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant:latest
```

Create `.env` from `.env.example`, then set the required values:

- `NODIFYCTX_CHAT_MODEL`
- `NODIFYCTX_EMBEDDING_MODEL`
- `NODIFYCTX_CHAT_BASE_URL` and/or `NODIFYCTX_EMBEDDINGS_BASE_URL` (or use the single `NODIFYCTX_MODEL_BASE_URL` convenience variable)
- `NODIFYCTX_MODEL_API_KEY` (or legacy `REPOCONTEXT_*` keys during migration)
- `NODIFYCTX_MODEL_TIMEOUT_SECONDS` to fail fast when the local model endpoint is down or unreachable
- `NODIFYCTX_QDRANT_URL` when using a running Qdrant container
- `NODIFYCTX_MAX_TOOL_ITERATIONS` to control how many tool rounds the agent may use
- `NODIFYCTX_GRAPH_RECURSION_LIMIT` to control the LangGraph recursion cap

For public repos, do not commit `.env`. This project now includes `.gitignore` and `.env.example` for that workflow.

## Usage

Index a repository and start the interactive agent:

```bash
python -m nodifyctx "/path/to/repository"
# or use the console script after installation
nodifyctx "/path/to/repository"
```

Skip re-indexing when you want to reuse an existing graph/vector store:

```bash
python -m nodifyctx "/path/to/repository" --skip-index
```

Ask one question on the command line and exit:

```bash
python -m nodifyctx "/path/to/repository" --skip-index --question "Describe this project."
```

Use a running Qdrant server explicitly:

```bash
python -m nodifyctx "/path/to/repository" --qdrant-url http://127.0.0.1:6333
```

By default, NodifyCtx keeps each local repository in its own scope. The scope ID is derived from the repository path, so indexing repo A won't overwrite or mix with repo B in Neo4j or Qdrant.

If you want to pin a custom scope name, pass an optional collection ID:

```bash
python -m nodifyctx "/path/to/repository" --collection-id mobile_app
```

Combine `--skip-index`, a running Qdrant service, and an existing index for a fast test:

```bash
python -m nodifyctx "/path/to/repository" --skip-index --qdrant-url http://127.0.0.1:6333 --question "Which component initializes notifications?"
```

If your chat and embeddings APIs are behind the same endpoint (e.g., a single LM Studio instance), you can pass the same URL for both with the convenience flag:

```bash
python -m nodifyctx "/path/to/repository" --skip-index --model-base-url http://127.0.0.1:1234/v1 --question "Where is axios setup?"
```

If your model endpoint is remote or occasionally unavailable, reduce the wait with an explicit timeout:

```bash
python -m nodifyctx "/path/to/repository" --model-timeout-seconds 3
```

## Testing A Specific Project

Recommended workflow for a target repo such as `~/example-repo` (macOS/Linux) or `C:\path\to\example-repo` (Windows):

1. Index it once.

```bash
python -m nodifyctx "~/example-repo" --qdrant-url http://127.0.0.1:6333
# or on macOS/Linux with expanded home
python -m nodifyctx "$HOME/example-repo" --qdrant-url http://127.0.0.1:6333
```

2. Reuse the existing graph/vector index for fast follow-up questions.

```bash
python -m nodifyctx "~/example-repo" --skip-index --qdrant-url http://127.0.0.1:6333 --question "Where is axios setup?"
```

3. Switch to interactive mode when you want a conversation instead of a one-shot answer.

```bash
python -m nodifyctx "~/example-repo" --skip-index --qdrant-url http://127.0.0.1:6333
```

`--skip-index` only works if the project has already been indexed into Neo4j and Qdrant with the same repository contents you want to search.

`--collection-id` is optional, but useful when you want a stable human-chosen scope name instead of the default path-derived one. `--qdrant-collection` remains available as a lower-level escape hatch if you need to force the exact Qdrant collection name.

## Local Tool Integration

The agent-facing tool layer lives in `src/nodifyctx/tools.py`. It exposes four functions:

- `semantic_search(query)` for vector retrieval over indexed entities
- `get_callers(node_name)` for inbound graph traversal
- `get_dependencies(node_name)` for outbound graph traversal
- `inspect_code(node_name)` for raw code lookup after disambiguation

These are already wrapped into LangChain tools by `build_langchain_tools()` and bound in `src/nodifyctx/agent.py`.

If you want to integrate NodifyCtx into another local agent runtime, the practical pattern is:

1. Index the repository with `KnowledgeGraphBuilder`.
2. Import the functions from `nodifyctx.tools` into your agent runtime.
3. Force your agent to start with `semantic_search` before calling `inspect_code`.
4. Use `NODIFYCTX_QDRANT_URL` when your agent may create multiple concurrent Qdrant clients.

Minimal Python example:

```python
from dotenv import load_dotenv

from nodifyctx.tools import semantic_search, get_dependencies, inspect_code

load_dotenv()

hits = semantic_search("axios setup in app startup")
deps = get_dependencies(hits[0]["name"])
code = inspect_code(hits[0]["name"])
```

## Current Validation

This project was validated against a local test repository such as `~/example-repo` (macOS/Linux) or `C:\path\to\example-repo` (Windows) with the current parser changes.

- Parsed 453 entities and 337 relationships from `.ts` and `.tsx` files.
- Indexed successfully into Neo4j and Qdrant.
- Returned relevant semantic hits for React and TypeScript questions.

## Troubleshooting

- If the agent stops after printing raw `<tool_call>` text, that is a tool-calling compatibility issue between the model output and the runtime. The current runtime now includes a fallback parser for that output pattern, but models with stronger native tool calling still behave better.
- If the agent hangs for a long time and eventually errors during an HTTP request to LM Studio, that is usually LLM service latency rather than graph traversal failure.
- For best results, prefer a model served through LM Studio that handles OpenAI-style tool calling reliably and keep `temperature=0`.
- If you hit `GraphRecursionError`, raise `NODIFYCTX_GRAPH_RECURSION_LIMIT` in `.env`. Example: `NODIFYCTX_GRAPH_RECURSION_LIMIT=200`.
- If you want to remove the practical recursion cap, set `NODIFYCTX_GRAPH_RECURSION_LIMIT=0`, `none`, or `unlimited`. The runtime maps that to a very large limit because LangGraph still expects an integer.
- If the agent exhausts its tool budget on broad review prompts, raise `NODIFYCTX_MAX_TOOL_ITERATIONS` or ask a narrower question.

During migration, the runtime still accepts legacy `REPOCONTEXT_*` environment variables, but new examples and defaults use `NODIFYCTX_*`.