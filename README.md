# RepoContext

Local code-graph indexing and exploration for Python, JavaScript, TypeScript, and React repositories.

RepoContext parses a local codebase into semantic entities, stores structure in Neo4j, stores embeddings in Qdrant, and exposes a small tool surface that an LLM can use for graph-guided code exploration.

## Supported source files

- `.py`
- `.js`
- `.jsx`
- `.ts`
- `.tsx`

The parser extracts classes, functions, methods, arrow-function assignments, and class field handlers, then resolves `CONTAINS` and `CALLS` relationships across the indexed repository.

## Setup

Install Python dependencies:

```bash
pip install -r requirements.txt
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

- `REPOCONTEXT_CHAT_MODEL`
- `REPOCONTEXT_EMBEDDING_MODEL`
- `REPOCONTEXT_CHAT_BASE_URL` and/or `REPOCONTEXT_EMBEDDINGS_BASE_URL` (or use the single `REPOCONTEXT_MODEL_BASE_URL` convenience variable)
- `REPOCONTEXT_MODEL_API_KEY` (or `REPOCONTEXT_LMSTUDIO_API_KEY` for legacy setups)
- `REPOCONTEXT_QDRANT_URL` when using a running Qdrant container
- `REPOCONTEXT_MAX_TOOL_ITERATIONS` to control how many tool rounds the agent may use
- `REPOCONTEXT_GRAPH_RECURSION_LIMIT` to control the LangGraph recursion cap

For public repos, do not commit `.env`. This project now includes `.gitignore` and `.env.example` for that workflow.

## Usage

Index a repository and start the interactive agent:

```bash
python main_agent.py "/path/to/repository"
# or on macOS/Linux
python main_agent.py "$HOME/path/to/repository"
```

Skip re-indexing when you want to reuse an existing graph/vector store:

```bash
python main_agent.py "/path/to/repository" --skip-index
```

Ask one question on the command line and exit:

```bash
python main_agent.py "/path/to/repository" --skip-index --question "Describe this project."
```

Use a running Qdrant server explicitly:

```bash
python main_agent.py "/path/to/repository" --qdrant-url http://127.0.0.1:6333
```

By default, RepoContext keeps each local repository in its own scope. The scope ID is derived from the repository path, so indexing repo A won't overwrite or mix with repo B in Neo4j or Qdrant.

If you want to pin a custom scope name, pass an optional collection ID:

```bash
python main_agent.py "/path/to/repository" --collection-id mobile_app
```

Combine `--skip-index`, a running Qdrant service, and an existing index for a fast test:

```bash
python main_agent.py "/path/to/repository" --skip-index --qdrant-url http://127.0.0.1:6333 --question "Which component initializes notifications?"
```

If your chat and embeddings APIs are behind the same endpoint (e.g., a single LM Studio instance), you can pass the same URL for both with the convenience flag:

```bash
python main_agent.py "/path/to/repository" --skip-index --model-base-url http://127.0.0.1:1234/v1 --question "Where is axios setup?"
```

## Testing A Specific Project

Recommended workflow for a target repo such as `~/example-repo` (macOS/Linux) or `C:\path\to\example-repo` (Windows):

1. Index it once.

```bash
python main_agent.py "~/example-repo" --qdrant-url http://127.0.0.1:6333
# or on macOS/Linux with expanded home
python main_agent.py "$HOME/example-repo" --qdrant-url http://127.0.0.1:6333
```

2. Reuse the existing graph/vector index for fast follow-up questions.

```bash
python main_agent.py "~/example-repo" --skip-index --qdrant-url http://127.0.0.1:6333 --question "Where is axios setup?"
```

3. Switch to interactive mode when you want a conversation instead of a one-shot answer.

```bash
python main_agent.py "~/example-repo" --skip-index --qdrant-url http://127.0.0.1:6333
```

`--skip-index` only works if the project has already been indexed into Neo4j and Qdrant with the same repository contents you want to search.

`--collection-id` is optional, but useful when you want a stable human-chosen scope name instead of the default path-derived one. `--qdrant-collection` remains available as a lower-level escape hatch if you need to force the exact Qdrant collection name.

## Local Tool Integration

The agent-facing tool layer lives in `agent_tools.py` and exposes four functions:

- `semantic_search(query)` for vector retrieval over indexed entities
- `get_callers(node_name)` for inbound graph traversal
- `get_dependencies(node_name)` for outbound graph traversal
- `inspect_code(node_name)` for raw code lookup after disambiguation

These are already wrapped into LangChain tools by `build_langchain_tools()` and bound in `main_agent.py`.

If you want to integrate RepoContext into another local agent runtime, the practical pattern is:

1. Index the repository with `KnowledgeGraphBuilder`.
2. Import the functions from `agent_tools.py` into your agent runtime.
3. Force your agent to start with `semantic_search` before calling `inspect_code`.
4. Use `REPOCONTEXT_QDRANT_URL` when your agent may create multiple concurrent Qdrant clients.

Minimal Python example:

```python
from dotenv import load_dotenv

from agent_tools import semantic_search, get_dependencies, inspect_code

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
- If you hit `GraphRecursionError`, raise `REPOCONTEXT_GRAPH_RECURSION_LIMIT` in `.env`. Example: `REPOCONTEXT_GRAPH_RECURSION_LIMIT=200`.
- If you want to remove the practical recursion cap, set `REPOCONTEXT_GRAPH_RECURSION_LIMIT=0`, `none`, or `unlimited`. The runtime maps that to a very large limit because LangGraph still expects an integer.
- If the agent exhausts its tool budget on broad review prompts, raise `REPOCONTEXT_MAX_TOOL_ITERATIONS` or ask a narrower question.