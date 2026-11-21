# Advanced RAG Agent with Context Compression & Semantic Memory

A FastAPI project scaffold implementing parent/child retrieval, dense+sparse search with Reciprocal Rank Fusion (RRF), Cohere reranking, persistent user memory, and latency telemetry.

## Requirements

- Python 3.10+
- OpenAI API key for embeddings and generated answers
- Cohere API key for hosted reranking
- Optional Qdrant server URL/API key (or local on-disk Qdrant)
- Optional Mem0 API key for hosted semantic memory

## Setup (Windows PowerShell)

```powershell
cd advanced-rag-memory-agent
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and add keys. Never commit `.env`.

```dotenv
OPENAI_API_KEY=your_openai_key
COHERE_API_KEY=your_cohere_key
# Optional hosted Qdrant; leave blank to use local disk at QDRANT_PATH
QDRANT_URL=
QDRANT_API_KEY=
# Optional: set MEMORY_BACKEND=mem0 and add MEM0_API_KEY to use Mem0
MEMORY_BACKEND=local
MEM0_API_KEY=
```

## Run the API

```powershell
uvicorn main:app --reload
```

Open API docs at http://127.0.0.1:8000/docs. Health check: http://127.0.0.1:8000/health.

## API endpoints

### `POST /api/ingest`

```json
{"text":"Your knowledge base text goes here.","source":"handbook"}
```

Splits text into parent segments (about 1000 characters) and child segments (about 200 characters), embeds children, and stores vectors in Qdrant when OpenAI credentials are configured.

### `POST /api/chat`

```json
{"query":"What is the policy?","user_id":"usr_99","session_id":"session_001"}
```

Returns `answer`, `sources`, and `telemetry` containing `retrieval_latency_ms`, `rerank_latency_ms`, `generation_latency_ms`, and `total_latency_ms`.

### `GET /api/users/{user_id}/memories`

Returns the stored facts/preferences for the requested user.

## Retrieval and compression flow

1. Parent documents are divided into smaller child chunks and mapped by parent ID.
2. Dense vector search uses OpenAI embeddings and Qdrant.
3. Sparse keyword search uses BM25.
4. Reciprocal Rank Fusion merges the two ranked lists.
5. Up to 20 candidates are passed to Cohere ReRank; the top 5 are sent to the generation prompt, subject to the configured minimum relevance score.

The Qdrant collection is configured for the 1536-dimensional `text-embedding-3-small` model. If you change embedding models, update the collection dimension accordingly.

## Semantic memory

The default `MEMORY_BACKEND=local` stores lightweight extracted preference facts in `data/memories.json` so the project can be run without a Mem0 account. For the assessment's hosted semantic-memory integration, set `MEMORY_BACKEND=mem0` and provide `MEM0_API_KEY`. The adapter uses Mem0's user-scoped add/search/get_all operations. Verify the installed Mem0 SDK version against its current documentation if the API changes.

## Tests

```powershell
pytest -q
```

Tests cover RRF ordering/merging, child splitting, rerank score filtering, memory extraction/persistence, and API smoke tests. Tests use local fakes and do not require API keys.

## Postman

Import `postman/Advanced-RAG-Memory-Agent.postman_collection.json` into Postman. Set `baseUrl` to `http://127.0.0.1:8000`.

## Important implementation notes

- Without `OPENAI_API_KEY`, the app uses a lightweight in-memory retrieval/demo mode; it does not provide hosted LLM generation or persistent Qdrant indexing. Add the key to activate embeddings, Qdrant and OpenAI generation.
- Without `COHERE_API_KEY`, reranking uses a deterministic local lexical fallback for development. Add the key to activate Cohere's hosted ReRank API.
- For real multi-user production deployments, use a remote Qdrant service and Mem0/Zep, configure authentication, add upload/file ingestion and enforce per-user access controls.
- API keys are read from environment variables and must not be committed.

## Suggested Git commit milestones

Create separate commits as you implement/review each milestone; do not fabricate historical commits:

1. `chore: scaffold FastAPI project and settings`
2. `feat: add parent child document splitting`
3. `feat: add dense Qdrant indexing`
4. `feat: add BM25 sparse retrieval and RRF`
5. `feat: integrate Cohere reranking`
6. `feat: add persistent user semantic memory`
7. `feat: add chat and memory API telemetry`
8. `test: add retrieval memory and API test suite`
9. `docs: add setup guide and Postman collection`

## GitHub submission

Create a **public** GitHub repository named `advanced-rag-memory-agent`, then push this folder:

```powershell
git init
git add .
git commit -m "chore: initial advanced RAG memory agent project"
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/advanced-rag-memory-agent.git
git push -u origin main
```

Replace `YOUR_GITHUB_USERNAME` with your GitHub login. The URL above is a template, not an already-created repository.
