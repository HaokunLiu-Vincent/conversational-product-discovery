# Conversational Product Discovery

A lightweight, model-agnostic service for conversational laptop discovery. The LLM
extracts structured preference updates and writes grounded prose; deterministic Python
code owns session state, constraint validation, catalogue filtering, and candidate ranking.

## Setup

```bash
conda env create -f environment.yml
conda activate product-discovery
cp .env.example .env
```

Set `PRODUCT_DISCOVERY_OPENAI_API_KEY` in `.env`, then start the API:

```bash
uvicorn product_discovery.api:app --reload
```

The default catalogue is `data/processed/laptop_product_database_v3.parquet`. Its exclusion
audit is recorded in `data/processed/laptop_product_database_v3_exclusions.csv`; v2 remains
the unchanged source catalogue.

## API

`GET /health` reports process health. `POST /v1/chat` accepts a session ID and the latest
user message:

```bash
curl -X POST http://127.0.0.1:8000/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"demo","message":"I need 16GB RAM under $1500"}'
```

Session preference state is process-local and intentionally ephemeral. The configured
candidate maximum limits only representative products in the response; the qualified count
always reflects the complete deterministically filtered catalogue.

## Docker

Build the production image from the repository root:

```bash
docker build -t product-discovery:local .
```

Run it with individual environment variables:

```bash
docker run --rm -p 8000:8000 \
  -e PRODUCT_DISCOVERY_OPENAI_API_KEY="$PRODUCT_DISCOVERY_OPENAI_API_KEY" \
  -e PRODUCT_DISCOVERY_OPENAI_MODEL="${PRODUCT_DISCOVERY_OPENAI_MODEL:-gpt-6-astra}" \
  product-discovery:local
```

Alternatively, load all `PRODUCT_DISCOVERY_*` settings from the local `.env` file:

```bash
docker run --rm -p 8000:8000 --env-file .env product-discovery:local
```

The `.env` file is excluded from the build context and is never stored in the image. The
container runs one Uvicorn worker so session state remains process-local for this version.
Once it starts, verify health and send a chat request:

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"docker-demo","message":"I need 16GB RAM under $1500"}'
```

## Development

```bash
pytest
ruff check src tests
ruff format --check src tests
```

Unit tests cover deterministic components. Integration tests exercise API wiring without
calling external services. Development evaluation data may be used for debugging, but do
not inspect or tune against held-out dialogues outside an explicitly requested formal run.

## Package layout

- `llm/`: provider interface and OpenAI Responses API adapter.
- `preferences/`: deterministic validation and in-memory state ownership.
- `products/`: parquet access, full constraint retrieval, and representative selection.
- `responses/`: grounded response boundary.
- `services/`: chat workflow orchestration.
- `api.py`: FastAPI application and dependency assembly.
