# Conversational Product Discovery

A lightweight, model-agnostic service for conversational laptop discovery. The LLM
extracts structured preference updates and writes grounded prose; deterministic Python
code owns session state, constraint validation, catalogue filtering, and candidate ranking.

## Setup

Runtime catalogue Parquet files are intentionally not stored in Git. Provision a local copy,
then configure its path explicitly:

```bash
conda env create -f environment.yml
conda activate product-discovery
cp .env.example .env
export PRODUCT_DISCOVERY_CATALOGUE_PATH="$PWD/data/processed/laptop_product_database_v3.parquet"
unset PRODUCT_DISCOVERY_CATALOGUE_URI
uvicorn product_discovery.api:app --reload
```

Set `PRODUCT_DISCOVERY_OPENAI_API_KEY` in `.env` before starting the API. The conventional
local path is `data/processed/laptop_product_database_v3.parquet`, but any readable local
path can be supplied with `PRODUCT_DISCOVERY_CATALOGUE_PATH`.

For deployment, `PRODUCT_DISCOVERY_CATALOGUE_URI` may point to an S3 object such as
`s3://catalogue-bucket/catalogues/laptop_product_database_v3.parquet`. When set, it takes
precedence over `PRODUCT_DISCOVERY_CATALOGUE_PATH`: the object is downloaded once to a
temporary local file and then processed by the same Parquet repository. Authentication uses
the standard AWS SDK credential provider chain, including IAM roles; do not put AWS
credentials in `.env`. Leave the URI unset to retain the local-file workflow.

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

The image contains application code only. To use a local catalogue, bind-mount it read-only
and set the in-container path:

```bash
docker run --rm -p 8000:8000 \
  --env-file .env \
  --mount type=bind,source="$PWD/data/processed/laptop_product_database_v3.parquet",target=/catalogue/catalogue.parquet,readonly \
  -e PRODUCT_DISCOVERY_CATALOGUE_PATH=/catalogue/catalogue.parquet \
  -e PRODUCT_DISCOVERY_CATALOGUE_URI= \
  product-discovery:local
```

Alternatively, use S3. The following example mounts an existing AWS profile read-only; any
AWS SDK credential-provider-chain source may be used instead:

```bash
docker run --rm -p 8000:8000 \
  --env-file .env \
  --mount type=bind,source="$HOME/.aws",target=/aws,readonly \
  -e AWS_CONFIG_FILE=/aws/config \
  -e AWS_SHARED_CREDENTIALS_FILE=/aws/credentials \
  -e AWS_PROFILE="${AWS_PROFILE:-default}" \
  -e PRODUCT_DISCOVERY_CATALOGUE_URI=s3://catalogue-bucket/catalogues/laptop_product_database_v3.parquet \
  product-discovery:local
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

For future AWS execution, set `PRODUCT_DISCOVERY_CATALOGUE_URI` and grant the workload IAM
role `s3:GetObject` access to that object. Do not bake credentials into the image or task
definition. The service downloads the object once at repository initialization and logs its
source, validated row count, and SHA-256 checksum as deployment evidence.

## Development

```bash
pytest
ruff check src tests
ruff format --check src tests
```

Unit tests cover deterministic components. Integration tests exercise API wiring without
calling external services. Development evaluation data may be used for debugging, but do
not inspect or tune against held-out dialogues outside an explicitly requested formal run.

## Continuous integration

GitHub Actions runs on pushes to `main` and pull requests targeting `main`. The Python 3.12
job installs `.[dev]`, runs Ruff lint and formatting checks, and executes the complete
non-live test suite. It does not use OpenAI or AWS credentials, contact S3, or require a
runtime catalogue. The optional live OpenAI test is explicitly excluded.

After the Python job succeeds, a separate job verifies that no Parquet catalogue is present
and builds the production Docker image without pushing it. Pip and Docker layer caches are
enabled to reduce repeat build time.

On pushes to `main` only, a third job runs after the Docker build and publishes the image to
private Amazon ECR. Configure the repository variables `AWS_ROLE_ARN`, `AWS_REGION`, and
`ECR_REPOSITORY`. GitHub assumes the role through OIDC, so no long-lived AWS access keys are
stored in GitHub. The image is tagged `sha-<full-git-sha>-r<run-attempt>`; no `latest` tag is
written, which is compatible with immutable ECR repositories. The published URI and digest
are added to the workflow summary. This stage publishes only—it does not access S3 or deploy
to ECS.

## Package layout

- `llm/`: provider interface and OpenAI Responses API adapter.
- `preferences/`: deterministic validation and in-memory state ownership.
- `products/`: parquet access, full constraint retrieval, and representative selection.
- `responses/`: grounded response boundary.
- `services/`: chat workflow orchestration.
- `api.py`: FastAPI application and dependency assembly.
