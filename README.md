# UTS 32040 Industry Project Artifacts

## Project Overview

This repository contains a laptop-domain conversational product-discovery component intended to
sit upstream of an existing recommendation or ranking system. Its runtime flow is:

```text
Natural-language multi-turn conversation
→ LLM structured preference update
→ deterministic validation
→ application-owned preference state
→ deterministic hard-constraint filtering
→ qualified catalogue candidates
→ representative product selection
→ grounded response and structured API output
```

The central design principle is: **LLM interprets language; deterministic application logic owns
and executes state.** The LLM proposes per-turn updates and writes responses, but it neither
mutates persistent state directly nor decides whether products satisfy hard constraints.

The implemented MVP is deliberately restricted to laptop discovery. Its structure may be adapted
to other structured product categories, but no other category was implemented or evaluated. The
system does not implement production pre-ranking, fine-ranking, business re-ranking, or a new
recommendation algorithm. Its qualified candidates and structured state are intended as inputs to
such downstream systems.

## Project Scope

| In scope | Out of scope |
|---|---|
| Laptop-domain preference extraction | Model training or fine-tuning |
| Multi-turn state retention, correction, and removal | Vector databases or embedding retrieval |
| Pydantic validation and canonicalization | Learned or behavioural recommendation ranking |
| Deterministic hard-constraint retrieval | Payments, inventory, and checkout |
| Deterministic representative selection | Distributed production session storage |
| Grounded response generation | Replacement of an existing recommender |
| FastAPI, Docker, CI, ECR publication, S3 loading, and AWS deployment validation | Fully automated application deployment |
| Development, held-out, and manual-response evaluation | General e-commerce validation |

## Architecture

`ChatService` orchestrates the provider-neutral LLM adapter, deterministic state manager,
catalogue search, representative selector, and response generator. Provider-specific OpenAI code
is isolated under `llm/`. Session state is process-local and therefore intentionally ephemeral.

Hard filtering evaluates the complete catalogue before `PRODUCT_DISCOVERY_MAX_CANDIDATES` limits
the representative products returned to the response layer. Representative selection samples
deterministically across the price-ordered qualified set; it is not learned ranking.

No standalone architecture figure is currently committed. The module structure below is the
authoritative implementation map.

## Preference State

The application owns this validated Pydantic state:

| Field | Type | Role |
|---|---|---|
| `budget_max` | positive number or null | Executable maximum price |
| `brand` | canonical string or null | Executable brand equality constraint |
| `ram_min_gb` | integer or null | Executable minimum RAM |
| `storage_min_gb` | integer or null | Executable minimum storage |
| `screen_size_max_in` | number or null | Executable maximum screen size |
| `gpu_required` | canonical string or null | Executable GPU equality constraint |
| `weight_max_kg` | positive number or null | Executable maximum weight |
| `usage` | categorical string or null | Semantic context only |
| `preference_notes` | string or null | Free-text semantic context only |

Updates preserve explicit `set` and `remove` operations. `usage` and `preference_notes` are passed
to response generation but never filter the catalogue.

## Product Catalogue

The normalized `Product` contract contains:

```text
product_id, title, brand, price, ram_gb, storage_gb, screen_in, gpu,
weight_kg, cpu, os, description, average_rating, rating_number, quality_score
```

Most fields are optional because missing catalogue values are retained as missing rather than
fabricated. A product with a missing value cannot satisfy an active constraint on that field.

The frozen runtime catalogue is `laptop_product_database_v3.parquet` with 4,933 records. Runtime
Parquet files are intentionally excluded from Git and the Docker build context. Supply a local
file through `PRODUCT_DISCOVERY_CATALOGUE_PATH`, or set `PRODUCT_DISCOVERY_CATALOGUE_URI` to an S3
object. S3 objects are downloaded once to a temporary local file and then processed by the same
validated Parquet repository. The v3 exclusion evidence is retained in
`data/processed/laptop_product_database_v3_exclusions.csv`.

## Repository Structure

```text
.
├── .github/workflows/ci.yml        # Python checks, Docker build, immutable ECR publish
├── data/
│   ├── evaluation/                 # Tracked schema and dataset summary metadata
│   └── processed/                  # Tracked v3 exclusion audit; Parquet files ignored
├── evaluation/
│   └── README.md                   # Evaluation runner and artifact conventions
├── notebooks/                      # Prototype, EDA, preprocessing, and dataset-generation work
├── src/product_discovery/
│   ├── api.py                      # FastAPI routes and dependency assembly
│   ├── configuration.py            # PRODUCT_DISCOVERY_* settings
│   ├── schemas.py                  # Pydantic application contracts
│   ├── evaluation/                 # Exact deterministic evaluation implementation
│   ├── llm/                        # Provider interface and OpenAI adapter
│   ├── preferences/                # Validation and in-memory state ownership
│   ├── products/                   # Parquet/S3 loading, retrieval, and selection
│   ├── responses/                  # Grounded response boundary
│   └── services/                   # Chat orchestration
├── tests/                          # Unit and integration tests
├── Dockerfile
├── environment.yml
├── LICENSE                         # MIT License for original project source code
└── pyproject.toml
```

Generated evaluation results live under `evaluation/results/` locally and are ignored by Git.
The JSONL/CSV benchmark splits and runtime Parquet catalogues are also excluded from public Git.

## Local Setup

Python 3.12 is required. The repository provides a Conda environment definition:

```bash
conda env create -f environment.yml
conda activate product-discovery
python -m pip install -e ".[dev]"
cp .env.example .env
```

Place the runtime catalogue at a local path and configure `.env`:

```dotenv
PRODUCT_DISCOVERY_OPENAI_API_KEY=<your-openai-api-key>
PRODUCT_DISCOVERY_OPENAI_MODEL=gpt-6-astra
PRODUCT_DISCOVERY_CATALOGUE_PATH=data/processed/laptop_product_database_v3.parquet
PRODUCT_DISCOVERY_MAX_CANDIDATES=5
```

Start the service:

```bash
python -m uvicorn product_discovery.api:app --reload
```

Useful endpoints:

- Health: <http://127.0.0.1:8000/health>
- OpenAPI/Swagger: <http://127.0.0.1:8000/docs>
- Chat: `POST http://127.0.0.1:8000/v1/chat`

Example request:

```bash
curl --fail --silent --show-error \
  -X POST http://127.0.0.1:8000/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"demo","message":"I need 16 GB RAM under $1500"}'
```

## Environment Configuration

| Variable | Purpose | Default |
|---|---|---|
| `PRODUCT_DISCOVERY_OPENAI_API_KEY` | Runtime OpenAI credential | Required for chat |
| `PRODUCT_DISCOVERY_OPENAI_MODEL` | OpenAI model identifier | `gpt-6-astra` |
| `PRODUCT_DISCOVERY_CATALOGUE_PATH` | Local Parquet path | `data/processed/laptop_product_database_v3.parquet` |
| `PRODUCT_DISCOVERY_CATALOGUE_URI` | Optional `s3://` catalogue URI | Unset |
| `PRODUCT_DISCOVERY_MAX_CANDIDATES` | Representative response limit, 1–20 | `5` |

When `PRODUCT_DISCOVERY_CATALOGUE_URI` is non-empty, it takes precedence over the local path.
AWS authentication uses the standard SDK credential provider chain, including workload IAM roles.
Never store OpenAI keys, AWS access keys, session tokens, private resource identifiers, or other
secrets in Git. `.env` is ignored and excluded from Docker; `.env.example` contains placeholders
only.

## Running with Docker

Build the canonical local image:

```bash
docker build -t product-discovery:local .
```

The image contains application code but no runtime catalogue or `.env`. Run with a read-only local
catalogue mount:

```bash
docker run --rm -p 8000:8000 \
  --env-file .env \
  --mount type=bind,source="$PWD/data/processed/laptop_product_database_v3.parquet",target=/catalogue/catalogue.parquet,readonly \
  -e PRODUCT_DISCOVERY_CATALOGUE_PATH=/catalogue/catalogue.parquet \
  -e PRODUCT_DISCOVERY_CATALOGUE_URI= \
  product-discovery:local
```

For local S3 testing, mount an existing AWS profile read-only rather than copying credentials into
the image:

```bash
docker run --rm -p 8000:8000 \
  --env-file .env \
  --mount type=bind,source="$HOME/.aws",target=/aws,readonly \
  -e AWS_CONFIG_FILE=/aws/config \
  -e AWS_SHARED_CREDENTIALS_FILE=/aws/credentials \
  -e AWS_PROFILE=<profile-name> \
  -e PRODUCT_DISCOVERY_CATALOGUE_URI='s3://<bucket>/<key>.parquet' \
  product-discovery:local
```

The container exposes port 8000 and uses one Uvicorn worker to preserve process-local session
state for this project version.

## Testing and Quality Checks

```bash
python -m pytest -m "not live_openai"
python -m ruff check src tests
python -m ruff format --check src tests
```

Routine tests use mocks or synthetic Parquet fixtures and require neither OpenAI/AWS credentials
nor the runtime catalogue. The optional provider test runs only when an API key is configured:

```bash
python -m pytest -m live_openai tests/integration/test_openai_adapter_live.py
```

## Evaluation Protocol

Ground truth was authored before model inference. Development data supported debugging and prompt
refinement; the protected held-out split was evaluated once after freezing tag
`evaluation-freeze-v1`. No post-held-out tuning was performed.

The evaluator uses exact deterministic comparisons:

- **Strict:** all fields, including exact `preference_notes` and clarification text.
- **Core:** seven hard fields plus categorical `usage`, excluding free text.
- **Executable:** only the seven hard constraints used by retrieval.

The separate manual response-faithfulness packet replays the full pipeline and supplies selected
product records for human claim checking. It does not use an LLM judge, embeddings, fuzzy matching,
or automatic faithfulness labels.

Development evaluation can be reproduced when the externally supplied development JSONL is
available:

```bash
python -m product_discovery.evaluation.run_development \
  --output-dir evaluation/results/dev
```

## Headline Evaluation Results

These values were verified from the frozen local result artifacts. Generated result files are not
tracked in Git.

| Evaluation | Dialogues / turns | Executable update exact | Executable turn-state exact | Executable final-state exact | Executable field-value accuracy |
|---|---:|---:|---:|---:|---:|
| Development | 60 / 155 | 99.35% | 98.71% | 98.33% | 99.82% |
| Protected held-out | 130 / 327 | 98.47% | 96.94% | 96.15% | 99.56% |

Held-out inference completed 327/327 calls with zero provider, extraction, parsing, or validation
failures. Median extraction latency was 1.669 seconds and p95 was 3.037 seconds.

A separate manual response-faithfulness audit covered 35 target responses across all 23 held-out
scenario labels. All 35 reviewed responses passed the audit rubric, and all 1,270 identified
factual product claims were supported by the supplied candidate records. There were zero
unsupported claims and no provider or runtime failures. These results apply to the audited sample
and should not be interpreted as a universal factual-correctness guarantee.

## Known Limitations

- Only the laptop domain was implemented and formally evaluated.
- The category schema and canonical values are manually designed.
- Controlled synthetic dialogues dominate the benchmark; ten held-out dialogues are manually
  authored adversarial cases.
- Only the frozen `gpt-5.6-terra` evaluation configuration was formally evaluated. The runtime
  model remains configurable.
- Session state is process-local and is not suitable for horizontally scaled persistence.
- Representative selection is deterministic sampling, not learned personalization or ranking.
- Product-family-to-manufacturer normalization remains incomplete.
- Real catalogue metadata may be missing, noisy, inconsistent, or stale.
- CI publishes an image but does not continuously deploy the application to ECS.
- The response-faithfulness packet is limited to 35 manually reviewable cases.

## CI/CD and AWS Deployment

GitHub Actions runs Python 3.12 tests and Ruff checks, then builds a clean-checkout Docker image.
On pushes to `main`, a third job uses GitHub OIDC to assume the role named by repository variable
`AWS_ROLE_ARN`, logs in to private ECR, and publishes an immutable tag:

```text
sha-<full-git-sha>-r<run-attempt>
```

No long-lived AWS access keys are stored in the workflow. The application supports an S3 runtime
catalogue through the AWS SDK role chain. An AWS deployment should inject the OpenAI key from a
managed secret and use distinct least-privilege execution and application roles.

The repository does **not** contain ECS Express Mode/Fargate service definitions, Secrets Manager
resources, CloudWatch configuration, or an ECS deployment job. Those resources and any deployment
evidence are managed outside this repository; CI currently stops after ECR publication.

## Reproducibility and Frozen Artifacts

| Item | Frozen identity |
|---|---|
| Git tag / commit | `evaluation-freeze-v1` / `0bd4c96` |
| Development dataset | `development_dialogues_v1`, SHA-256 `5dfaf66eda5311a54bbb028db3130e1ca2aa04712aba28d94865332a6338018d` |
| Held-out dataset | `heldout_test_dialogues_v1`, SHA-256 `552724a9fa86c446148dc8ec498fa6d2e7d27d1e2d42ebe1004848bec1f087c0` |
| Runtime catalogue | v3, 4,933 rows, SHA-256 `ea1a6c0d7ba49864b17db2b2d46bccebdf95612552072c6fb83abddeb6e7014e` |
| Frozen evaluation model | `gpt-5.6-terra` |

Tracked metadata is available in `data/evaluation/`; catalogue exclusions are documented in
`data/processed/laptop_product_database_v3_exclusions.csv`. Raw datasets, runtime Parquet files,
provider predictions, and generated result directories are intentionally excluded from Git.

## Project Report and Evidence

- Implementation and tests: `src/product_discovery/` and `tests/`
- Evaluation method: `evaluation/README.md`
- Dataset/schema metadata: `data/evaluation/`
- Catalogue contamination audit: `data/processed/laptop_product_database_v3_exclusions.csv`
- Prototypes and data preparation: `notebooks/`

No final written project report, standalone architecture figure, deployment screenshot, or AWS
deployment evidence bundle is currently committed. Generated evaluation artifacts may exist
locally under ignored `evaluation/results/` paths but are not part of a clean public checkout.

## Academic Context

- **Subject:** UTS 32040 Industry Project
- **Student:** Haokun Liu
- **Institution:** University of Technology Sydney

## License and Third-Party Data

The original source code in this repository is licensed under the MIT License. See `LICENSE`.

Third-party datasets, product metadata, review content, external model/provider services,
trademarks, and other externally sourced materials are not covered by the MIT License and remain
subject to their original licenses, terms of use, and copyright restrictions.

This project uses data derived from Amazon Reviews 2023 for academic evaluation and
experimentation. Redistribution rights for the original dataset and any substantial reproduced
source content are governed by the original dataset terms, not by this repository's MIT License.
