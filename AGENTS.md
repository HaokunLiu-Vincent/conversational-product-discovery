# Project

This repository implements a lightweight, model-agnostic conversational
product-discovery component for laptop e-commerce.

# Core architecture

LLM:
- interprets natural language
- outputs structured state updates
- generates final grounded responses

Deterministic application logic:
- validates updates
- owns session preference state
- executes product constraints
- retrieves qualified candidates

The LLM must never directly modify persistent state or decide whether
a product satisfies a hard constraint.

# Scope

Do not introduce:
- model training
- fine-tuning
- vector databases
- embedding retrieval
- LangChain
- learned recommendation ranking
- Redis/database session storage

unless explicitly requested.

# Frozen hard constraints

- budget_max
- brand
- ram_min_gb
- storage_min_gb
- screen_size_max_in
- gpu_required
- weight_max_kg

# Semantic context

- usage
- preference_notes

These fields do not perform deterministic catalogue filtering.

# Product data

Use:
data/processed/laptop_product_database_v3.parquet

Runtime Parquet catalogues are provisioned externally and are not stored in Git. Set
PRODUCT_DISCOVERY_CATALOGUE_PATH for local development or PRODUCT_DISCOVERY_CATALOGUE_URI
for S3-backed runtime loading.

Treat v2 as the immutable source catalogue. The v3 exclusion audit is stored at
data/processed/laptop_product_database_v3_exclusions.csv.

Do not modify the product preprocessing pipeline unless explicitly requested.

# Evaluation

Development data may be used for debugging and prompt refinement.

Do not inspect or tune against heldout_test_dialogues_v1.jsonl unless
explicitly asked to run formal evaluation.

# Engineering

Use Python 3.12.
Use Pydantic v2 models for application contracts.
Use FastAPI for the REST API.
Write unit tests for deterministic components.
Keep provider-specific LLM logic behind an adapter interface.
