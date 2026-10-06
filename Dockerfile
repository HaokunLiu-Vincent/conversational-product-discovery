# syntax=docker/dockerfile:1

FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

COPY pyproject.toml README.md ./
COPY src/ src/

RUN python -m pip wheel --wheel-dir /wheels .


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PRODUCT_DISCOVERY_CATALOGUE_PATH=data/processed/laptop_product_database_v3.parquet

WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app app

COPY --from=builder /wheels /wheels
RUN python -m pip install --no-cache-dir --no-index \
        --find-links=/wheels conversational-product-discovery \
    && rm -rf /wheels

COPY --chown=app:app data/processed/laptop_product_database_v3.parquet \
    data/processed/laptop_product_database_v3.parquet

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).read()"

CMD ["uvicorn", "product_discovery.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
