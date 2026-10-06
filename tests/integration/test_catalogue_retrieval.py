from pathlib import Path

import pandas as pd
import pytest

from product_discovery.products.repository import ParquetProductRepository
from product_discovery.products.retrieval import RetrievalEngine
from product_discovery.schemas import PreferenceState


@pytest.fixture
def catalogue_path(tmp_path: Path) -> Path:
    path = tmp_path / "catalogue.parquet"
    common = {
        "brand": "Lenovo",
        "ram_gb": 16,
        "storage_gb": 512,
        "screen_in": 14.0,
        "gpu": "RTX 4060",
        "weight_kg": 1.5,
    }
    records = [
        {
            **common,
            "product_id": "qualified-cheap",
            "title": "Qualified cheap laptop",
            "price": 999.0,
            "ram_gb": 32,
            "storage_gb": 1024,
            "screen_in": 13.3,
            "weight_kg": 1.2,
        },
        {
            **common,
            "product_id": "qualified-boundary",
            "title": "Qualified boundary laptop",
            "price": 1500.0,
        },
        {
            **common,
            "product_id": "over-budget",
            "title": "Over-budget laptop",
            "price": 1500.01,
        },
        {
            **common,
            "product_id": "wrong-brand",
            "title": "Wrong-brand laptop",
            "brand": "Dell",
            "price": 1100.0,
        },
        {
            **common,
            "product_id": "low-ram",
            "title": "Low-RAM laptop",
            "price": 1150.0,
            "ram_gb": 8,
        },
        {
            **common,
            "product_id": "low-storage",
            "title": "Low-storage laptop",
            "price": 1200.0,
            "storage_gb": 256,
        },
        {
            **common,
            "product_id": "large-screen",
            "title": "Large-screen laptop",
            "price": 1250.0,
            "screen_in": 15.6,
        },
        {
            **common,
            "product_id": "wrong-gpu",
            "title": "Wrong-GPU laptop",
            "price": 1300.0,
            "gpu": "Integrated",
        },
        {
            **common,
            "product_id": "too-heavy",
            "title": "Heavy laptop",
            "price": 1350.0,
            "weight_kg": 1.6,
        },
    ]
    pd.DataFrame(records).to_parquet(path, index=False)
    return path


def test_parquet_retrieval_applies_every_hard_constraint(catalogue_path: Path) -> None:
    engine = RetrievalEngine(ParquetProductRepository(catalogue_path))
    state = PreferenceState(
        budget_max=1500,
        brand="lenovo",
        ram_min_gb=16,
        storage_min_gb=512,
        screen_size_max_in=14,
        gpu_required="rtx 4060",
        weight_max_kg=1.5,
    )

    qualified = engine.retrieve(state)

    assert [product.product_id for product in qualified] == [
        "qualified-cheap",
        "qualified-boundary",
    ]


def test_parquet_retrieval_returns_full_fixture_and_ignores_semantic_context(
    catalogue_path: Path,
) -> None:
    engine = RetrievalEngine(ParquetProductRepository(catalogue_path))

    unconstrained = engine.retrieve(PreferenceState())
    semantic = engine.retrieve(
        PreferenceState(usage="gaming", preference_notes="quiet and understated")
    )

    expected_ids = [
        "qualified-cheap",
        "wrong-brand",
        "low-ram",
        "low-storage",
        "large-screen",
        "wrong-gpu",
        "too-heavy",
        "qualified-boundary",
        "over-budget",
    ]
    assert [product.product_id for product in unconstrained] == expected_ids
    assert [product.product_id for product in semantic] == expected_ids
