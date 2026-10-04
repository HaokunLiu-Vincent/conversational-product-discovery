from pathlib import Path

import pandas as pd
import pytest

from product_discovery.products.repository import ParquetProductRepository


def _catalogue_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "product_id": "one",
                "title": "Laptop One",
                "brand": "Dell",
                "price": 1000.0,
                "ram_gb": 16.0,
                "storage_gb": 512.0,
                "screen_in": 14.0,
                "gpu": "Integrated",
                "weight_kg": 1.4,
            },
            {
                "product_id": "two",
                "title": "Laptop Two",
                "brand": "HP",
                "price": 1200.0,
                "ram_gb": 16.0,
                "storage_gb": 512.0,
                "screen_in": 15.6,
                "gpu": None,
                "weight_kg": None,
            },
        ]
    )


def test_parquet_repository_loads_normalizes_and_caches(tmp_path: Path) -> None:
    path = tmp_path / "catalogue.parquet"
    _catalogue_frame().to_parquet(path, index=False)
    repository = ParquetProductRepository(path)

    first = repository.list_products()
    second = repository.list_products()

    assert first is second
    assert [product.product_id for product in first] == ["one", "two"]
    assert first[1].gpu is None
    assert first[1].weight_kg is None


def test_parquet_repository_rejects_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "invalid.parquet"
    _catalogue_frame().drop(columns=["gpu"]).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="missing required columns: gpu"):
        ParquetProductRepository(path).list_products()


def test_parquet_repository_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="catalogue not found"):
        ParquetProductRepository(tmp_path / "missing.parquet").list_products()
