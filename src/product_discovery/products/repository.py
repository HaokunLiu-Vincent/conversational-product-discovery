"""Product repository contracts and parquet implementation."""

from collections.abc import Sequence
from pathlib import Path
from threading import RLock
from typing import Protocol

import pandas as pd

from product_discovery.schemas import Product

_REQUIRED_COLUMNS = {
    "product_id",
    "title",
    "brand",
    "price",
    "ram_gb",
    "storage_gb",
    "screen_in",
    "gpu",
    "weight_kg",
}


class ProductRepository(Protocol):
    def list_products(self) -> Sequence[Product]:
        """Return normalized catalogue products."""


class ParquetProductRepository:
    """Lazily load the immutable local catalogue through pandas/pyarrow."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._products: tuple[Product, ...] | None = None
        self._lock = RLock()

    def list_products(self) -> Sequence[Product]:
        with self._lock:
            if self._products is None:
                self._products = self._load()
            return self._products

    def _load(self) -> tuple[Product, ...]:
        if not self._path.is_file():
            raise FileNotFoundError(f"catalogue not found: {self._path}")

        frame = pd.read_parquet(self._path, engine="pyarrow")
        missing = sorted(_REQUIRED_COLUMNS - set(frame.columns))
        if missing:
            raise ValueError(f"catalogue is missing required columns: {', '.join(missing)}")

        normalized = frame.astype(object).where(pd.notna(frame), None)
        records = normalized.to_dict(orient="records")
        return tuple(Product.model_validate(record) for record in records)


class InMemoryProductRepository:
    """Small repository useful for deterministic tests and local composition."""

    def __init__(self, products: Sequence[Product]) -> None:
        self._products = tuple(products)

    def list_products(self) -> Sequence[Product]:
        return self._products
