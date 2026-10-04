"""Catalogue persistence and deterministic retrieval."""

from product_discovery.products.repository import (
    InMemoryProductRepository,
    ParquetProductRepository,
    ProductRepository,
)
from product_discovery.products.retrieval import RetrievalEngine
from product_discovery.products.search import ProductSearchService
from product_discovery.products.selection import RepresentativeProductSelector

__all__ = [
    "InMemoryProductRepository",
    "ParquetProductRepository",
    "ProductRepository",
    "ProductSearchService",
    "RepresentativeProductSelector",
    "RetrievalEngine",
]
