"""Catalogue persistence and deterministic retrieval."""

from product_discovery.products.repository import (
    InMemoryProductRepository,
    ParquetProductRepository,
    ProductRepository,
)
from product_discovery.products.retrieval import RetrievalEngine
from product_discovery.products.search import ProductSearchService
from product_discovery.products.selection import RepresentativeProductSelector
from product_discovery.products.source import ConfiguredProductRepository

__all__ = [
    "ConfiguredProductRepository",
    "InMemoryProductRepository",
    "ParquetProductRepository",
    "ProductRepository",
    "ProductSearchService",
    "RepresentativeProductSelector",
    "RetrievalEngine",
]
