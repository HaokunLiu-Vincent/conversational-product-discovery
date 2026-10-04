from pathlib import Path

from product_discovery.products.repository import ParquetProductRepository
from product_discovery.products.retrieval import RetrievalEngine
from product_discovery.schemas import PreferenceState

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CATALOGUE_PATH = REPOSITORY_ROOT / "data/processed/laptop_product_database_v3.parquet"
EXCLUDED_PRODUCT_IDS = {
    "B0044JKIRI",
    "B01LP6FP3Y",
    "B07CNBY9YY",
    "B07GDBMFZG",
    "B08L15C73G",
}


def test_real_catalogue_matches_development_candidate_count() -> None:
    engine = RetrievalEngine(ParquetProductRepository(CATALOGUE_PATH))

    qualified = engine.retrieve(PreferenceState(budget_max=2000))

    assert len(qualified) == 4876


def test_contaminated_products_are_absent_from_unconstrained_retrieval() -> None:
    engine = RetrievalEngine(ParquetProductRepository(CATALOGUE_PATH))

    qualified = engine.retrieve(PreferenceState())
    retrieved_ids = {product.product_id for product in qualified}

    assert len(qualified) == 4933
    assert retrieved_ids.isdisjoint(EXCLUDED_PRODUCT_IDS)
