from product_discovery.products.repository import InMemoryProductRepository
from product_discovery.products.retrieval import RetrievalEngine
from product_discovery.schemas import PreferenceState, Product


def _products() -> list[Product]:
    return [
        Product(
            product_id="qualified",
            title="Qualified laptop",
            brand="Lenovo",
            price=1200,
            ram_gb=16,
            storage_gb=512,
            screen_in=14,
            gpu="RTX 4060",
            weight_kg=1.5,
        ),
        Product(
            product_id="over-budget",
            title="Expensive laptop",
            brand="Lenovo",
            price=1800,
            ram_gb=32,
            storage_gb=1024,
            screen_in=14,
            gpu="RTX 4060",
            weight_kg=1.7,
        ),
        Product(
            product_id="wrong-gpu",
            title="Integrated laptop",
            brand="Lenovo",
            price=900,
            ram_gb=16,
            storage_gb=512,
            screen_in=13,
            gpu="Integrated",
            weight_kg=1.2,
        ),
    ]


def test_filters_all_hard_constraints() -> None:
    engine = RetrievalEngine(InMemoryProductRepository(_products()))
    state = PreferenceState(
        brand="lenovo",
        budget_max=1500,
        ram_min_gb=16,
        storage_min_gb=512,
        screen_size_max_in=14,
        gpu_required="rtx 4060",
        weight_max_kg=1.5,
    )

    result = engine.retrieve(state)

    assert [product.product_id for product in result] == ["qualified"]


def test_semantic_context_does_not_filter_and_results_are_price_ordered() -> None:
    engine = RetrievalEngine(InMemoryProductRepository(_products()))
    state = PreferenceState(usage="gaming", preference_notes="quiet and understated")

    result = engine.retrieve(state)

    assert [product.product_id for product in result] == [
        "wrong-gpu",
        "qualified",
        "over-budget",
    ]
