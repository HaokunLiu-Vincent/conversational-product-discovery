from product_discovery.products.repository import InMemoryProductRepository
from product_discovery.products.retrieval import RetrievalEngine
from product_discovery.products.search import ProductSearchService
from product_discovery.products.selection import RepresentativeProductSelector
from product_discovery.schemas import PreferenceState, Product


def _priced_products(count: int) -> list[Product]:
    return [
        Product(product_id=str(index), title=f"Laptop {index}", price=float(index * 100))
        for index in range(count)
    ]


def test_selects_low_middle_and_high_price_representatives() -> None:
    selected = RepresentativeProductSelector().select(_priced_products(7), maximum=3)

    assert [product.product_id for product in selected] == ["0", "3", "6"]


def test_max_candidates_does_not_limit_qualified_count() -> None:
    products = _priced_products(9)
    service = ProductSearchService(
        retrieval=RetrievalEngine(InMemoryProductRepository(products)),
        selector=RepresentativeProductSelector(),
        max_candidates=3,
    )

    result = service.search(PreferenceState())

    assert result.qualified_candidate_count == 9
    assert len(result.representative_products) == 3
    assert [product.product_id for product in result.representative_products] == ["0", "4", "8"]


def test_returns_all_products_when_below_maximum() -> None:
    products = _priced_products(2)

    result = RepresentativeProductSelector().summarize(products, maximum=5)

    assert result.qualified_candidate_count == 2
    assert result.representative_products == products
