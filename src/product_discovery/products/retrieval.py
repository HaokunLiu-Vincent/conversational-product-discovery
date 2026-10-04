"""Deterministic filtering over frozen hard constraints."""

from math import inf

from product_discovery.products.repository import ProductRepository
from product_discovery.schemas import PreferenceState, Product


class RetrievalEngine:
    """Return every qualified product in deterministic price order."""

    def __init__(self, repository: ProductRepository) -> None:
        self._repository = repository

    def retrieve(self, state: PreferenceState) -> list[Product]:
        qualified = [
            product for product in self._repository.list_products() if self._matches(product, state)
        ]
        qualified.sort(key=self._sort_key)
        return qualified

    @staticmethod
    def _matches(product: Product, state: PreferenceState) -> bool:
        if state.budget_max is not None and (
            product.price is None or product.price > state.budget_max
        ):
            return False
        if state.brand is not None and (
            product.brand is None or product.brand.casefold() != state.brand.casefold()
        ):
            return False
        if state.ram_min_gb is not None and (
            product.ram_gb is None or product.ram_gb < state.ram_min_gb
        ):
            return False
        if state.storage_min_gb is not None and (
            product.storage_gb is None or product.storage_gb < state.storage_min_gb
        ):
            return False
        if state.screen_size_max_in is not None and (
            product.screen_in is None or product.screen_in > state.screen_size_max_in
        ):
            return False
        if state.gpu_required is not None and (
            product.gpu is None or product.gpu.casefold() != state.gpu_required.casefold()
        ):
            return False
        if state.weight_max_kg is not None and (
            product.weight_kg is None or product.weight_kg > state.weight_max_kg
        ):
            return False
        # usage and preference_notes are intentionally not hard filters.
        return True

    @staticmethod
    def _sort_key(product: Product) -> tuple[float, str]:
        return (
            product.price if product.price is not None else inf,
            product.product_id,
        )
