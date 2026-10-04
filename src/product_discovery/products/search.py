"""Composition of complete qualification and representative selection."""

from product_discovery.products.retrieval import RetrievalEngine
from product_discovery.products.selection import RepresentativeProductSelector
from product_discovery.schemas import PreferenceState, RetrievalResult


class ProductSearchService:
    """Build an API-facing result without truncating qualification counts."""

    def __init__(
        self,
        retrieval: RetrievalEngine,
        selector: RepresentativeProductSelector,
        max_candidates: int,
    ) -> None:
        if max_candidates < 1:
            raise ValueError("max_candidates must be at least 1")
        self._retrieval = retrieval
        self._selector = selector
        self._max_candidates = max_candidates

    def search(self, state: PreferenceState) -> RetrievalResult:
        qualified = self._retrieval.retrieve(state)
        return self._selector.summarize(qualified, self._max_candidates)
