"""Deterministic representative selection from all qualified products."""

from collections.abc import Sequence

from product_discovery.schemas import Product, RetrievalResult


class RepresentativeProductSelector:
    """Sample evenly across the price-ordered qualified candidate set."""

    def select(self, qualified: Sequence[Product], maximum: int) -> list[Product]:
        if maximum < 1:
            raise ValueError("maximum must be at least 1")
        if len(qualified) <= maximum:
            return list(qualified)
        if maximum == 1:
            return [qualified[len(qualified) // 2]]

        last = len(qualified) - 1
        denominator = maximum - 1
        positions = [(index * last + denominator // 2) // denominator for index in range(maximum)]
        return [qualified[position] for position in positions]

    def summarize(self, qualified: Sequence[Product], maximum: int) -> RetrievalResult:
        """Count the full set before limiting products returned upstream."""

        return RetrievalResult(
            qualified_candidate_count=len(qualified),
            representative_products=self.select(qualified, maximum),
        )
