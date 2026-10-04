"""Grounded response boundary."""

from collections.abc import Sequence

from product_discovery.llm.adapter import LLMAdapter
from product_discovery.schemas import PreferenceState, Product


class GroundedResponseGenerator:
    def __init__(self, llm: LLMAdapter) -> None:
        self._llm = llm

    def generate(
        self,
        state: PreferenceState,
        candidates: Sequence[Product],
    ) -> str:
        return self._llm.generate_grounded_response(state, candidates)
