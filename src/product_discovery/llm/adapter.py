"""Provider-neutral LLM boundary."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from product_discovery.schemas import PreferenceState, PreferenceUpdate, Product


class LLMAdapterError(RuntimeError):
    """Raised when a provider cannot produce a usable application response."""


class LLMAdapter(ABC):
    """Translate language to proposals and grounded prose, never application state."""

    @abstractmethod
    def extract_preference_update(
        self, user_message: str, current_state: PreferenceState
    ) -> PreferenceUpdate:
        """Propose a structured update from the user's latest message."""

    @abstractmethod
    def generate_grounded_response(
        self,
        state: PreferenceState,
        candidates: Sequence[Product],
    ) -> str:
        """Generate prose using only state and deterministically retrieved products."""
