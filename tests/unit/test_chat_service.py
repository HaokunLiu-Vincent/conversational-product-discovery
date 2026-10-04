from collections import deque
from collections.abc import Sequence

from product_discovery.llm.adapter import LLMAdapter
from product_discovery.preferences.manager import PreferenceStateManager
from product_discovery.products.repository import InMemoryProductRepository
from product_discovery.products.retrieval import RetrievalEngine
from product_discovery.products.search import ProductSearchService
from product_discovery.products.selection import RepresentativeProductSelector
from product_discovery.responses.generator import GroundedResponseGenerator
from product_discovery.schemas import (
    ChatRequest,
    PreferenceState,
    PreferenceUpdate,
    Product,
)
from product_discovery.services.chat import ChatService


class FakeLLMAdapter(LLMAdapter):
    def __init__(self, updates: list[PreferenceUpdate]) -> None:
        self._updates = deque(updates)
        self.extraction_states: list[PreferenceState] = []
        self.grounding_calls: list[tuple[PreferenceState, list[Product]]] = []

    def extract_preference_update(
        self, user_message: str, current_state: PreferenceState
    ) -> PreferenceUpdate:
        del user_message
        self.extraction_states.append(current_state.model_copy(deep=True))
        return self._updates.popleft()

    def generate_grounded_response(
        self,
        state: PreferenceState,
        candidates: Sequence[Product],
    ) -> str:
        self.grounding_calls.append((state.model_copy(deep=True), list(candidates)))
        return "Grounded fake response."


def _products() -> list[Product]:
    return [
        Product(
            product_id="dell-low",
            title="Dell Low",
            brand="Dell",
            price=500,
            ram_gb=8,
            description="Low-priced Dell description.",
        ),
        Product(
            product_id="lenovo-mid",
            title="Lenovo Mid",
            brand="Lenovo",
            price=1000,
            ram_gb=16,
            description="Lenovo description.",
        ),
        Product(
            product_id="dell-mid",
            title="Dell Mid",
            brand="Dell",
            price=1500,
            ram_gb=16,
            description="Mid-priced Dell description.",
        ),
        Product(
            product_id="dell-high",
            title="Dell High",
            brand="Dell",
            price=2000,
            ram_gb=32,
            description="High-priced Dell description.",
        ),
    ]


def _service(adapter: FakeLLMAdapter) -> ChatService:
    search = ProductSearchService(
        retrieval=RetrievalEngine(InMemoryProductRepository(_products())),
        selector=RepresentativeProductSelector(),
        max_candidates=2,
    )
    return ChatService(
        llm=adapter,
        state_manager=PreferenceStateManager(),
        product_search=search,
        response_generator=GroundedResponseGenerator(adapter),
    )


def test_chat_applies_set_and_remove_updates_before_full_retrieval() -> None:
    adapter = FakeLLMAdapter(
        [
            PreferenceUpdate(
                updates=[
                    {"field": "brand", "operation": "set", "value": "Dell"},
                    {"field": "usage", "operation": "set", "value": "gaming"},
                    {
                        "field": "preference_notes",
                        "operation": "set",
                        "value": "Prefers quiet fans.",
                    },
                ]
            ),
            PreferenceUpdate(
                updates=[
                    {"field": "brand", "operation": "remove", "value": None},
                    {"field": "ram_min_gb", "operation": "set", "value": 16},
                ]
            ),
        ]
    )
    service = _service(adapter)

    first = service.chat(ChatRequest(session_id="session", message="A Dell for gaming"))
    second = service.chat(
        ChatRequest(session_id="session", message="Any brand, but make it 16GB RAM")
    )

    assert first.preferences.brand == "Dell"
    assert first.preferences.usage == "gaming"
    assert first.qualified_candidate_count == 3
    assert len(first.representative_products) == 2
    assert second.preferences.brand is None
    assert second.preferences.ram_min_gb == 16
    assert second.preferences.usage == "gaming"
    assert second.qualified_candidate_count == 3
    assert len(second.representative_products) == 2
    assert adapter.extraction_states[1].brand == "Dell"
    grounded_state, grounded_products = adapter.grounding_calls[-1]
    assert grounded_state.preference_notes == "Prefers quiet fans."
    assert all(product.description for product in grounded_products)


def test_chat_applies_safe_updates_and_returns_clarification_without_generation() -> None:
    adapter = FakeLLMAdapter(
        [
            PreferenceUpdate(
                updates=[{"field": "budget_max", "operation": "set", "value": 1600}],
                clarification_needed=True,
                clarification_question="What maximum weight would you like?",
            )
        ]
    )
    service = _service(adapter)

    response = service.chat(
        ChatRequest(session_id="session", message="Under $1600 and fairly light")
    )

    assert response.preferences.budget_max == 1600
    assert response.clarification_needed is True
    assert response.message == "What maximum weight would you like?"
    assert response.qualified_candidate_count == 3
    assert adapter.grounding_calls == []
