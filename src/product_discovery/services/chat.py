"""Chat orchestration across LLM and deterministic application boundaries."""

from product_discovery.llm.adapter import LLMAdapter
from product_discovery.preferences.manager import PreferenceStateManager
from product_discovery.products.search import ProductSearchService
from product_discovery.responses.generator import GroundedResponseGenerator
from product_discovery.schemas import ChatRequest, ChatResponse


class ChatService:
    def __init__(
        self,
        llm: LLMAdapter,
        state_manager: PreferenceStateManager,
        product_search: ProductSearchService,
        response_generator: GroundedResponseGenerator,
    ) -> None:
        self._llm = llm
        self._state_manager = state_manager
        self._product_search = product_search
        self._response_generator = response_generator

    def chat(self, request: ChatRequest) -> ChatResponse:
        current = self._state_manager.get(request.session_id)
        proposed_update = self._llm.extract_preference_update(request.message, current)
        state = self._state_manager.apply(request.session_id, proposed_update)
        result = self._product_search.search(state)
        if proposed_update.clarification_needed:
            message = proposed_update.clarification_question
            if message is None:  # Defensive guard; schema validation normally prevents this.
                raise ValueError("clarification response is missing a question")
        else:
            message = self._response_generator.generate(
                state,
                result.representative_products,
            )
        return ChatResponse(
            session_id=request.session_id,
            message=message,
            preferences=state,
            clarification_needed=proposed_update.clarification_needed,
            clarification_question=proposed_update.clarification_question,
            qualified_candidate_count=result.qualified_candidate_count,
            representative_products=result.representative_products,
        )
