from fastapi.testclient import TestClient

from product_discovery.api import app, get_chat_service
from product_discovery.schemas import ChatResponse, PreferenceState


def test_health() -> None:
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


class StubChatService:
    def chat(self, request: object) -> ChatResponse:
        del request
        return ChatResponse(
            session_id="api-session",
            message="Grounded response.",
            preferences=PreferenceState(ram_min_gb=16),
            clarification_needed=False,
            clarification_question=None,
            qualified_candidate_count=42,
            representative_products=[],
        )


def test_v1_chat_uses_injected_chat_service() -> None:
    app.dependency_overrides[get_chat_service] = lambda: StubChatService()
    try:
        response = TestClient(app).post(
            "/v1/chat",
            json={"session_id": "api-session", "message": "I need 16GB RAM"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["qualified_candidate_count"] == 42
    assert response.json()["preferences"]["ram_min_gb"] == 16
