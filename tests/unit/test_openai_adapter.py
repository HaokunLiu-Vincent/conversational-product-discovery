import json
import logging
from types import SimpleNamespace

import pytest
from openai import OpenAIError

from product_discovery.llm.adapter import LLMAdapterError
from product_discovery.llm.openai_adapter import OpenAIAdapter
from product_discovery.schemas import PreferenceState, PreferenceUpdate, Product


class StubResponses:
    def __init__(self) -> None:
        self.parse_arguments: dict[str, object] | None = None
        self.create_arguments: dict[str, object] | None = None

    def parse(self, **kwargs: object) -> SimpleNamespace:
        self.parse_arguments = kwargs
        return SimpleNamespace(
            output_parsed=PreferenceUpdate(
                updates=[{"field": "ram_min_gb", "operation": "set", "value": 16}]
            )
        )

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.create_arguments = kwargs
        return SimpleNamespace(output_text="A grounded response.")


class StubClient:
    def __init__(self) -> None:
        self.responses = StubResponses()


class FailingExtractionResponses(StubResponses):
    def parse(self, **kwargs: object) -> SimpleNamespace:
        del kwargs
        raise OpenAIError("provider unavailable")


class FailingExtractionClient:
    def __init__(self) -> None:
        self.responses = FailingExtractionResponses()


def test_uses_structured_output_for_turn_update_only() -> None:
    client = StubClient()
    adapter = OpenAIAdapter(api_key="unused", model="test-model", client=client)
    current = PreferenceState(brand="Dell")

    update = adapter.extract_preference_update("Make the RAM 16 GB", current)

    assert update.updates[0].field == "ram_min_gb"
    assert client.responses.parse_arguments is not None
    assert client.responses.parse_arguments["text_format"] is PreferenceUpdate
    payload = json.loads(str(client.responses.parse_arguments["input"]))
    assert payload == {
        "current_validated_state": current.model_dump(mode="json"),
        "latest_user_message": "Make the RAM 16 GB",
    }


def test_logs_underlying_preference_extraction_exception(
    caplog: pytest.LogCaptureFixture,
) -> None:
    adapter = OpenAIAdapter(
        api_key="must-not-be-logged",
        model="test-model",
        client=FailingExtractionClient(),
    )

    with (
        caplog.at_level(logging.ERROR, logger="product_discovery.llm.openai_adapter"),
        pytest.raises(LLMAdapterError, match="OpenAI preference extraction failed"),
    ):
        adapter.extract_preference_update("Find me a laptop", PreferenceState())

    assert "OpenAI preference extraction failed" in caplog.text
    assert "provider unavailable" in caplog.text
    assert "must-not-be-logged" not in caplog.text


def test_grounded_response_receives_only_validated_grounding_data() -> None:
    client = StubClient()
    adapter = OpenAIAdapter(api_key="unused", model="test-model", client=client)
    state = PreferenceState(
        budget_max=1500,
        usage="gaming",
        preference_notes="Prefers quiet fans.",
    )
    product = Product(
        product_id="one",
        title="Laptop One",
        price=1200,
        description="A catalogue description.",
    )

    message = adapter.generate_grounded_response(state, [product])

    assert message == "A grounded response."
    assert client.responses.create_arguments is not None
    payload = json.loads(str(client.responses.create_arguments["input"]))
    assert set(payload) == {
        "validated_preference_state",
        "usage",
        "preference_notes",
        "representative_products",
    }
    assert payload["validated_preference_state"]["budget_max"] == 1500
    assert "usage" not in payload["validated_preference_state"]
    assert payload["usage"] == "gaming"
    assert payload["preference_notes"] == "Prefers quiet fans."
    assert payload["representative_products"][0]["description"] == ("A catalogue description.")
    assert client.responses.create_arguments["store"] is False
