import os

import pytest

from product_discovery.llm.openai_adapter import OpenAIAdapter
from product_discovery.schemas import PreferenceState

API_KEY = os.getenv("PRODUCT_DISCOVERY_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("PRODUCT_DISCOVERY_OPENAI_MODEL", "gpt-6-astra")


@pytest.mark.live_openai
@pytest.mark.skipif(not API_KEY, reason="OpenAI API key is not configured")
def test_real_openai_adapter_returns_only_a_structured_update() -> None:
    adapter = OpenAIAdapter(api_key=API_KEY or "", model=MODEL)

    update = adapter.extract_preference_update(
        "I need at least 16 GB of RAM.",
        PreferenceState(),
    )

    assert update.clarification_needed is False
    assert any(
        item.field == "ram_min_gb" and item.operation == "set" and item.value == 16
        for item in update.updates
    )
