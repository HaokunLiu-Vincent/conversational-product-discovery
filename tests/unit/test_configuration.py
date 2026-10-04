from pathlib import Path

import pytest
from pydantic import ValidationError

from product_discovery.configuration import Settings


def test_settings_load_product_discovery_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRODUCT_DISCOVERY_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("PRODUCT_DISCOVERY_CATALOGUE_PATH", "/tmp/catalogue.parquet")
    monkeypatch.setenv("PRODUCT_DISCOVERY_MAX_CANDIDATES", "7")

    settings = Settings(_env_file=None)

    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "test-key"
    assert settings.catalogue_path == Path("/tmp/catalogue.parquet")
    assert settings.max_candidates == 7


def test_settings_reject_invalid_representative_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRODUCT_DISCOVERY_MAX_CANDIDATES", "0")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)
