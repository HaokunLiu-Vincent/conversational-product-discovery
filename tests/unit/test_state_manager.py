import pytest
from pydantic import ValidationError

from product_discovery.preferences.manager import PreferenceStateManager
from product_discovery.schemas import PreferenceUpdate


def _update(field: str, value: object) -> PreferenceUpdate:
    return PreferenceUpdate.model_validate(
        {"updates": [{"field": field, "operation": "set", "value": value}]}
    )


def test_manager_isolates_sessions_and_returned_state() -> None:
    manager = PreferenceStateManager()

    first = manager.apply("one", _update("brand", "Dell"))
    first.brand = "HP"

    assert manager.get("one").brand == "Dell"
    assert manager.get("two").brand is None


def test_manager_does_not_store_an_invalid_update() -> None:
    manager = PreferenceStateManager()
    manager.apply("session", _update("ram_min_gb", 16))

    with pytest.raises(ValidationError):
        manager.apply("session", _update("ram_min_gb", 1024))

    assert manager.get("session").ram_min_gb == 16


def test_manager_reset_removes_session_state() -> None:
    manager = PreferenceStateManager()
    manager.apply("session", _update("brand", "Dell"))

    manager.reset("session")

    assert manager.get("session").brand is None
