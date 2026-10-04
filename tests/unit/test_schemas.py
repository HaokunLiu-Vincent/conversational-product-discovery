import pytest
from pydantic import ValidationError

from product_discovery.schemas import PreferenceState, PreferenceUpdate, PreferenceUpdateItem


def test_preference_state_rejects_unknown_fields_and_invalid_ranges() -> None:
    with pytest.raises(ValidationError):
        PreferenceState.model_validate({"budget_max": 0})

    with pytest.raises(ValidationError):
        PreferenceState.model_validate({"unknown": "value"})


def test_update_operation_and_value_must_agree() -> None:
    with pytest.raises(ValidationError):
        PreferenceUpdateItem(field="brand", operation="set", value=None)

    with pytest.raises(ValidationError):
        PreferenceUpdateItem(field="brand", operation="remove", value="Dell")


def test_update_rejects_duplicate_field_mutations() -> None:
    with pytest.raises(ValidationError):
        PreferenceUpdate(
            updates=[
                {"field": "brand", "operation": "set", "value": "Dell"},
                {"field": "brand", "operation": "remove", "value": None},
            ]
        )


def test_clarification_requires_a_question() -> None:
    with pytest.raises(ValidationError):
        PreferenceUpdate(clarification_needed=True)
