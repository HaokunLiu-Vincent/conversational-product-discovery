import pytest
from pydantic import ValidationError

from product_discovery.preferences.validation import PreferenceStateValidator
from product_discovery.schemas import PreferenceState, PreferenceUpdate


def test_applies_normalized_values_and_explicit_removals() -> None:
    current = PreferenceState(brand="Lenovo", ram_min_gb=16, usage="study")
    update = PreferenceUpdate(
        updates=[
            {"field": "budget_max", "operation": "set", "value": "1500"},
            {"field": "gpu_required", "operation": "set", "value": "rtx 4060"},
            {"field": "brand", "operation": "remove", "value": None},
        ]
    )

    result = PreferenceStateValidator().apply(current, update)

    assert result.budget_max == 1500
    assert result.gpu_required == "RTX 4060"
    assert result.brand is None
    assert result.ram_min_gb == 16
    assert result.usage == "study"


def test_rejects_noncanonical_and_out_of_range_values() -> None:
    validator = PreferenceStateValidator()

    with pytest.raises(ValueError, match="brand must be one of"):
        validator.apply(
            PreferenceState(),
            PreferenceUpdate(
                updates=[{"field": "brand", "operation": "set", "value": "Unknown Brand"}]
            ),
        )

    with pytest.raises(ValidationError):
        validator.apply(
            PreferenceState(),
            PreferenceUpdate(updates=[{"field": "ram_min_gb", "operation": "set", "value": 1024}]),
        )
