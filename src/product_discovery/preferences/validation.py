"""Deterministic validation and merging of proposed preference updates."""

from collections.abc import Callable

from product_discovery.schemas import (
    CANONICAL_BRANDS,
    CANONICAL_GPU_VALUES,
    USAGE_VALUES,
    PreferenceState,
    PreferenceUpdate,
    UpdateValue,
)


def _number(value: UpdateValue, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{field} must be numeric")
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be numeric") from exc


def _integer(value: UpdateValue, field: str) -> int:
    number = _number(value, field)
    if not number.is_integer():
        raise ValueError(f"{field} must be a whole number")
    return int(number)


def _text(value: UpdateValue, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-blank text")
    return value.strip()


def _canonical(value: UpdateValue, field: str, allowed: frozenset[str]) -> str:
    text = _text(value, field)
    by_casefold = {item.casefold(): item for item in allowed}
    try:
        return by_casefold[text.casefold()]
    except KeyError as exc:
        raise ValueError(f"{field} must be one of {sorted(allowed)}") from exc


class PreferenceStateValidator:
    """Atomically validate and apply a structured update to existing state."""

    def __init__(self) -> None:
        self._parsers: dict[str, Callable[[UpdateValue], object]] = {
            "budget_max": lambda value: _number(value, "budget_max"),
            "brand": lambda value: _canonical(value, "brand", CANONICAL_BRANDS),
            "ram_min_gb": lambda value: _integer(value, "ram_min_gb"),
            "storage_min_gb": lambda value: _integer(value, "storage_min_gb"),
            "screen_size_max_in": lambda value: _number(value, "screen_size_max_in"),
            "gpu_required": lambda value: _canonical(value, "gpu_required", CANONICAL_GPU_VALUES),
            "weight_max_kg": lambda value: _number(value, "weight_max_kg"),
            "usage": lambda value: _canonical(value, "usage", USAGE_VALUES),
            "preference_notes": lambda value: _text(value, "preference_notes"),
        }

    def apply(self, current: PreferenceState, update: PreferenceUpdate) -> PreferenceState:
        merged = current.model_dump()
        for item in update.updates:
            if item.operation == "remove":
                merged[item.field] = None
            else:
                merged[item.field] = self._parsers[item.field](item.value)

        # Final Pydantic validation enforces numeric ranges and contract types.
        return PreferenceState.model_validate(merged)
