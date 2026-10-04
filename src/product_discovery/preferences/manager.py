"""Process-local preference state ownership."""

from threading import RLock

from product_discovery.preferences.validation import PreferenceStateValidator
from product_discovery.schemas import PreferenceState, PreferenceUpdate


class PreferenceStateManager:
    """Thread-safe in-memory session state for the local application."""

    def __init__(self, validator: PreferenceStateValidator | None = None) -> None:
        self._validator = validator or PreferenceStateValidator()
        self._states: dict[str, PreferenceState] = {}
        self._lock = RLock()

    def get(self, session_id: str) -> PreferenceState:
        with self._lock:
            return self._states.get(session_id, PreferenceState()).model_copy(deep=True)

    def apply(self, session_id: str, update: PreferenceUpdate) -> PreferenceState:
        with self._lock:
            current = self._states.get(session_id, PreferenceState())
            updated = self._validator.apply(current, update)
            self._states[session_id] = updated
            return updated.model_copy(deep=True)

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._states.pop(session_id, None)
