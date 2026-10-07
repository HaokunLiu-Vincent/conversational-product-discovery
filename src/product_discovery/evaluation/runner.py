"""Deterministic scoring and artifact generation for development evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from openai import OpenAI, OpenAIError
from pydantic import ValidationError

from product_discovery.llm.adapter import LLMAdapterError
from product_discovery.llm.openai_adapter import OpenAIAdapter
from product_discovery.preferences.manager import PreferenceStateManager
from product_discovery.preferences.validation import PreferenceStateValidator
from product_discovery.schemas import PreferenceState, PreferenceUpdate

EXECUTABLE_FIELDS = (
    "budget_max",
    "brand",
    "ram_min_gb",
    "storage_min_gb",
    "screen_size_max_in",
    "gpu_required",
    "weight_max_kg",
)
CORE_FIELDS = (*EXECUTABLE_FIELDS, "usage")
FREE_TEXT_FIELD = "preference_notes"
STATE_FIELDS = (*CORE_FIELDS, FREE_TEXT_FIELD)
CORRECTNESS_FIELDS = (
    "field_correct",
    "operation_correct",
    "value_correct",
    "exact_update_correct",
    "clarification_needed_correct",
    "clarification_question_exact",
    "exact_structured_output_correct",
    "state_exact",
    "active_field_set_exact",
    "core_update_exact",
    "executable_update_exact",
    "core_state_exact",
    "executable_state_exact",
    "executable_active_field_set_exact",
)

STRUCTURAL_ERROR_CATEGORIES = frozenset(
    {
        "extra_structured_field",
        "missing_structured_field",
        "wrong_structured_value",
        "wrong_operation",
        "brand_normalization",
        "other_structural",
    }
)


@dataclass(frozen=True)
class ExtractionResult:
    """One structured extraction plus non-secret provider evidence."""

    update: PreferenceUpdate
    raw_model_output: str | None = None
    usage: dict[str, Any] | None = None
    provider_response_id: str | None = None


class PreferenceExtractor(Protocol):
    """Evaluation boundary for live and mocked structured extraction."""

    def extract(self, user_text: str, state: PreferenceState) -> ExtractionResult:
        """Return one proposed update without mutating state."""


class _RecordingResponses:
    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self.raw_model_output: str | None = None
        self.usage: dict[str, Any] | None = None
        self.response_id: str | None = None

    def reset(self) -> None:
        self.raw_model_output = None
        self.usage = None
        self.response_id = None

    def parse(self, **kwargs: Any) -> Any:
        response = self._delegate.parse(**kwargs)
        output_text = getattr(response, "output_text", None)
        if output_text:
            self.raw_model_output = str(output_text)
        elif getattr(response, "output_parsed", None) is not None:
            parsed = response.output_parsed
            payload = parsed.model_dump(mode="json") if hasattr(parsed, "model_dump") else parsed
            self.raw_model_output = json.dumps(payload, sort_keys=True)
        self.usage = _json_mapping(getattr(response, "usage", None))
        response_id = getattr(response, "id", None)
        self.response_id = str(response_id) if response_id is not None else None
        return response


class _RecordingClient:
    def __init__(self, client: Any) -> None:
        self.responses = _RecordingResponses(client.responses)


class OpenAIPreferenceExtractor:
    """Record non-secret output metadata around the production OpenAI adapter."""

    def __init__(self, api_key: str, model: str) -> None:
        self._client = _RecordingClient(OpenAI(api_key=api_key))
        self._adapter = OpenAIAdapter(api_key=api_key, model=model, client=self._client)

    def extract(self, user_text: str, state: PreferenceState) -> ExtractionResult:
        self._client.responses.reset()
        update = self._adapter.extract_preference_update(user_text, state)
        return ExtractionResult(
            update=update,
            raw_model_output=self._client.responses.raw_model_output,
            usage=self._client.responses.usage,
            provider_response_id=self._client.responses.response_id,
        )


def load_development_dialogues(path: Path) -> list[dict[str, Any]]:
    """Load and validate only a development dialogue JSONL file."""

    if "heldout" in path.name.casefold():
        raise ValueError("held-out evaluation inputs are not supported by this runner")

    dialogues = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    _validate_development_dialogues(dialogues)
    return dialogues


def evaluate_dialogues(
    dialogues: Sequence[Mapping[str, Any]],
    extractor: PreferenceExtractor,
    *,
    metadata: Mapping[str, Any],
    on_turn: Callable[[int, int, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Evaluate structured updates and accumulated state without retrieval."""

    _validate_development_dialogues(dialogues)
    manager = PreferenceStateManager()
    records: list[dict[str, Any]] = []
    drift_origins: dict[str, dict[str, int]] = defaultdict(dict)
    total_turns = sum(len(dialogue["turns"]) for dialogue in dialogues)
    completed = 0

    for dialogue in dialogues:
        dialogue_id = str(dialogue["dialogue_id"])
        for turn in dialogue["turns"]:
            completed += 1
            expected_update = _expected_update(turn)
            expected_before = PreferenceState.model_validate(turn["oracle_state_before"])
            expected_after = PreferenceState.model_validate(turn["oracle_state_after"])
            predicted_before = manager.get(dialogue_id)
            started = time.perf_counter()
            result: ExtractionResult | None = None
            predicted_update: PreferenceUpdate | None = None
            inference_succeeded = False
            validation_succeeded = False
            failure_type: str | None = None
            error_type: str | None = None
            error_message: str | None = None

            try:
                result = extractor.extract(str(turn["user_text"]), predicted_before)
                predicted_update = result.update
                inference_succeeded = True
            except Exception as exc:  # Evaluation must preserve later turns and failure evidence.
                failure_type = _extraction_failure_type(exc)
                error_type = type(exc).__name__
                error_message = _safe_error_message(exc, failure_type)

            if predicted_update is not None:
                try:
                    predicted_after = manager.apply(dialogue_id, predicted_update)
                    validation_succeeded = True
                except (ValidationError, ValueError) as exc:
                    predicted_after = manager.get(dialogue_id)
                    failure_type = "validation_failure"
                    error_type = type(exc).__name__
                    error_message = "The predicted update failed deterministic validation."
            else:
                predicted_after = manager.get(dialogue_id)

            latency_ms = round((time.perf_counter() - started) * 1000, 3)
            expected_items = _canonical_update_items(expected_update, expected_after)
            predicted_items = (
                _canonical_update_items(predicted_update, predicted_after)
                if validation_succeeded and predicted_update is not None
                else None
            )
            correctness = _correctness(
                expected_update=expected_update,
                predicted_update=predicted_update,
                expected_items=expected_items,
                predicted_items=predicted_items,
                expected_state=expected_after,
                predicted_state=predicted_after,
                inference_succeeded=inference_succeeded,
                validation_succeeded=validation_succeeded,
            )
            drifting_fields = _drifting_fields(expected_after, predicted_after)
            dialogue_drift_origins = drift_origins[dialogue_id]
            for field in STATE_FIELDS:
                if field in drifting_fields:
                    dialogue_drift_origins.setdefault(field, int(turn["turn_index"]))
                else:
                    dialogue_drift_origins.pop(field, None)
            first_divergence_turn = (
                min(dialogue_drift_origins[field] for field in drifting_fields)
                if drifting_fields
                else None
            )
            primary_category, secondary_categories = _classify_error(
                expected_update=expected_update,
                predicted_update=predicted_update,
                expected_items=expected_items,
                predicted_items=predicted_items,
                correctness=correctness,
                drifting_fields=drifting_fields,
                failure_type=failure_type,
            )
            record = {
                "dialogue_id": dialogue_id,
                "recipe": dialogue["recipe"],
                "source": dialogue["source"],
                "scenario_type": turn["scenario_type"],
                "turn_index": turn["turn_index"],
                "user_text": turn["user_text"],
                "expected_update": expected_update.model_dump(mode="json"),
                "predicted_update": (
                    predicted_update.model_dump(mode="json")
                    if predicted_update is not None
                    else None
                ),
                "canonical_expected_update": expected_items,
                "canonical_predicted_update": predicted_items,
                "raw_model_output": result.raw_model_output if result is not None else None,
                "expected_state_before": expected_before.model_dump(mode="json"),
                "predicted_state_before": predicted_before.model_dump(mode="json"),
                "expected_resulting_state": expected_after.model_dump(mode="json"),
                "predicted_resulting_state": predicted_after.model_dump(mode="json"),
                "correctness": correctness,
                "primary_error_category": primary_category,
                "secondary_error_categories": secondary_categories,
                "drifting_fields": drifting_fields,
                "first_state_divergence_turn": first_divergence_turn,
                "drift_free_text_only": drifting_fields == [FREE_TEXT_FIELD],
                "drift_affects_core_structured": any(
                    field in CORE_FIELDS for field in drifting_fields
                ),
                "drift_affects_executable": any(
                    field in EXECUTABLE_FIELDS for field in drifting_fields
                ),
                "free_text_only_update_mismatch": (
                    not correctness["exact_update_correct"] and correctness["core_update_exact"]
                ),
                "structural_update_mismatch": not correctness["core_update_exact"],
                "inference_succeeded": inference_succeeded,
                "validation_succeeded": validation_succeeded,
                "latency_ms": latency_ms,
                "usage": result.usage if result is not None else None,
                "provider_response_id": (
                    result.provider_response_id if result is not None else None
                ),
                "failure_type": failure_type,
                "error_type": error_type,
                "error_message": error_message,
                "oracle_candidate_count": turn.get("oracle_candidate_count"),
                "annotation_notes": turn.get("annotation_notes"),
            }
            records.append(record)
            if on_turn is not None:
                on_turn(completed, total_turns, record)

    metrics = _build_metrics(dialogues, records, metadata)
    errors = [
        record
        for record in records
        if record["failure_type"] is not None
        or not record["correctness"]["exact_structured_output_correct"]
        or not record["correctness"]["state_exact"]
    ]
    return {"predictions": records, "errors": errors, "metrics": metrics}


def write_evaluation_artifacts(report: Mapping[str, Any], output_dir: Path) -> list[Path]:
    """Atomically write reusable predictions, metrics, errors, and summary artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / "dev_predictions.jsonl",
        output_dir / "dev_metrics.json",
        output_dir / "dev_metrics.csv",
        output_dir / "dev_errors.jsonl",
        output_dir / "dev_summary.md",
        output_dir / "dev_error_taxonomy.csv",
        output_dir / "dev_cascading_drift.csv",
    ]
    _write_text_atomic(paths[0], _jsonl(report["predictions"]))
    _write_text_atomic(paths[1], json.dumps(report["metrics"], indent=2, sort_keys=True) + "\n")
    _write_metrics_csv(paths[2], report["metrics"])
    _write_text_atomic(paths[3], _jsonl(report["errors"]))
    _write_text_atomic(paths[4], _summary_markdown(report["metrics"]))
    _write_error_taxonomy_csv(paths[5], report["errors"])
    _write_cascading_drift_csv(paths[6], report["errors"])
    return paths


def file_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _validate_development_dialogues(dialogues: Sequence[Mapping[str, Any]]) -> None:
    if not dialogues:
        raise ValueError("development evaluation dataset is empty")
    ids: set[str] = set()
    validator = PreferenceStateValidator()
    for dialogue in dialogues:
        if dialogue.get("split") != "development":
            raise ValueError("evaluation runner accepts only split=development")
        dialogue_id = str(dialogue["dialogue_id"])
        if dialogue_id in ids:
            raise ValueError(f"duplicate dialogue_id: {dialogue_id}")
        ids.add(dialogue_id)
        previous = PreferenceState()
        for expected_index, turn in enumerate(dialogue["turns"], start=1):
            if turn["turn_index"] != expected_index:
                raise ValueError(f"non-sequential turn index in {dialogue_id}")
            before = PreferenceState.model_validate(turn["oracle_state_before"])
            after = PreferenceState.model_validate(turn["oracle_state_after"])
            if before != previous:
                raise ValueError(
                    f"oracle state chain mismatch in {dialogue_id} turn {expected_index}"
                )
            calculated = validator.apply(before, _expected_update(turn))
            if calculated != after:
                raise ValueError(f"oracle update mismatch in {dialogue_id} turn {expected_index}")
            previous = after


def _expected_update(turn: Mapping[str, Any]) -> PreferenceUpdate:
    return PreferenceUpdate.model_validate(
        {
            "updates": turn["oracle_update"],
            "clarification_needed": turn["oracle_clarification_needed"],
            "clarification_question": turn["oracle_clarification_question"],
        }
    )


def _canonical_update_items(
    update: PreferenceUpdate,
    resulting_state: PreferenceState,
) -> list[dict[str, Any]]:
    return sorted(
        (
            {
                "field": item.field,
                "operation": item.operation,
                "value": (
                    None if item.operation == "remove" else getattr(resulting_state, item.field)
                ),
            }
            for item in update.updates
        ),
        key=lambda item: item["field"],
    )


def _correctness(
    *,
    expected_update: PreferenceUpdate,
    predicted_update: PreferenceUpdate | None,
    expected_items: list[dict[str, Any]],
    predicted_items: list[dict[str, Any]] | None,
    expected_state: PreferenceState,
    predicted_state: PreferenceState,
    inference_succeeded: bool,
    validation_succeeded: bool,
) -> dict[str, Any]:
    expected_fields = sorted(item["field"] for item in expected_items)
    predicted_fields = (
        sorted(item["field"] for item in predicted_items) if predicted_items is not None else []
    )
    expected_operations = sorted(item["operation"] for item in expected_items) or ["no-op"]
    predicted_operations = (
        (sorted(item["operation"] for item in predicted_items) or ["no-op"])
        if predicted_items is not None
        else []
    )
    expected_values = {item["field"]: item["value"] for item in expected_items}
    predicted_values = (
        {item["field"]: item["value"] for item in predicted_items}
        if predicted_items is not None
        else {}
    )
    clarification_needed_correct = (
        predicted_update is not None
        and predicted_update.clarification_needed == expected_update.clarification_needed
    )
    clarification_question_exact = (
        predicted_update is not None
        and predicted_update.clarification_question == expected_update.clarification_question
    )
    exact_update_correct = predicted_items == expected_items
    core_update_exact = _filtered_update_exact(
        expected_items,
        predicted_items,
        CORE_FIELDS,
    )
    executable_update_exact = _filtered_update_exact(
        expected_items,
        predicted_items,
        EXECUTABLE_FIELDS,
    )
    expected_dump = expected_state.model_dump(mode="json")
    predicted_dump = predicted_state.model_dump(mode="json")
    expected_active = sorted(key for key, value in expected_dump.items() if value is not None)
    predicted_active = sorted(key for key, value in predicted_dump.items() if value is not None)
    state_field_matches = sum(
        expected_dump[field] == predicted_dump[field] for field in STATE_FIELDS
    )
    active_status_matches = sum(
        (expected_dump[field] is not None) == (predicted_dump[field] is not None)
        for field in STATE_FIELDS
    )
    core_state_field_matches = sum(
        expected_dump[field] == predicted_dump[field] for field in CORE_FIELDS
    )
    executable_field_matches = sum(
        expected_dump[field] == predicted_dump[field] for field in EXECUTABLE_FIELDS
    )
    executable_expected_active = sorted(
        field for field in EXECUTABLE_FIELDS if expected_dump[field] is not None
    )
    executable_predicted_active = sorted(
        field for field in EXECUTABLE_FIELDS if predicted_dump[field] is not None
    )
    exact_structured = (
        inference_succeeded
        and validation_succeeded
        and exact_update_correct
        and clarification_needed_correct
        and clarification_question_exact
    )
    return {
        "field_correct": predicted_items is not None and predicted_fields == expected_fields,
        "operation_correct": (
            predicted_items is not None and predicted_operations == expected_operations
        ),
        "value_correct": predicted_items is not None and predicted_values == expected_values,
        "exact_update_correct": predicted_items is not None and exact_update_correct,
        "clarification_needed_correct": clarification_needed_correct,
        "clarification_question_exact": clarification_question_exact,
        "exact_structured_output_correct": exact_structured,
        "state_exact": predicted_state == expected_state,
        "active_field_set_exact": predicted_active == expected_active,
        "core_update_exact": core_update_exact,
        "executable_update_exact": executable_update_exact,
        "core_state_exact": all(
            expected_dump[field] == predicted_dump[field] for field in CORE_FIELDS
        ),
        "executable_state_exact": all(
            expected_dump[field] == predicted_dump[field] for field in EXECUTABLE_FIELDS
        ),
        "executable_active_field_set_exact": (
            executable_expected_active == executable_predicted_active
        ),
        "state_field_matches": state_field_matches,
        "active_field_status_matches": active_status_matches,
        "field_count": len(STATE_FIELDS),
        "core_state_field_matches": core_state_field_matches,
        "core_field_count": len(CORE_FIELDS),
        "executable_field_matches": executable_field_matches,
        "executable_field_count": len(EXECUTABLE_FIELDS),
    }


def _filtered_update_exact(
    expected_items: Sequence[Mapping[str, Any]],
    predicted_items: Sequence[Mapping[str, Any]] | None,
    fields: Sequence[str],
) -> bool:
    if predicted_items is None:
        return False
    included = frozenset(fields)
    expected = [item for item in expected_items if item["field"] in included]
    predicted = [item for item in predicted_items if item["field"] in included]
    return predicted == expected


def _drifting_fields(
    expected_state: PreferenceState,
    predicted_state: PreferenceState,
) -> list[str]:
    expected = expected_state.model_dump(mode="json")
    predicted = predicted_state.model_dump(mode="json")
    return [field for field in STATE_FIELDS if expected[field] != predicted[field]]


def _classify_error(
    *,
    expected_update: PreferenceUpdate,
    predicted_update: PreferenceUpdate | None,
    expected_items: Sequence[Mapping[str, Any]],
    predicted_items: Sequence[Mapping[str, Any]] | None,
    correctness: Mapping[str, Any],
    drifting_fields: Sequence[str],
    failure_type: str | None,
) -> tuple[str | None, list[str]]:
    categories: list[str] = []

    if failure_type is not None or predicted_items is None or predicted_update is None:
        categories.append("other_structural")
    else:
        expected_structured = {
            str(item["field"]): item for item in expected_items if item["field"] in CORE_FIELDS
        }
        predicted_structured = {
            str(item["field"]): item for item in predicted_items if item["field"] in CORE_FIELDS
        }
        extra_fields = set(predicted_structured) - set(expected_structured)
        missing_fields = set(expected_structured) - set(predicted_structured)
        shared_fields = set(expected_structured) & set(predicted_structured)

        expected_brand = expected_structured.get("brand")
        predicted_brand = predicted_structured.get("brand")
        if expected_brand != predicted_brand and (
            expected_brand is not None or predicted_brand is not None
        ):
            categories.append("brand_normalization")
        if extra_fields:
            categories.append("extra_structured_field")
        if missing_fields:
            categories.append("missing_structured_field")
        if any(
            expected_structured[field]["operation"] != predicted_structured[field]["operation"]
            for field in shared_fields
        ) or (not expected_structured) != (not predicted_structured):
            categories.append("wrong_operation")
        if any(
            expected_structured[field]["operation"] == predicted_structured[field]["operation"]
            and expected_structured[field]["value"] != predicted_structured[field]["value"]
            for field in shared_fields
        ):
            categories.append("wrong_structured_value")
        if not correctness["clarification_needed_correct"]:
            categories.append("other_structural")

    if (
        correctness["clarification_needed_correct"]
        and not correctness["clarification_question_exact"]
    ):
        categories.append("clarification_text_only")

    if correctness["exact_update_correct"] and not correctness["state_exact"]:
        categories.append("cascading_state_drift")
        if list(drifting_fields) == [FREE_TEXT_FIELD]:
            categories.append("free_text_only")
    elif not correctness["exact_update_correct"] and correctness["core_update_exact"]:
        categories.append("free_text_only")

    if not correctness["core_update_exact"] and not any(
        category in STRUCTURAL_ERROR_CATEGORIES for category in categories
    ):
        categories.append("other_structural")

    categories = list(dict.fromkeys(categories))
    if not categories:
        return None, []

    precedence = (
        "cascading_state_drift",
        "clarification_text_only",
        "free_text_only",
        "brand_normalization",
        "extra_structured_field",
        "missing_structured_field",
        "wrong_operation",
        "wrong_structured_value",
        "other_structural",
    )
    primary = next(category for category in precedence if category in categories)
    secondary = [category for category in categories if category != primary]
    return primary, secondary


def _extraction_failure_type(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return "parsing_failure"
    if isinstance(exc, LLMAdapterError) and isinstance(exc.__cause__, OpenAIError):
        return "api_provider_failure"
    return "extraction_failure"


def _safe_error_message(exc: Exception, failure_type: str) -> str:
    if failure_type == "api_provider_failure":
        return "The OpenAI provider request failed."
    if failure_type == "parsing_failure":
        return "The structured preference update could not be parsed."
    if isinstance(exc, LLMAdapterError):
        return str(exc)
    return "Preference extraction failed."


def _build_metrics(
    dialogues: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    by_scenario_records: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_turn_records: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_dialogue_records: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_scenario_records[str(record["scenario_type"])].append(record)
        by_turn_records[str(record["turn_index"])].append(record)
        by_dialogue_records[str(record["dialogue_id"])].append(record)

    overall = _aggregate_records(records)
    final_correct = _final_state_count(by_dialogue_records, "state_exact")
    core_final_correct = _final_state_count(by_dialogue_records, "core_state_exact")
    executable_final_correct = _final_state_count(
        by_dialogue_records,
        "executable_state_exact",
    )
    overall.update(
        {
            "total_dialogues": len(dialogues),
            "final_dialogue_state_exact_count": final_correct,
            "final_dialogue_state_exact_accuracy": _rate(final_correct, len(dialogues)),
            "core_final_dialogue_state_exact_count": core_final_correct,
            "core_final_dialogue_state_exact_accuracy": _rate(
                core_final_correct,
                len(dialogues),
            ),
            "executable_final_dialogue_state_exact_count": executable_final_correct,
            "executable_final_dialogue_state_exact_accuracy": _rate(
                executable_final_correct,
                len(dialogues),
            ),
        }
    )
    by_dialogue: dict[str, Any] = {}
    for dialogue_id, dialogue_records in sorted(by_dialogue_records.items()):
        aggregate = _aggregate_records(dialogue_records)
        aggregate["final_state_exact"] = dialogue_records[-1]["correctness"]["state_exact"]
        aggregate["core_final_state_exact"] = dialogue_records[-1]["correctness"][
            "core_state_exact"
        ]
        aggregate["executable_final_state_exact"] = dialogue_records[-1]["correctness"][
            "executable_state_exact"
        ]
        by_dialogue[dialogue_id] = aggregate

    return {
        "reproducibility": dict(metadata),
        "overall": overall,
        "by_scenario": {
            key: _aggregate_records(value) for key, value in sorted(by_scenario_records.items())
        },
        "by_turn_index": {
            key: _aggregate_records(value)
            for key, value in sorted(by_turn_records.items(), key=lambda item: int(item[0]))
        },
        "by_dialogue": by_dialogue,
        "response_generation_evaluation": {
            "automated_faithfulness_supported": False,
            "reason": (
                "The development benchmark has preference-state and clarification ground truth "
                "but no expected responses, supported claims, or reference product sets."
            ),
        },
        "estimated_cost": None,
        "estimated_cost_reason": "No explicit model pricing configuration is present.",
    }


def _final_state_count(
    by_dialogue_records: Mapping[str, Sequence[Mapping[str, Any]]],
    correctness_field: str,
) -> int:
    return sum(
        bool(dialogue_records[-1]["correctness"][correctness_field])
        for dialogue_records in by_dialogue_records.values()
    )


def _aggregate_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(records)
    counts = {
        field: sum(bool(record["correctness"][field]) for record in records)
        for field in CORRECTNESS_FIELDS
    }
    failure_counts = Counter(
        str(record["failure_type"]) for record in records if record["failure_type"] is not None
    )
    state_matches = sum(int(record["correctness"]["state_field_matches"]) for record in records)
    active_matches = sum(
        int(record["correctness"]["active_field_status_matches"]) for record in records
    )
    field_total = sum(int(record["correctness"]["field_count"]) for record in records)
    core_state_matches = sum(
        int(record["correctness"]["core_state_field_matches"]) for record in records
    )
    core_field_total = sum(int(record["correctness"]["core_field_count"]) for record in records)
    executable_matches = sum(
        int(record["correctness"]["executable_field_matches"]) for record in records
    )
    executable_field_total = sum(
        int(record["correctness"]["executable_field_count"]) for record in records
    )
    primary_categories = Counter(
        str(record["primary_error_category"])
        for record in records
        if record["primary_error_category"] is not None
    )
    category_occurrences: Counter[str] = Counter()
    for record in records:
        categories = [record["primary_error_category"], *record["secondary_error_categories"]]
        category_occurrences.update(
            str(category) for category in categories if category is not None
        )
    latencies = [float(record["latency_ms"]) for record in records]
    successful = sum(bool(record["inference_succeeded"]) for record in records)
    result: dict[str, Any] = {
        "total_turns": total,
        "successful_inference_calls": successful,
        "failed_inference_calls": total - successful,
        "validation_failures": failure_counts.get("validation_failure", 0),
        "extraction_failures": failure_counts.get("extraction_failure", 0),
        "parsing_failures": failure_counts.get("parsing_failure", 0),
        "api_provider_failures": failure_counts.get("api_provider_failure", 0),
        "state_field_value_accuracy": _rate(state_matches, field_total),
        "core_state_field_value_accuracy": _rate(core_state_matches, core_field_total),
        "executable_field_value_accuracy": _rate(
            executable_matches,
            executable_field_total,
        ),
        "active_field_status_accuracy": _rate(active_matches, field_total),
        "strict_update_mismatch_count": sum(
            not record["correctness"]["exact_update_correct"] for record in records
        ),
        "core_update_mismatch_count": sum(
            not record["correctness"]["core_update_exact"] for record in records
        ),
        "executable_update_mismatch_count": sum(
            not record["correctness"]["executable_update_exact"] for record in records
        ),
        "free_text_only_mismatch_count": sum(
            bool(record["free_text_only_update_mismatch"]) for record in records
        ),
        "structural_mismatch_count": sum(
            bool(record["structural_update_mismatch"]) for record in records
        ),
        "cascading_state_drift_count": sum(
            record["primary_error_category"] == "cascading_state_drift" for record in records
        ),
        "primary_error_category_counts": dict(sorted(primary_categories.items())),
        "error_category_occurrence_counts": dict(sorted(category_occurrences.items())),
        "latency_ms": _latency_stats(latencies),
        "token_usage": _aggregate_usage(records),
    }
    for field, count in counts.items():
        result[field.removesuffix("_correct") + "_count"] = count
        result[field.removesuffix("_correct") + "_accuracy"] = _rate(count, total)
    return result


def _aggregate_usage(records: Sequence[Mapping[str, Any]]) -> dict[str, int] | None:
    totals: Counter[str] = Counter()
    for record in records:
        usage = record.get("usage")
        if not isinstance(usage, Mapping):
            continue
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                totals[key] += value
    return dict(totals) if totals else None


def _latency_stats(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "mean": round(statistics.fmean(ordered), 3),
        "median": round(statistics.median(ordered), 3),
        "p95": round(_percentile(ordered, 0.95), 3),
        "min": round(ordered[0], 3),
        "max": round(ordered[-1], 3),
    }


def _percentile(ordered: Sequence[float], quantile: float) -> float:
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _json_mapping(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json")
        return dict(dumped) if isinstance(dumped, Mapping) else None
    return dict(value) if isinstance(value, Mapping) else None


def _jsonl(records: Sequence[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)


def _write_text_atomic(path: Path, content: str) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as temporary:
        temporary.write(content)
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def _write_metrics_csv(path: Path, metrics: Mapping[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for scope, groups in (
        ("overall", {"all": metrics["overall"]}),
        ("scenario", metrics["by_scenario"]),
        ("turn_index", metrics["by_turn_index"]),
        ("dialogue", metrics["by_dialogue"]),
    ):
        for group, values in groups.items():
            latency = values.get("latency_ms") or {}
            row = {"scope": scope, "group": group}
            row.update(
                {key: value for key, value in values.items() if not isinstance(value, (dict, list))}
            )
            row.update({f"latency_ms_{key}": value for key, value in latency.items()})
            rows.append(row)
    fieldnames = sorted({key for row in rows for key in row}, key=lambda key: (key != "scope", key))
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as temporary:
        writer = csv.DictWriter(temporary, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def _write_error_taxonomy_csv(
    path: Path,
    errors: Sequence[Mapping[str, Any]],
) -> None:
    rows = [_error_csv_row(record) for record in errors]
    _write_dict_rows_atomic(path, rows, _error_csv_fieldnames())


def _write_cascading_drift_csv(
    path: Path,
    errors: Sequence[Mapping[str, Any]],
) -> None:
    rows = [
        _error_csv_row(record)
        for record in errors
        if record["primary_error_category"] == "cascading_state_drift"
    ]
    _write_dict_rows_atomic(path, rows, _error_csv_fieldnames())


def _error_csv_row(record: Mapping[str, Any]) -> dict[str, Any]:
    correctness = record["correctness"]
    return {
        "dialogue_id": record["dialogue_id"],
        "turn_index": record["turn_index"],
        "scenario": record["scenario_type"],
        "expected_update": json.dumps(record["expected_update"], sort_keys=True),
        "predicted_update": json.dumps(record["predicted_update"], sort_keys=True),
        "expected_state": json.dumps(record["expected_resulting_state"], sort_keys=True),
        "predicted_state": json.dumps(record["predicted_resulting_state"], sort_keys=True),
        "strict_update_match": correctness["exact_update_correct"],
        "core_update_match": correctness["core_update_exact"],
        "executable_update_match": correctness["executable_update_exact"],
        "strict_state_match": correctness["state_exact"],
        "core_state_match": correctness["core_state_exact"],
        "executable_state_match": correctness["executable_state_exact"],
        "primary_error_category": record["primary_error_category"],
        "secondary_error_categories": json.dumps(record["secondary_error_categories"]),
        "drifting_fields": json.dumps(record["drifting_fields"]),
        "first_state_divergence_turn": record["first_state_divergence_turn"],
        "drift_free_text_only": record["drift_free_text_only"],
        "drift_affects_core_structured": record["drift_affects_core_structured"],
        "drift_affects_executable": record["drift_affects_executable"],
    }


def _error_csv_fieldnames() -> list[str]:
    return [
        "dialogue_id",
        "turn_index",
        "scenario",
        "expected_update",
        "predicted_update",
        "expected_state",
        "predicted_state",
        "strict_update_match",
        "core_update_match",
        "executable_update_match",
        "strict_state_match",
        "core_state_match",
        "executable_state_match",
        "primary_error_category",
        "secondary_error_categories",
        "drifting_fields",
        "first_state_divergence_turn",
        "drift_free_text_only",
        "drift_affects_core_structured",
        "drift_affects_executable",
    ]


def _write_dict_rows_atomic(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as temporary:
        writer = csv.DictWriter(temporary, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def _summary_markdown(metrics: Mapping[str, Any]) -> str:
    overall = metrics["overall"]
    reproducibility = metrics["reproducibility"]
    lines = [
        "# Development Evaluation Summary",
        "",
        f"- Evaluated at: {reproducibility['evaluation_timestamp_utc']}",
        f"- Git commit: `{reproducibility['git_commit_sha']}`",
        f"- Dataset: `{reproducibility['dataset_identifier']}`",
        f"- Model: `{reproducibility['openai_model']}`",
        f"- Dialogues: {overall['total_dialogues']}",
        f"- Turns: {overall['total_turns']}",
        f"- Successful inference calls: {overall['successful_inference_calls']}",
        "",
        "## Core metrics",
        "",
        "- Exact structured output accuracy: "
        f"{_percent(overall['exact_structured_output_accuracy'])}",
        f"- Exact update accuracy: {_percent(overall['exact_update_accuracy'])}",
        f"- Turn state exact accuracy: {_percent(overall['state_exact_accuracy'])}",
        "- Final dialogue state exact accuracy: "
        f"{_percent(overall['final_dialogue_state_exact_accuracy'])}",
        f"- Active-field status accuracy: {_percent(overall['active_field_status_accuracy'])}",
        f"- State field value accuracy: {_percent(overall['state_field_value_accuracy'])}",
        "",
        "## Core structured-state metrics",
        "",
        "Core fields are the seven executable fields plus `usage`; `preference_notes` is excluded.",
        "",
        "- Exact update excluding preference notes: "
        f"{_percent(overall['core_update_exact_accuracy'])}",
        f"- Core turn-state exact accuracy: {_percent(overall['core_state_exact_accuracy'])}",
        "- Core final dialogue state exact accuracy: "
        f"{_percent(overall['core_final_dialogue_state_exact_accuracy'])}",
        "- Core state field-value accuracy: "
        f"{_percent(overall['core_state_field_value_accuracy'])}",
        "",
        "## Executable-state metrics",
        "",
        "Executable metrics include only the seven deterministic hard-constraint fields.",
        "",
        f"- Exact hard-field update: {_percent(overall['executable_update_exact_accuracy'])}",
        "- Executable turn-state exact accuracy: "
        f"{_percent(overall['executable_state_exact_accuracy'])}",
        "- Executable final dialogue state exact accuracy: "
        f"{_percent(overall['executable_final_dialogue_state_exact_accuracy'])}",
        "- Executable active-field-set exact accuracy: "
        f"{_percent(overall['executable_active_field_set_exact_accuracy'])}",
        "- Executable field-value accuracy: "
        f"{_percent(overall['executable_field_value_accuracy'])}",
        "",
        "## Error taxonomy",
        "",
        f"- Strict update mismatches: {overall['strict_update_mismatch_count']}",
        f"- Free-text-only update mismatches: {overall['free_text_only_mismatch_count']}",
        f"- Structural update mismatches: {overall['structural_mismatch_count']}",
        f"- Cascading state-drift turns: {overall['cascading_state_drift_count']}",
        "- Primary categories: "
        f"`{json.dumps(overall['primary_error_category_counts'], sort_keys=True)}`",
        "",
        "Clarification-question matching is intentionally exact. Wording differences receive no "
        "semantic or fuzzy credit.",
        "",
        "## Reliability",
        "",
        f"- API/provider failures: {overall['api_provider_failures']}",
        f"- Extraction failures: {overall['extraction_failures']}",
        f"- Parsing failures: {overall['parsing_failures']}",
        f"- Validation failures: {overall['validation_failures']}",
        f"- Latency mean / median / p95 (ms): {_latency_line(overall['latency_ms'])}",
        f"- Token usage: `{json.dumps(overall['token_usage'], sort_keys=True)}`",
        "- Estimated cost: unavailable; no explicit pricing configuration is present.",
        "",
        "## Metrics by scenario",
        "",
        "| Scenario | Turns | Strict update | Strict state | Core state | Executable state | "
        "Free-text only | Structural |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario, values in metrics["by_scenario"].items():
        lines.append(
            f"| {scenario} | {values['total_turns']} | "
            f"{_percent(values['exact_update_accuracy'])} | "
            f"{_percent(values['state_exact_accuracy'])} | "
            f"{_percent(values['core_state_exact_accuracy'])} | "
            f"{_percent(values['executable_state_exact_accuracy'])} | "
            f"{values['free_text_only_mismatch_count']} | "
            f"{values['structural_mismatch_count']} |"
        )
    lines.extend(
        [
            "",
            "## Response generation",
            "",
            metrics["response_generation_evaluation"]["reason"],
            "Automatic response-faithfulness scoring was not performed.",
            "",
        ]
    )
    return "\n".join(lines)


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def _latency_line(value: Mapping[str, Any] | None) -> str:
    if value is None:
        return "n/a"
    return f"{value['mean']} / {value['median']} / {value['p95']}"
