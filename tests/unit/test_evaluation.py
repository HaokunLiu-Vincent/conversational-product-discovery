import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from product_discovery.evaluation.runner import (
    ExtractionResult,
    evaluate_dialogues,
    load_development_dialogues,
    write_evaluation_artifacts,
)
from product_discovery.llm.adapter import LLMAdapterError
from product_discovery.schemas import PreferenceState, PreferenceUpdate


def _state(**updates: object) -> dict[str, object]:
    return PreferenceState(**updates).model_dump(mode="json")


def _turn(
    index: int,
    *,
    before: dict[str, object],
    after: dict[str, object],
    updates: list[dict[str, object]],
    scenario: str = "test_scenario",
    clarification_needed: bool = False,
    clarification_question: str | None = None,
) -> dict[str, Any]:
    return {
        "turn_index": index,
        "user_text": f"turn {index}",
        "scenario_type": scenario,
        "oracle_update": updates,
        "oracle_clarification_needed": clarification_needed,
        "oracle_clarification_question": clarification_question,
        "oracle_state_before": before,
        "oracle_state_after": after,
        "oracle_candidate_count": None,
        "annotation_notes": None,
    }


def _dialogue(
    turns: list[dict[str, Any]],
    dialogue_id: str = "dialogue-1",
) -> dict[str, Any]:
    return {
        "dialogue_id": dialogue_id,
        "recipe": "unit_test",
        "source": "synthetic_test",
        "split": "development",
        "turns": turns,
    }


class FakeExtractor:
    def __init__(self, results: Sequence[ExtractionResult | Exception]) -> None:
        self._results = iter(results)

    def extract(self, user_text: str, state: PreferenceState) -> ExtractionResult:
        result = next(self._results)
        if isinstance(result, Exception):
            raise result
        return result


def _metadata() -> dict[str, str]:
    return {
        "evaluation_timestamp_utc": "2026-01-01T00:00:00+00:00",
        "git_commit_sha": "abc123",
        "dataset_identifier": "synthetic-development",
        "openai_model": "mock-model",
    }


def test_evaluate_dialogues_scores_canonical_updates_and_state(tmp_path: Path) -> None:
    empty = _state()
    budget = _state(budget_max=1000)
    turns = [
        _turn(
            1,
            before=empty,
            after=budget,
            updates=[{"field": "budget_max", "operation": "set", "value": 1000}],
        ),
        _turn(
            2,
            before=budget,
            after=empty,
            updates=[{"field": "budget_max", "operation": "remove", "value": None}],
        ),
        _turn(3, before=empty, after=empty, updates=[]),
    ]
    results = [
        ExtractionResult(
            update=PreferenceUpdate.model_validate(
                {"updates": [{"field": "budget_max", "operation": "set", "value": "1000"}]}
            ),
            raw_model_output='{"updates": [{"field": "budget_max"}]}',
            usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            provider_response_id="response-1",
        ),
        ExtractionResult(
            update=PreferenceUpdate.model_validate(
                {"updates": [{"field": "budget_max", "operation": "remove", "value": None}]}
            ),
            usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        ),
        ExtractionResult(
            update=PreferenceUpdate(),
            usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        ),
    ]

    report = evaluate_dialogues([_dialogue(turns)], FakeExtractor(results), metadata=_metadata())

    overall = report["metrics"]["overall"]
    assert overall["total_dialogues"] == 1
    assert overall["total_turns"] == 3
    assert overall["exact_update_accuracy"] == 1.0
    assert overall["exact_structured_output_accuracy"] == 1.0
    assert overall["state_exact_accuracy"] == 1.0
    assert overall["core_state_exact_accuracy"] == 1.0
    assert overall["executable_state_exact_accuracy"] == 1.0
    assert overall["final_dialogue_state_exact_accuracy"] == 1.0
    assert overall["core_final_dialogue_state_exact_accuracy"] == 1.0
    assert overall["executable_final_dialogue_state_exact_accuracy"] == 1.0
    assert overall["token_usage"] == {
        "input_tokens": 30,
        "output_tokens": 15,
        "total_tokens": 45,
    }
    assert report["metrics"]["by_turn_index"]["2"]["operation_accuracy"] == 1.0
    assert report["errors"] == []

    paths = write_evaluation_artifacts(report, tmp_path)
    assert {path.name for path in paths} == {
        "dev_predictions.jsonl",
        "dev_metrics.json",
        "dev_metrics.csv",
        "dev_errors.jsonl",
        "dev_summary.md",
        "dev_error_taxonomy.csv",
        "dev_cascading_drift.csv",
    }
    assert len((tmp_path / "dev_predictions.jsonl").read_text().splitlines()) == 3
    assert (tmp_path / "dev_errors.jsonl").read_text() == ""
    assert len((tmp_path / "dev_error_taxonomy.csv").read_text().splitlines()) == 1
    assert len((tmp_path / "dev_cascading_drift.csv").read_text().splitlines()) == 1
    assert (
        "Automatic response-faithfulness scoring was not performed."
        in (tmp_path / "dev_summary.md").read_text()
    )


def test_evaluate_dialogues_preserves_failure_and_continues() -> None:
    empty = _state()
    dialogue = _dialogue([_turn(1, before=empty, after=empty, updates=[])])
    report = evaluate_dialogues(
        [dialogue],
        FakeExtractor([LLMAdapterError("OpenAI preference extraction failed")]),
        metadata=_metadata(),
    )

    prediction = report["predictions"][0]
    overall = report["metrics"]["overall"]
    assert prediction["failure_type"] == "extraction_failure"
    assert prediction["error_message"] == "OpenAI preference extraction failed"
    assert prediction["correctness"]["state_exact"] is True
    assert prediction["correctness"]["exact_structured_output_correct"] is False
    assert overall["successful_inference_calls"] == 0
    assert overall["extraction_failures"] == 1
    assert len(report["errors"]) == 1


def test_load_development_dialogues_rejects_heldout_path_before_reading(
    tmp_path: Path,
) -> None:
    forbidden_path = tmp_path / "heldout_test_dialogues_v1.jsonl"

    with pytest.raises(ValueError, match="held-out evaluation inputs are not supported"):
        load_development_dialogues(forbidden_path)


def test_load_development_dialogues_validates_synthetic_development_file(
    tmp_path: Path,
) -> None:
    empty = _state()
    source = tmp_path / "synthetic_development.jsonl"
    source.write_text(
        json.dumps(_dialogue([_turn(1, before=empty, after=empty, updates=[])])) + "\n",
        encoding="utf-8",
    )

    dialogues = load_development_dialogues(source)

    assert len(dialogues) == 1
    assert dialogues[0]["split"] == "development"


def test_evaluation_distinguishes_strict_core_executable_and_error_taxonomy(
    tmp_path: Path,
) -> None:
    empty = _state()
    expected_note = _state(preference_notes="Prefers a quiet laptop.")
    budget = _state(budget_max=1000)
    lenovo = _state(brand="Lenovo")
    dialogues = [
        _dialogue(
            [
                _turn(
                    1,
                    before=empty,
                    after=expected_note,
                    updates=[
                        {
                            "field": "preference_notes",
                            "operation": "set",
                            "value": "Prefers a quiet laptop.",
                        }
                    ],
                    scenario="free_text",
                ),
                _turn(
                    2,
                    before=expected_note,
                    after=expected_note,
                    updates=[],
                    scenario="cascade",
                ),
            ],
            "free-text-dialogue",
        ),
        _dialogue(
            [
                _turn(
                    1,
                    before=empty,
                    after=expected_note,
                    updates=[
                        {
                            "field": "preference_notes",
                            "operation": "set",
                            "value": "Prefers a quiet laptop.",
                        }
                    ],
                    scenario="extra_usage",
                )
            ],
            "extra-usage-dialogue",
        ),
        _dialogue(
            [
                _turn(
                    1,
                    before=empty,
                    after=budget,
                    updates=[{"field": "budget_max", "operation": "set", "value": 1000}],
                    scenario="hard_value",
                )
            ],
            "hard-value-dialogue",
        ),
        _dialogue(
            [
                _turn(
                    1,
                    before=empty,
                    after=lenovo,
                    updates=[{"field": "brand", "operation": "set", "value": "Lenovo"}],
                    scenario="brand_normalization",
                )
            ],
            "brand-dialogue",
        ),
        _dialogue(
            [
                _turn(
                    1,
                    before=empty,
                    after=empty,
                    updates=[],
                    scenario="clarification",
                    clarification_needed=True,
                    clarification_question="What maximum weight would you like?",
                )
            ],
            "clarification-dialogue",
        ),
    ]
    results = [
        ExtractionResult(
            PreferenceUpdate.model_validate(
                {
                    "updates": [
                        {
                            "field": "preference_notes",
                            "operation": "set",
                            "value": "Wants a quiet laptop.",
                        }
                    ]
                }
            )
        ),
        ExtractionResult(PreferenceUpdate()),
        ExtractionResult(
            PreferenceUpdate.model_validate(
                {
                    "updates": [
                        {
                            "field": "preference_notes",
                            "operation": "set",
                            "value": "Prefers a quiet laptop.",
                        },
                        {"field": "usage", "operation": "set", "value": "all_round"},
                    ]
                }
            )
        ),
        ExtractionResult(
            PreferenceUpdate.model_validate(
                {"updates": [{"field": "budget_max", "operation": "set", "value": 1200}]}
            )
        ),
        ExtractionResult(
            PreferenceUpdate.model_validate(
                {
                    "updates": [
                        {
                            "field": "preference_notes",
                            "operation": "set",
                            "value": "Interested in an IdeaPad.",
                        }
                    ]
                }
            )
        ),
        ExtractionResult(
            PreferenceUpdate(
                clarification_needed=True,
                clarification_question="How light should it be?",
            )
        ),
    ]

    report = evaluate_dialogues(dialogues, FakeExtractor(results), metadata=_metadata())
    records = report["predictions"]

    free_text = records[0]
    assert free_text["correctness"]["exact_update_correct"] is False
    assert free_text["correctness"]["core_update_exact"] is True
    assert free_text["correctness"]["executable_update_exact"] is True
    assert free_text["correctness"]["state_exact"] is False
    assert free_text["correctness"]["core_state_exact"] is True
    assert free_text["correctness"]["executable_state_exact"] is True
    assert free_text["correctness"]["executable_active_field_set_exact"] is True
    assert free_text["primary_error_category"] == "free_text_only"

    cascade = records[1]
    assert cascade["correctness"]["exact_update_correct"] is True
    assert cascade["correctness"]["state_exact"] is False
    assert cascade["primary_error_category"] == "cascading_state_drift"
    assert cascade["first_state_divergence_turn"] == 1
    assert cascade["drifting_fields"] == ["preference_notes"]
    assert cascade["drift_free_text_only"] is True

    extra_usage = records[2]
    assert extra_usage["correctness"]["core_update_exact"] is False
    assert extra_usage["correctness"]["executable_update_exact"] is True
    assert extra_usage["correctness"]["executable_state_exact"] is True
    assert extra_usage["primary_error_category"] == "extra_structured_field"
    assert "wrong_operation" in extra_usage["secondary_error_categories"]

    hard_value = records[3]
    assert hard_value["correctness"]["executable_update_exact"] is False
    assert hard_value["correctness"]["executable_state_exact"] is False
    assert hard_value["primary_error_category"] == "wrong_structured_value"

    brand = records[4]
    assert brand["primary_error_category"] == "brand_normalization"
    assert "missing_structured_field" in brand["secondary_error_categories"]
    assert brand["correctness"]["executable_update_exact"] is False

    clarification = records[5]
    assert clarification["correctness"]["clarification_needed_correct"] is True
    assert clarification["correctness"]["clarification_question_exact"] is False
    assert clarification["primary_error_category"] == "clarification_text_only"

    overall = report["metrics"]["overall"]
    assert overall["free_text_only_mismatch_count"] == 1
    assert overall["structural_mismatch_count"] == 3
    assert overall["cascading_state_drift_count"] == 1

    write_evaluation_artifacts(report, tmp_path)
    taxonomy_lines = (tmp_path / "dev_error_taxonomy.csv").read_text().splitlines()
    cascading_lines = (tmp_path / "dev_cascading_drift.csv").read_text().splitlines()
    assert len(taxonomy_lines) == 7
    assert "primary_error_category" in taxonomy_lines[0]
    assert len(cascading_lines) == 2
    assert "first_state_divergence_turn" in cascading_lines[0]
