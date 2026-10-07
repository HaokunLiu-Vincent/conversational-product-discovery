"""Command-line entry point for the protected development-only evaluation workflow."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from product_discovery.configuration import Settings
from product_discovery.evaluation.runner import (
    OpenAIPreferenceExtractor,
    evaluate_dialogues,
    file_sha256,
    load_development_dialogues,
    write_evaluation_artifacts,
)

DEFAULT_INPUT = Path("data/evaluation/development_dialogues_v1.jsonl")
DEFAULT_OUTPUT_DIR = Path("evaluation/results/dev")


def main() -> int:
    args = _parse_args()
    dialogues = load_development_dialogues(args.input)
    turn_count = sum(len(dialogue["turns"]) for dialogue in dialogues)
    if len(dialogues) != args.expected_dialogues or turn_count != args.expected_turns:
        raise ValueError(
            "development dataset size mismatch: "
            f"expected {args.expected_dialogues} dialogues/{args.expected_turns} turns, "
            f"found {len(dialogues)} dialogues/{turn_count} turns"
        )

    print(f"Development input: {args.input}")
    print(f"Verified development split: {len(dialogues)} dialogues, {turn_count} turns")
    settings = Settings()
    if settings.openai_api_key is None:
        raise RuntimeError("PRODUCT_DISCOVERY_OPENAI_API_KEY is not configured")

    command = "python -m product_discovery.evaluation.run_development " + shlex.join(
        ["--output-dir", str(args.output_dir)]
    )
    metadata: dict[str, Any] = {
        "evaluation_timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit_sha": _git_sha(),
        "dataset_identifier": args.input.stem,
        "dataset_path": str(args.input),
        "dataset_sha256": file_sha256(args.input),
        "evaluated_dialogues": len(dialogues),
        "evaluated_turns": turn_count,
        "openai_model": settings.openai_model,
        "catalogue_used": False,
        "catalogue_version": None,
        "catalogue_sha256": None,
        "python_version": sys.version.split()[0],
        "package_version": version("conversational-product-discovery"),
        "evaluation_command": command,
    }
    extractor = OpenAIPreferenceExtractor(
        api_key=settings.openai_api_key.get_secret_value(),
        model=settings.openai_model,
    )
    report = evaluate_dialogues(
        dialogues,
        extractor,
        metadata=metadata,
        on_turn=_progress,
    )
    paths = write_evaluation_artifacts(report, args.output_dir)
    overall = report["metrics"]["overall"]
    print(
        "Evaluation complete: "
        f"exact_update={overall['exact_update_accuracy']}, "
        f"state_exact={overall['state_exact_accuracy']}, "
        f"failures={overall['failed_inference_calls']}"
    )
    for path in paths:
        print(f"Wrote {path}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-dialogues", type=int, default=60)
    parser.add_argument("--expected-turns", type=int, default=155)
    return parser.parse_args()


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _progress(completed: int, total: int, record: dict[str, Any]) -> None:
    status = record["failure_type"] or "ok"
    print(
        f"[{completed}/{total}] {record['dialogue_id']} turn {record['turn_index']}: {status}",
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
