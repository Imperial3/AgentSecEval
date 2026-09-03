from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import openai

from .baseline_agent import (
    result_path,
    run_baseline_task,
)
from .config import load_baseline_config
from .dataset import (
    ROOT,
    list_task_ids,
    normalize_task_id,
)


def utc_now() -> str:
    """Return current UTC time in ISO-8601 format."""

    return datetime.now(
        timezone.utc
    ).isoformat()


def _git_commit() -> str | None:
    """
    Return the current Git commit SHA when available.
    """

    try:
        result = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )

    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
    ):
        return None

    return result.stdout.strip()


def _git_status_before_run() -> list[str]:
    """
    Record working-tree changes before an experiment run.
    """

    try:
        result = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )

    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
    ):
        return []

    return [
        line
        for line
        in result.stdout.splitlines()
        if line.strip()
    ]


def _write_manifest(
    payload: dict[str, Any],
) -> Path:
    """
    Write the batch-run manifest.
    """

    path = (
        ROOT
        / "results"
        / "baseline"
        / "run-manifest.json"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return path


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Run the AgentSecEval baseline condition."
        )
    )

    selection = (
        parser.add_mutually_exclusive_group(
            required=True
        )
    )

    selection.add_argument(
        "--task",
        help=(
            "Run one benchmark task. "
            "Example: TASK-001"
        ),
    )

    selection.add_argument(
        "--all",
        action="store_true",
        help=(
            "Run all AgentSecEval v0.1 tasks."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace an existing task result."
        ),
    )

    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Development-only model override. "
            "Do not use this option for the "
            "reported baseline experiment."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """
    Run either one benchmark task or the full baseline batch.
    """

    args = parse_args()

    config = load_baseline_config()

    # Single-task mode.
    if args.task:
        task_id = normalize_task_id(
            args.task
        )

        payload = run_baseline_task(
            task_id,
            overwrite=args.overwrite,
            model_override=args.model,
        )

        print(
            f"{task_id}: "
            f"{payload['status']}"
        )

        print(
            "Result: "
            f"{result_path(task_id)}"
        )

        return

    # Full-batch mode.
    task_ids = list_task_ids()

    run_started_at = utc_now()

    git_commit = _git_commit()

    git_status_before_run = (
        _git_status_before_run()
    )

    completed: list[
        dict[str, str]
    ] = []

    failed: list[
        dict[str, str]
    ] = []

    skipped: list[str] = []

    requested_model = (
        args.model
        or config.model
    )

    print(
        "AgentSecEval baseline batch run"
    )

    print(
        f"Model: {requested_model}"
    )

    print(
        f"Tasks: {len(task_ids)}"
    )

    print()

    for task_id in task_ids:

        path = result_path(
            task_id
        )

        if (
            path.exists()
            and not args.overwrite
        ):
            print(
                f"{task_id}: SKIPPED "
                "(result exists)"
            )

            skipped.append(
                task_id
            )

            continue

        try:
            payload = run_baseline_task(
                task_id,
                overwrite=args.overwrite,
                model_override=args.model,
            )

        except Exception as exc:
            print(
                f"{task_id}: ERROR - "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            failed.append(
                {
                    "task_id":
                        task_id,
                    "error_type":
                        type(exc).__name__,
                    "error":
                        str(exc),
                }
            )

            continue

        status = payload[
            "status"
        ]

        print(
            f"{task_id}: {status}"
        )

        completed.append(
            {
                "task_id":
                    task_id,
                "status":
                    status,
            }
        )

    manifest = {
        "benchmark":
            "AgentSecEval",
        "benchmark_version":
            "v0.1",
        "condition":
            "baseline",
        "run_started_at":
            run_started_at,
        "run_finished_at":
            utc_now(),
        "requested_model":
            requested_model,
        "configured_model":
            config.model,
        "development_model_override":
            args.model is not None,
        "reasoning_effort":
            config.reasoning_effort,
        "max_steps":
            config.max_steps,
        "max_output_tokens":
            config.max_output_tokens,
        "tool_choice":
            config.tool_choice,
        "parallel_tool_calls":
            config.parallel_tool_calls,
        "expected_task_count":
            len(task_ids),
        "completed":
            completed,
        "failed":
            failed,
        "skipped":
            skipped,
        "git_commit":
            git_commit,
        "git_status_before_run":
            git_status_before_run,
        "python_version":
            platform.python_version(),
        "python_executable":
            sys.executable,
        "openai_sdk_version":
            openai.__version__,
    }

    manifest_path = _write_manifest(
        manifest
    )

    print()

    print(
        "Batch run complete."
    )

    print(
        f"Completed: {len(completed)}"
    )

    print(
        f"Failed: {len(failed)}"
    )

    print(
        f"Skipped: {len(skipped)}"
    )

    print(
        "Manifest: "
        f"{manifest_path}"
    )


if __name__ == "__main__":
    main()