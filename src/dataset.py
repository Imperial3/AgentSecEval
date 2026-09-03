from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


# Repository root:
# AgentSecEval/
ROOT = Path(__file__).resolve().parents[1]

# Agent-visible task definitions live here.
TASKS_DIR = ROOT / "tasks"

# Source metadata and local source paths live here.
REGISTRY_PATH = (
    ROOT
    / "data"
    / "source-registry.json"
)

# Valid benchmark IDs:
# TASK-001
# TASK-002
# ...
TASK_ID_PATTERN = re.compile(
    r"^TASK-(\d{3})$"
)


def read_json(path: Path) -> Any:
    """
    Read a UTF-8 JSON file.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"JSON file not found: {path}"
        )

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {path}: {exc}"
        ) from exc


def normalize_task_id(task_id: str) -> str:
    """
    Normalize and validate a benchmark task ID.

    Example:
        task-001 -> TASK-001
    """

    normalized = (
        task_id
        .strip()
        .upper()
    )

    if not TASK_ID_PATTERN.fullmatch(
        normalized
    ):
        raise ValueError(
            "Task ID must have the form TASK-001."
        )

    return normalized


def task_path(task_id: str) -> Path:
    """
    Convert TASK-001 into tasks/task-001.json.
    """

    normalized = normalize_task_id(
        task_id
    )

    match = TASK_ID_PATTERN.fullmatch(
        normalized
    )

    if match is None:
        raise ValueError(
            f"Invalid task ID: {task_id}"
        )

    number = match.group(1)

    return (
        TASKS_DIR
        / f"task-{number}.json"
    )


def load_task(
    task_id: str,
) -> dict[str, Any]:
    """
    Load one benchmark task.

    Important:
    This function reads only the task definition.
    It does not load gold/answers.json.
    """

    normalized = normalize_task_id(
        task_id
    )

    path = task_path(
        normalized
    )

    task = read_json(
        path
    )

    if not isinstance(task, dict):
        raise TypeError(
            f"Task file must contain a JSON object: {path}"
        )

    file_task_id = task.get(
        "task_id"
    )

    if file_task_id != normalized:
        raise ValueError(
            f"Task ID mismatch in {path}. "
            f"Expected {normalized}, "
            f"found {file_task_id!r}."
        )

    required_fields = {
        "task_id",
        "category",
        "title",
        "question",
        "snapshot_date",
        "must_abstain",
        "adversarial",
        "source_ids",
    }

    missing_fields = (
        required_fields
        .difference(task)
    )

    if missing_fields:
        raise ValueError(
            f"{normalized} is missing fields: "
            f"{sorted(missing_fields)}"
        )

    if not isinstance(
        task["source_ids"],
        list,
    ):
        raise TypeError(
            f"{normalized} source_ids must be a list."
        )

    if not task["source_ids"]:
        raise ValueError(
            f"{normalized} must contain at least one source_id."
        )

    return task


def list_task_ids() -> list[str]:
    """
    Return all benchmark task IDs in filename order.
    """

    if not TASKS_DIR.exists():
        raise FileNotFoundError(
            f"Tasks directory not found: {TASKS_DIR}"
        )

    task_ids: list[str] = []

    for path in sorted(
        TASKS_DIR.glob(
            "task-*.json"
        )
    ):
        task = read_json(
            path
        )

        if not isinstance(task, dict):
            raise TypeError(
                f"Task file must contain an object: {path}"
            )

        task_id = normalize_task_id(
            task["task_id"]
        )

        task_ids.append(
            task_id
        )

    if len(task_ids) != len(
        set(task_ids)
    ):
        raise ValueError(
            "Duplicate task IDs detected."
        )

    return task_ids


def load_registry(
) -> dict[str, dict[str, Any]]:
    """
    Load source-registry.json and index it by source_id.

    Internal fields such as authority/adversarial status
    are available to the environment implementation,
    but they are not automatically exposed to the agent.
    """

    raw = read_json(
        REGISTRY_PATH
    )

    if not isinstance(raw, dict):
        raise TypeError(
            "source-registry.json must contain "
            "a JSON object."
        )

    sources = raw.get(
        "sources"
    )

    if not isinstance(
        sources,
        list,
    ):
        raise TypeError(
            "source-registry.json must contain "
            'a "sources" list.'
        )

    registry: dict[
        str,
        dict[str, Any]
    ] = {}

    for source in sources:

        if not isinstance(
            source,
            dict,
        ):
            raise TypeError(
                "Every registry source must "
                "be a JSON object."
            )

        source_id = source.get(
            "source_id"
        )

        if not isinstance(
            source_id,
            str,
        ) or not source_id.strip():
            raise ValueError(
                "Every source registry entry "
                "must have a non-empty source_id."
            )

        if source_id in registry:
            raise ValueError(
                "Duplicate source_id in registry: "
                f"{source_id}"
            )

        local_path = source.get(
            "local_path"
        )

        if not isinstance(
            local_path,
            str,
        ) or not local_path.strip():
            raise ValueError(
                f"{source_id} has no valid local_path."
            )

        registry[source_id] = source

    return registry