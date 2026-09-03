from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


# Repository root:
# AgentSecEval/
ROOT = Path(__file__).resolve().parents[1]

# Baseline configuration file:
# AgentSecEval/configs/baseline.json
CONFIG_PATH = ROOT / "configs" / "baseline.json"


@dataclass(frozen=True)
class BaselineConfig:
    condition: str
    model: str
    reasoning_effort: str
    max_steps: int
    max_output_tokens: int
    tool_choice: str
    parallel_tool_calls: bool


def load_baseline_config() -> BaselineConfig:
    """
    Load and validate the committed baseline configuration.

    This function does not load benchmark gold answers.
    """

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Baseline configuration not found: {CONFIG_PATH}"
        )

    raw = json.loads(
        CONFIG_PATH.read_text(encoding="utf-8")
    )

    required_keys = {
        "condition",
        "model",
        "reasoning_effort",
        "max_steps",
        "max_output_tokens",
        "tool_choice",
        "parallel_tool_calls",
    }

    missing_keys = required_keys.difference(raw)

    if missing_keys:
        raise ValueError(
            "Missing baseline configuration keys: "
            f"{sorted(missing_keys)}"
        )

    if raw["condition"] != "baseline":
        raise ValueError(
            "configs/baseline.json must use "
            'condition="baseline".'
        )

    allowed_reasoning_efforts = {
        "none",
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    }

    if raw["reasoning_effort"] not in allowed_reasoning_efforts:
        raise ValueError(
            "Unsupported reasoning_effort: "
            f"{raw['reasoning_effort']}"
        )

    if raw["tool_choice"] not in {
        "required",
        "auto",
    }:
        raise ValueError(
            "tool_choice must be either "
            '"required" or "auto".'
        )

    if not isinstance(raw["max_steps"], int):
        raise TypeError(
            "max_steps must be an integer."
        )

    if raw["max_steps"] < 1:
        raise ValueError(
            "max_steps must be at least 1."
        )

    if not isinstance(
        raw["max_output_tokens"],
        int,
    ):
        raise TypeError(
            "max_output_tokens must be an integer."
        )

    if raw["max_output_tokens"] < 1:
        raise ValueError(
            "max_output_tokens must be at least 1."
        )

    if not isinstance(
        raw["parallel_tool_calls"],
        bool,
    ):
        raise TypeError(
            "parallel_tool_calls must be a boolean."
        )

    return BaselineConfig(
        condition=raw["condition"],
        model=raw["model"],
        reasoning_effort=raw["reasoning_effort"],
        max_steps=raw["max_steps"],
        max_output_tokens=raw["max_output_tokens"],
        tool_choice=raw["tool_choice"],
        parallel_tool_calls=raw["parallel_tool_calls"],
    )


def get_openai_api_key() -> str:
    """
    Load OPENAI_API_KEY from the local .env file.

    The .env file is intentionally excluded from Git.
    """

    env_path = ROOT / ".env"

    load_dotenv(env_path)

    api_key = os.getenv(
        "OPENAI_API_KEY",
        "",
    ).strip()

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. "
            "Add it to the local .env file before "
            "running API-backed benchmark tasks."
        )

    return api_key