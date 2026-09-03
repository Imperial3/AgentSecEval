from __future__ import annotations

import unittest

from src.baseline_agent import (
    BASELINE_INSTRUCTIONS,
    result_path,
)
from src.config import (
    load_baseline_config,
)
from src.dataset import ROOT


class BaselineAgentTests(unittest.TestCase):

    def test_baseline_config_is_frozen(
        self,
    ) -> None:
        config = load_baseline_config()

        self.assertEqual(
            config.condition,
            "baseline",
        )
        self.assertEqual(
            config.model,
            "gpt-5.6-terra",
        )
        self.assertEqual(
            config.reasoning_effort,
            "low",
        )
        self.assertEqual(
            config.max_steps,
            10,
        )
        self.assertEqual(
            config.max_output_tokens,
            2000,
        )
        self.assertEqual(
            config.tool_choice,
            "required",
        )
        self.assertFalse(
            config.parallel_tool_calls
        )

    def test_baseline_prompt_is_not_security_primed(
        self,
    ) -> None:
        prompt = (
            BASELINE_INSTRUCTIONS
            .lower()
        )

        forbidden = [
            "adversarial",
            "prompt injection",
            "malicious source",
            "authoritative source",
        ]

        for term in forbidden:
            self.assertNotIn(
                term,
                prompt,
            )

    def test_baseline_agent_has_no_gold_path(
        self,
    ) -> None:
        path = (
            ROOT
            / "src"
            / "baseline_agent.py"
        )

        source = (
            path
            .read_text(
                encoding="utf-8"
            )
            .lower()
        )

        forbidden = [
            "answers.json",
            "load_gold",
            "import gold",
            "from .gold",
        ]

        for term in forbidden:
            self.assertNotIn(
                term,
                source,
            )

    def test_result_path_is_correct(
        self,
    ) -> None:
        expected = (
            ROOT
            / "results"
            / "baseline"
            / "task-001.json"
        )

        self.assertEqual(
            result_path(
                "TASK-001"
            ),
            expected,
        )


if __name__ == "__main__":
    unittest.main()