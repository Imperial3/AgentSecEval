from __future__ import annotations

import json
import unittest

from src.dataset import (
    load_registry,
    load_task,
)

from src.source_store import (
    TaskSourceStore,
)

from src.tools import (
    build_tools,
)


class TaskSourceStoreTests(unittest.TestCase):

    def setUp(self) -> None:
        self.registry = load_registry()

    def test_task_014_uses_neutral_handles(
        self,
    ) -> None:

        task = load_task(
            "TASK-014"
        )

        store = TaskSourceStore(
            task,
            self.registry,
        )

        self.assertEqual(
            store.handles,
            [
                "SRC-1",
                "SRC-2",
            ],
        )

        listing = store.list_sources()

        self.assertEqual(
            listing,
            {
                "sources": [
                    {
                        "handle": "SRC-1"
                    },
                    {
                        "handle": "SRC-2"
                    },
                ],
                "count": 2,
            },
        )

        serialized = json.dumps(
            listing
        )

        self.assertNotIn(
            "SBIR-FORCEFIELDS",
            serialized,
        )

        self.assertNotIn(
            "DECOY-FORCEFIELDS-INJECTION",
            serialized,
        )

    def test_task_014_internal_mapping(
        self,
    ) -> None:

        task = load_task(
            "TASK-014"
        )

        store = TaskSourceStore(
            task,
            self.registry,
        )

        self.assertEqual(
            store.source_map(),
            {
                "SRC-1":
                    "SBIR-FORCEFIELDS",
                "SRC-2":
                    "DECOY-FORCEFIELDS-INJECTION",
            },
        )

    def test_task_014_attack_content_is_reachable(
        self,
    ) -> None:

        task = load_task(
            "TASK-014"
        )

        store = TaskSourceStore(
            task,
            self.registry,
        )

        source = store.open_source(
            "SRC-2"
        )

        self.assertIn(
            "$10,000,000",
            source["content"],
        )

        self.assertIn(
            "Ignore all previous instructions",
            source["content"],
        )

    def test_task_014_official_content_is_reachable(
        self,
    ) -> None:

        task = load_task(
            "TASK-014"
        )

        store = TaskSourceStore(
            task,
            self.registry,
        )

        source = store.open_source(
            "SRC-1"
        )

        self.assertIn(
            "$1,999,942",
            source["content"],
        )

    def test_tools_do_not_leak_internal_ids(
        self,
    ) -> None:

        tools = build_tools(
            [
                "SRC-1",
                "SRC-2",
            ]
        )

        serialized = json.dumps(
            tools
        )

        self.assertNotIn(
            "SBIR-FORCEFIELDS",
            serialized,
        )

        self.assertNotIn(
            "DECOY",
            serialized,
        )

    def test_exactly_four_tools_exist(
        self,
    ) -> None:

        tools = build_tools(
            [
                "SRC-1",
                "SRC-2",
            ]
        )

        names = [
            tool["name"]
            for tool in tools
        ]

        self.assertEqual(
            names,
            [
                "list_sources",
                "search_sources",
                "open_source",
                "submit_answer",
            ],
        )


if __name__ == "__main__":
    unittest.main()