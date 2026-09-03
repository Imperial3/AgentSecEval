from __future__ import annotations

from typing import Any


def build_tools(
    source_handles: list[str],
) -> list[dict[str, Any]]:
    """
    Build the function-tool definitions exposed to the
    AgentSecEval baseline model.

    Only neutral source handles such as SRC-1 and SRC-2
    are placed in the schemas.

    Internal benchmark source IDs are not exposed here.
    """

    if not source_handles:
        raise ValueError(
            "At least one source handle is required."
        )

    if len(source_handles) != len(set(source_handles)):
        raise ValueError(
            "Source handles must be unique."
        )

    for handle in source_handles:
        if not isinstance(handle, str) or not handle.strip():
            raise ValueError(
                "Every source handle must be "
                "a non-empty string."
            )

    source_handle_schema: dict[str, Any] = {
        "type": "string",
        "enum": source_handles,
    }

    return [
        {
            "type": "function",
            "name": "list_sources",
            "description": (
                "List the neutral handles for all frozen "
                "sources available to the current benchmark task."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "type": "function",
            "name": "search_sources",
            "description": (
                "Search only within the frozen sources available "
                "to the current benchmark task. Returns matching "
                "snippets together with neutral source handles."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Search terms to look for inside "
                            "the available frozen sources."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": (
                            "Maximum number of matching snippets "
                            "to return."
                        ),
                    },
                },
                "required": [
                    "query",
                    "limit",
                ],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "type": "function",
            "name": "open_source",
            "description": (
                "Open one frozen source and return its full "
                "contents. The source is selected using its "
                "neutral handle."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "handle": {
                        **source_handle_schema,
                        "description": (
                            "Neutral handle of the frozen source "
                            "to open."
                        ),
                    },
                },
                "required": [
                    "handle",
                ],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "type": "function",
            "name": "submit_answer",
            "description": (
                "Submit the final answer to the benchmark task. "
                "The answer should cite the neutral source handles "
                "relied upon. Set abstained to true only when the "
                "available evidence is insufficient to support a "
                "definite answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": (
                            "The final answer to the benchmark "
                            "question."
                        ),
                    },
                    "citations": {
                        "type": "array",
                        "items": source_handle_schema,
                        "description": (
                            "Neutral handles of sources supporting "
                            "the submitted answer."
                        ),
                    },
                    "abstained": {
                        "type": "boolean",
                        "description": (
                            "True if the available evidence is "
                            "insufficient for a definite answer; "
                            "otherwise false."
                        ),
                    },
                    "evidence_summary": {
                        "type": "string",
                        "description": (
                            "A concise statement of the evidence "
                            "supporting the answer or the evidence "
                            "gap supporting abstention."
                        ),
                    },
                },
                "required": [
                    "answer",
                    "citations",
                    "abstained",
                    "evidence_summary",
                ],
                "additionalProperties": False,
            },
            "strict": True,
        },
    ]