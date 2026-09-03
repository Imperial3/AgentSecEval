from __future__ import annotations

import re
from typing import Any

from .dataset import ROOT


STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "what",
    "when",
    "where",
    "which",
    "would",
    "could",
    "should",
    "about",
    "into",
    "under",
    "using",
    "only",
    "does",
    "have",
    "has",
    "had",
    "are",
    "was",
    "were",
    "can",
}


class TaskSourceStore:
    """
    Controlled source environment for one benchmark task.

    The agent sees neutral handles such as SRC-1 and SRC-2.

    Internal benchmark source IDs remain inside the
    environment and are not automatically shown to the model.
    """

    def __init__(
        self,
        task: dict[str, Any],
        registry: dict[str, dict[str, Any]],
    ) -> None:

        source_ids = task.get(
            "source_ids",
            [],
        )

        if not isinstance(
            source_ids,
            list,
        ):
            raise TypeError(
                "task source_ids must be a list."
            )

        if not source_ids:
            raise ValueError(
                f"{task.get('task_id')} "
                "has no source_ids."
            )

        self._registry = registry

        self._handle_to_source_id: dict[
            str,
            str
        ] = {}

        for index, source_id in enumerate(
            source_ids,
            start=1,
        ):

            if source_id not in registry:
                raise KeyError(
                    f"{task.get('task_id')} "
                    "references unknown source_id: "
                    f"{source_id}"
                )

            handle = f"SRC-{index}"

            self._handle_to_source_id[
                handle
            ] = source_id

    @property
    def handles(
        self,
    ) -> list[str]:
        """
        Return neutral source handles available
        to this task.
        """

        return list(
            self._handle_to_source_id.keys()
        )

    def source_map(
        self,
    ) -> dict[str, str]:
        """
        Return the internal handle -> source_id mapping.

        This is intended for experiment logging,
        not for model input.
        """

        return dict(
            self._handle_to_source_id
        )

    def source_id_for_handle(
        self,
        handle: str,
    ) -> str:
        """
        Resolve a neutral handle to its internal source ID.
        """

        if handle not in self._handle_to_source_id:
            raise KeyError(
                f"Unknown source handle: {handle}"
            )

        return self._handle_to_source_id[
            handle
        ]

    def _content_for_handle(
        self,
        handle: str,
    ) -> str:
        """
        Read the frozen local source associated with a handle.
        """

        source_id = (
            self.source_id_for_handle(
                handle
            )
        )

        record = self._registry[
            source_id
        ]

        local_path = record.get(
            "local_path"
        )

        if not isinstance(
            local_path,
            str,
        ) or not local_path.strip():
            raise ValueError(
                f"Source {source_id} "
                "does not have a valid local_path."
            )

        path = ROOT / local_path

        if not path.exists():
            raise FileNotFoundError(
                f"Frozen source file not found: {path}"
            )

        content = path.read_text(
            encoding="utf-8"
        )

        if not content.strip():
            raise ValueError(
                f"Frozen source file is empty: {path}"
            )

        return content

    def list_sources(
        self,
    ) -> dict[str, Any]:
        """
        Return only neutral source handles.

        Internal source IDs and adversarial labels
        are intentionally omitted.
        """

        return {
            "sources": [
                {
                    "handle": handle
                }
                for handle in self.handles
            ],
            "count": len(
                self.handles
            ),
        }

    def open_source(
        self,
        handle: str,
    ) -> dict[str, Any]:
        """
        Return one complete frozen source to the agent.
        """

        return {
            "handle": handle,
            "content": (
                self._content_for_handle(
                    handle
                )
            ),
        }

    def search_sources(
        self,
        query: str,
        limit: int = 5,
    ) -> dict[str, Any]:
        """
        Deterministically search only the frozen
        sources assigned to this task.

        This performs simple keyword matching.
        It does not use the internet or an embedding API.
        """

        query = query.strip()

        if not query:
            return {
                "query": query,
                "results": [],
            }

        if not isinstance(
            limit,
            int,
        ):
            raise TypeError(
                "Search limit must be an integer."
            )

        limit = max(
            1,
            min(
                limit,
                10,
            ),
        )

        tokens = {
            token
            for token in re.findall(
                r"[a-z0-9$%.\-]+",
                query.lower(),
            )
            if (
                len(token) >= 3
                and token not in STOPWORDS
            )
        }

        matches: list[
            dict[str, Any]
        ] = []

        for handle in self.handles:

            content = (
                self._content_for_handle(
                    handle
                )
            )

            paragraphs = [
                paragraph.strip()
                for paragraph
                in re.split(
                    r"\n\s*\n",
                    content,
                )
                if paragraph.strip()
            ]

            for (
                paragraph_index,
                paragraph,
            ) in enumerate(
                paragraphs
            ):

                lowercase_paragraph = (
                    paragraph.lower()
                )

                score = sum(
                    1
                    for token in tokens
                    if token
                    in lowercase_paragraph
                )

                if (
                    query.lower()
                    in lowercase_paragraph
                ):
                    score += 3

                if score <= 0:
                    continue

                matches.append(
                    {
                        "handle": handle,
                        "paragraph_index":
                            paragraph_index,
                        "score": score,
                        "snippet":
                            paragraph[:1500],
                    }
                )

        matches.sort(
            key=lambda item: (
                -item["score"],
                item["handle"],
                item["paragraph_index"],
            )
        )

        return {
            "query": query,
            "results":
                matches[:limit],
        }