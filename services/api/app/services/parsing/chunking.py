"""
Chunking strategy for parsed content items.

Rules:
  - Text items: short consecutive blocks on the same page are merged
    (avoids fragmenting one visual paragraph into many tiny chunks).
    Heading-labelled items are never merged with their successor.
  - Image and table items: never merged, never reordered —
    they stay exactly where docling placed them in reading order.
"""

from __future__ import annotations
from typing import Any

from app.services.parsing.models import (
    ContentItem,
    ImageContentItem,
    TableContentItem,
    TextContentItem,
)

_MERGE_THRESHOLD_CHARS = 80
_NO_MERGE_LABELS = {"section_header", "title", "page_header", "page_footer"}


def build_text_chunks(raw_items: list[dict[str, Any]]) -> list[ContentItem]:
    """
    Convert the flat list of raw dicts (from parser.py) into typed and
    lightly-merged ContentItem objects.

    Each raw dict must contain:
        item_type  "text" | "image" | "table"
        -- text --
        text, page, bbox, label
        -- image --
        page, bbox, file_path, caption
        -- table --
        text (markdown), page, bbox, caption, file_path
    """
    if not raw_items:
        return []

    result: list[ContentItem] = []
    buf: dict[str, Any] | None = None

    def flush_buf() -> None:
        nonlocal buf
        if buf is None:
            return
        text = buf["text"].strip()
        if text:
            result.append(
                TextContentItem(
                    index=0,
                    text=text,
                    page_start=buf.get("page"),
                    page_end=buf.get("page"),
                    label=buf.get("label"),
                    bbox=buf.get("bbox"),
                )
            )
        buf = None

    for raw in raw_items:
        t = raw["item_type"]

        if t == "image":
            flush_buf()
            result.append(
                ImageContentItem(
                    index=0,
                    page=raw.get("page"),
                    bbox=raw.get("bbox"),
                    file_path=raw.get("file_path"),
                    caption=raw.get("caption") or None,
                )
            )
            continue

        if t == "table":
            flush_buf()
            md = (raw.get("text") or "").strip()
            if md:
                result.append(
                    TableContentItem(
                        index=0,
                        text=md,
                        page_start=raw.get("page"),
                        page_end=raw.get("page"),
                        bbox=raw.get("bbox"),
                        caption=raw.get("caption") or None,
                        file_path=raw.get("file_path"),
                    )
                )
            continue

        # Text item
        text = (raw.get("text") or "").strip()
        if not text:
            continue

        label = raw.get("label") or ""

        if buf is None:
            buf = dict(raw)
            buf["text"] = text
        else:
            same_page = raw.get("page") == buf.get("page")
            buf_short = len(buf["text"]) < _MERGE_THRESHOLD_CHARS
            buf_label = buf.get("label") or ""
            can_merge = (
                same_page
                and buf_short
                and buf_label not in _NO_MERGE_LABELS
                and label not in _NO_MERGE_LABELS
            )

            if can_merge:
                buf["text"] = buf["text"].rstrip() + " " + text.lstrip()
                if buf.get("bbox") and raw.get("bbox"):
                    b, n = buf["bbox"], raw["bbox"]
                    buf["bbox"] = {
                        "x0": min(b["x0"], n["x0"]),
                        "y0": min(b["y0"], n["y0"]),
                        "x1": max(b["x1"], n["x1"]),
                        "y1": max(b["y1"], n["y1"]),
                    }
            else:
                flush_buf()
                buf = dict(raw)
                buf["text"] = text

    flush_buf()

    # Re-index sequentially
    for i, item in enumerate(result):
        item.index = i

    return result
