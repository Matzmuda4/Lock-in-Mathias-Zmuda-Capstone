"""
Internal Pydantic types produced by the parsing pipeline.

Flat ordered list of content items — text, images, and tables —
in document reading order.  Images and tables both have an optional
caption field (from docling's caption_text()) so the reader can display
"Figure 1: …" / "Table 1: …" labels.
"""

from typing import Literal, Optional, Union
from pydantic import BaseModel


class TextContentItem(BaseModel):
    item_type: Literal["text"] = "text"
    index: int
    text: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    label: Optional[str] = None
    bbox: Optional[dict] = None


class ImageContentItem(BaseModel):
    item_type: Literal["image"] = "image"
    index: int
    page: Optional[int] = None
    bbox: Optional[dict] = None
    file_path: Optional[str] = None
    caption: Optional[str] = None   # from PictureItem.caption_text(doc)


class TableContentItem(BaseModel):
    item_type: Literal["table"] = "table"
    index: int
    text: str              # markdown representation (good for LLM context)
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    bbox: Optional[dict] = None
    caption: Optional[str] = None   # from TableItem.caption_text(doc)
    file_path: Optional[str] = None  # optional rendered table image


ContentItem = Union[TextContentItem, ImageContentItem, TableContentItem]


class ParseResult(BaseModel):
    """Complete output of one docling parse run — all items in reading order."""

    items: list[ContentItem]
    raw_doc: Optional[dict] = None
