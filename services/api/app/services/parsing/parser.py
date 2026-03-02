"""
Docling parsing service.

Iterates ALL document items (text, images, tables) in reading order so the
reader displays figures and tables inline at their correct document position.

Each figure and table has:
  - A caption extracted via item.caption_text(doc)  (e.g. "Figure 1: Title")
  - An image saved to parsed_cache/{doc_id}/  (PNG, for display)
  - Tables also carry their markdown representation (for LLM context)

Performance
-----------
The DocumentConverter is a module-level singleton — ML models are loaded once
per server process.  The first parse is slow; all subsequent parses run
inference only and are significantly faster.

Note: changing do_table_structure requires a server restart to recreate the
singleton with the new configuration.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models import DocumentAsset, DocumentChunk, DocumentParseJob
from app.db.session import async_session_factory
from app.services.parsing.chunking import build_text_chunks
from app.services.parsing.models import (
    ContentItem,
    ImageContentItem,
    TableContentItem,
    ParseResult,
)

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="docling")

# ─── Converter singleton ───────────────────────────────────────────────────────

_converter: Any = None
_converter_lock = threading.Lock()


def _get_converter() -> Any:
    """Return (and lazily initialise) the module-level DocumentConverter."""
    global _converter
    if _converter is None:
        with _converter_lock:
            if _converter is None:
                from docling.document_converter import DocumentConverter, PdfFormatOption
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import PdfPipelineOptions

                pipeline_options = PdfPipelineOptions()
                pipeline_options.do_ocr = False
                pipeline_options.do_table_structure = True
                # Required for item.get_image() to return a PIL image instead of None
                pipeline_options.generate_picture_images = True
                pipeline_options.generate_table_images = True
                pipeline_options.images_scale = 2.0  # 2× for readable resolution

                _converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(
                            pipeline_options=pipeline_options
                        )
                    }
                )
                logger.info("DocumentConverter initialised (layout + table + picture image models loaded)")
    return _converter


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _extract_bbox(prov_item: Any) -> dict | None:
    bbox_obj = getattr(prov_item, "bbox", None)
    if bbox_obj is None:
        return None
    try:
        return {
            "x0": float(bbox_obj.l),
            "y0": float(bbox_obj.t),
            "x1": float(bbox_obj.r),
            "y1": float(bbox_obj.b),
        }
    except Exception:
        return None


def _save_image(item: Any, doc: Any, dest: Path) -> str | None:
    """Save a PictureItem or TableItem as a PNG and return the file path."""
    try:
        pil_img = item.get_image(doc, prov_index=0)
        if pil_img is not None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            pil_img.save(str(dest))
            logger.debug("Saved image asset: %s", dest)
            return str(dest)
        else:
            logger.warning(
                "get_image() returned None for %s -> %s. "
                "Ensure generate_picture_images=True is set in PdfPipelineOptions.",
                type(item).__name__, dest.name,
            )
    except Exception as exc:
        logger.warning("Could not save image to %s: %s", dest, exc)
    return None


def _get_caption(item: Any, doc: Any) -> str | None:
    """Return caption text, or None if empty / not available."""
    try:
        cap = item.caption_text(doc)
        return cap.strip() or None
    except Exception:
        return None


def _prov_page(item: Any) -> int | None:
    prov = item.prov[0] if getattr(item, "prov", None) else None
    return getattr(prov, "page_no", None) if prov else None


def _label_value(item: Any) -> str | None:
    label_raw = getattr(item, "label", None)
    if label_raw is None:
        return None
    return label_raw.value if hasattr(label_raw, "value") else str(label_raw)


import re as _re

# Patterns produced by broken font encodings in some PDFs:
#   /uniBF0  /uniXXXX  /uniXXXXXX  (PostScript glyph names)
#   \ufffd   (Unicode replacement character)
_GLYPH_NOISE_RE = _re.compile(r"/uni[0-9A-Fa-f]{4,6}|\ufffd")


def _clean_text(text: str) -> str:
    """Strip unresolved PDF glyph names and unicode replacement characters."""
    cleaned = _GLYPH_NOISE_RE.sub("", text)
    # Collapse any double-spaces that result from removal
    cleaned = _re.sub(r"  +", " ", cleaned)
    return cleaned.strip()


# ─── Sync parse (runs in thread-pool) ─────────────────────────────────────────

def _sync_parse(file_path: Path, cache_dir: Path) -> ParseResult:
    """
    Full document parse.  Returns an ordered list of ContentItems —
    text, images, and tables — in document reading order.
    """
    try:
        from docling.datamodel.document import PictureItem, TableItem
    except ImportError:
        PictureItem = None  # type: ignore[assignment,misc]
        TableItem = None    # type: ignore[assignment,misc]

    converter = _get_converter()
    result = converter.convert(source=str(file_path))
    doc = result.document

    cache_dir.mkdir(parents=True, exist_ok=True)

    img_counter = 0   # counts saved image/table files
    raw_items: list[dict[str, Any]] = []

    for item, _level in doc.iterate_items():
        # ── Table ─────────────────────────────────────────────────────────
        if TableItem is not None and isinstance(item, TableItem):
            prov = item.prov[0] if getattr(item, "prov", None) else None
            page = getattr(prov, "page_no", None) if prov else None
            bbox = _extract_bbox(prov) if prov else None
            caption = _get_caption(item, doc)

            # Markdown for LLM reading
            table_md = ""
            try:
                table_md = item.export_to_markdown(doc)
            except Exception:
                pass

            # Image of table for visual display
            img_path = _save_image(item, doc, cache_dir / f"table_{img_counter}.png")
            if img_path:
                img_counter += 1

            raw_items.append({
                "item_type": "table",
                "text": table_md,
                "page": page,
                "bbox": bbox,
                "caption": caption,
                "file_path": img_path,
            })
            continue

        # ── Picture ───────────────────────────────────────────────────────
        if PictureItem is not None and isinstance(item, PictureItem):
            prov = item.prov[0] if getattr(item, "prov", None) else None
            page = getattr(prov, "page_no", None) if prov else None
            bbox = _extract_bbox(prov) if prov else None
            caption = _get_caption(item, doc)

            img_path = _save_image(item, doc, cache_dir / f"image_{img_counter}.png")
            if img_path:
                img_counter += 1

            raw_items.append({
                "item_type": "image",
                "page": page,
                "bbox": bbox,
                "file_path": img_path,
                "caption": caption,
            })
            continue

        # ── Text ──────────────────────────────────────────────────────────
        text = getattr(item, "text", None)
        if not text or not text.strip():
            continue

        text = _clean_text(text)
        if not text:
            continue

        prov = item.prov[0] if getattr(item, "prov", None) else None
        page = getattr(prov, "page_no", None) if prov else None
        bbox = _extract_bbox(prov) if prov else None
        label = _label_value(item)

        raw_items.append({
            "item_type": "text",
            "text": text,
            "page": page,
            "bbox": bbox,
            "label": label,
        })

    ordered_items = build_text_chunks(raw_items)

    raw_doc: dict | None = None
    try:
        raw_doc = doc.export_to_dict()
    except Exception:
        pass

    return ParseResult(items=ordered_items, raw_doc=raw_doc)


# ─── DB persistence ───────────────────────────────────────────────────────────

async def _update_job_status(
    db: AsyncSession,
    document_id: int,
    status: str,
    error: str | None = None,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
) -> None:
    result = await db.execute(
        select(DocumentParseJob).where(DocumentParseJob.document_id == document_id)
    )
    job = result.scalar_one_or_none()
    if job is None:
        return
    job.status = status
    if error is not None:
        job.error = error
    if started_at is not None:
        job.started_at = started_at
    if finished_at is not None:
        job.finished_at = finished_at
    await db.commit()


async def _save_results(
    db: AsyncSession, document_id: int, parse_result: ParseResult
) -> None:
    """
    Two-phase persistence:
      1. Insert DocumentAsset rows for images and tables → flush → get IDs.
      2. Insert DocumentChunk rows in order; visual chunks reference their
         asset ID in meta so the reader can load the image inline.
    """
    await db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document_id))
    await db.execute(delete(DocumentAsset).where(DocumentAsset.document_id == document_id))

    # Phase 1 — assets for image and table items
    asset_by_index: dict[int, DocumentAsset] = {}
    for item in parse_result.items:
        if isinstance(item, (ImageContentItem, TableContentItem)) and item.file_path:
            asset = DocumentAsset(
                document_id=document_id,
                asset_type="table" if isinstance(item, TableContentItem) else "image",
                page=item.page_start if isinstance(item, TableContentItem) else item.page,
                bbox=item.bbox,
                file_path=item.file_path,
                meta={
                    "label": "table" if isinstance(item, TableContentItem) else "picture",
                    "caption": item.caption,
                },
            )
            db.add(asset)
            asset_by_index[item.index] = asset

    await db.flush()  # assigns PKs before referencing in chunks

    # Phase 2 — chunks in reading order
    for item in parse_result.items:
        if isinstance(item, ImageContentItem):
            asset = asset_by_index.get(item.index)
            db.add(DocumentChunk(
                document_id=document_id,
                chunk_index=item.index,
                page_start=item.page,
                page_end=item.page,
                text="",
                meta={
                    "chunk_type": "image",
                    "asset_id": asset.id if asset else None,
                    "caption": item.caption,
                    "bbox": item.bbox,
                },
            ))

        elif isinstance(item, TableContentItem):
            asset = asset_by_index.get(item.index)
            db.add(DocumentChunk(
                document_id=document_id,
                chunk_index=item.index,
                page_start=item.page_start,
                page_end=item.page_end,
                text=item.text,   # markdown — for LLM context
                meta={
                    "chunk_type": "table",
                    "asset_id": asset.id if asset else None,
                    "caption": item.caption,
                    "bbox": item.bbox,
                },
            ))

        else:
            db.add(DocumentChunk(
                document_id=document_id,
                chunk_index=item.index,
                page_start=item.page_start,
                page_end=item.page_end,
                text=item.text,
                meta={
                    "chunk_type": "text",
                    "label": item.label,
                    "bbox": item.bbox,
                },
            ))

    await db.commit()


# ─── Public interface ─────────────────────────────────────────────────────────

async def run_parse_job(document_id: int, file_path: str) -> None:
    """Entry point for FastAPI BackgroundTasks."""
    cache_dir = settings.parsed_cache_dir / str(document_id)

    async with async_session_factory() as db:
        await _update_job_status(
            db, document_id, "running", started_at=datetime.now(timezone.utc)
        )

    try:
        loop = asyncio.get_event_loop()
        parse_result = await loop.run_in_executor(
            _executor, _sync_parse, Path(file_path), cache_dir
        )
    except Exception as exc:
        logger.exception("Parse job failed for document %d", document_id)
        async with async_session_factory() as db:
            await _update_job_status(
                db, document_id, "failed", error=str(exc),
                finished_at=datetime.now(timezone.utc),
            )
        return

    async with async_session_factory() as db:
        await _save_results(db, document_id, parse_result)
        await _update_job_status(
            db, document_id, "succeeded", finished_at=datetime.now(timezone.utc)
        )

    text_count = sum(1 for i in parse_result.items if i.item_type == "text")
    image_count = sum(1 for i in parse_result.items if i.item_type == "image")
    table_count = sum(1 for i in parse_result.items if i.item_type == "table")
    logger.info(
        "Parse succeeded for doc %d: %d text, %d image, %d table chunks",
        document_id, text_count, image_count, table_count,
    )
