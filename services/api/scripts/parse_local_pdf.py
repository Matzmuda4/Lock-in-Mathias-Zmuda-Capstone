#!/usr/bin/env python3
"""
Manual parse verification script.

Usage (from repo root):
    cd services/api
    source .venv/bin/activate
    python scripts/parse_local_pdf.py ../../pdfs/test.pdf

Place any PDF under ./pdfs/ at the repo root before running.
Extracted images are written to ./parsed_cache/local_test/.
The script prints a summary: chunk count, asset count, and the first 3 chunks.
"""

import sys
from pathlib import Path

# Allow running from any directory
sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../../pdfs/test.pdf")
    pdf_path = pdf_path.resolve()

    if not pdf_path.exists():
        print(f"[error] File not found: {pdf_path}")
        print("Place a PDF at ./pdfs/test.pdf (repo root) and re-run.")
        sys.exit(1)

    cache_dir = Path("parsed_cache") / "local_test"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[parse] {pdf_path}")
    print(f"[cache] {cache_dir.resolve()}")

    from app.services.parsing.parser import _sync_parse  # type: ignore[import]

    result = _sync_parse(pdf_path, cache_dir)

    print(f"\n{'─' * 50}")
    print(f"  Chunks : {len(result.chunks)}")
    print(f"  Assets : {len(result.assets)}")
    print(f"{'─' * 50}")

    print("\n── First 3 chunks ──")
    for chunk in result.chunks[:3]:
        preview = chunk.text[:120].replace("\n", " ")
        print(f"  [{chunk.chunk_index}] page={chunk.page_start}  {preview!r}")

    if result.assets:
        print("\n── Assets ──")
        for asset in result.assets:
            print(f"  [{asset.asset_index}] page={asset.page}  path={asset.file_path}")

    print("\n[done]")


if __name__ == "__main__":
    main()
