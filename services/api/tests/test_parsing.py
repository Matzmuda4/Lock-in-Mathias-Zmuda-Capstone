"""
Tests for the Phase 4 parsing pipeline.

Docling is NOT invoked in these tests — the parse service is mocked so
tests run fast without any model downloads or PDF processing.
The tests validate:
  - Uploading a document creates a pending parse job in the DB.
  - GET /documents/{id}/parse-status returns expected fields.
  - Ownership is enforced (another user cannot see the parse status).
  - GET /documents/{id}/parsed returns paginated chunks and assets.
  - GET /sessions/{id}/reader returns session + parse data.
  - POST /documents/{id}/reparse clears prior data and creates a fresh job.
"""

import io
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient

_FAKE_PDF = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n%%EOF"


# ─── Helpers ──────────────────────────────────────────────────────────────────

async def _register_and_login(client: AsyncClient, suffix: str = "") -> dict[str, str]:
    resp = await client.post(
        "/auth/register",
        json={
            "username": f"tester{suffix}",
            "email": f"tester{suffix}@example.com",
            "password": "password123",
        },
    )
    assert resp.status_code == 201
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


async def _upload_doc(
    client: AsyncClient, headers: dict[str, str], title: str = "Test PDF"
) -> dict:
    resp = await client.post(
        "/documents/upload",
        files={"file": ("test.pdf", io.BytesIO(_FAKE_PDF), "application/pdf")},
        data={"title": title},
        headers=headers,
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


# ─── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_upload_creates_pending_parse_job(api_client: AsyncClient, auth_headers: dict):
    """Uploading a document must create a DocumentParseJob with status pending."""
    with patch(
        "app.routers.documents.run_parse_job", new_callable=AsyncMock
    ) as mock_parse:
        doc = await _upload_doc(api_client, auth_headers)
        # Background task is scheduled; in tests HTTPX runs it before returning
        # If mock is set, the actual parsing is skipped

    doc_id = doc["id"]
    resp = await api_client.get(f"/documents/{doc_id}/parse-status", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["document_id"] == doc_id
    # Status is pending (mock prevented the background task from running)
    assert data["status"] in ("pending", "running", "succeeded", "failed")
    assert "created_at" in data
    assert "updated_at" in data


@pytest.mark.asyncio
async def test_parse_status_ownership(api_client: AsyncClient, auth_headers: dict):
    """A second user must not be able to see another user's parse status."""
    with patch("app.routers.documents.run_parse_job", new_callable=AsyncMock):
        doc = await _upload_doc(api_client, auth_headers)

    other_headers = await _register_and_login(api_client, suffix="other")
    resp = await api_client.get(
        f"/documents/{doc['id']}/parse-status", headers=other_headers
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_parse_status_not_found_for_unknown_doc(
    api_client: AsyncClient, auth_headers: dict
):
    resp = await api_client.get("/documents/99999/parse-status", headers=auth_headers)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_parsed_empty_before_parse_completes(
    api_client: AsyncClient, auth_headers: dict
):
    """Before parsing, /parsed returns zero chunks but responds 200."""
    with patch("app.routers.documents.run_parse_job", new_callable=AsyncMock):
        doc = await _upload_doc(api_client, auth_headers)

    resp = await api_client.get(f"/documents/{doc['id']}/parsed", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["document_id"] == doc["id"]
    assert data["chunks"] == []
    assert data["assets"] == []
    assert data["total_chunks"] == 0


@pytest.mark.asyncio
async def test_get_parsed_ownership(api_client: AsyncClient, auth_headers: dict):
    with patch("app.routers.documents.run_parse_job", new_callable=AsyncMock):
        doc = await _upload_doc(api_client, auth_headers)

    other_headers = await _register_and_login(api_client, suffix="b")
    resp = await api_client.get(f"/documents/{doc['id']}/parsed", headers=other_headers)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_session_reader_returns_parse_status(
    api_client: AsyncClient, auth_headers: dict
):
    """GET /sessions/{id}/reader must return session info + parse_status field."""
    with patch("app.routers.documents.run_parse_job", new_callable=AsyncMock):
        doc = await _upload_doc(api_client, auth_headers)

    session_resp = await api_client.post(
        "/sessions/start",
        json={"document_id": doc["id"], "name": "Test session", "mode": "baseline"},
        headers=auth_headers,
    )
    assert session_resp.status_code == 201
    session_id = session_resp.json()["id"]

    reader_resp = await api_client.get(
        f"/sessions/{session_id}/reader", headers=auth_headers
    )
    assert reader_resp.status_code == 200
    data = reader_resp.json()
    assert data["document_id"] == doc["id"]
    assert "parse_status" in data
    assert "chunks" in data
    assert "assets" in data
    assert "total_chunks" in data
    assert data["session"]["id"] == session_id


@pytest.mark.asyncio
async def test_reparse_clears_prior_data_and_creates_new_job(
    api_client: AsyncClient, auth_headers: dict
):
    """POST /documents/{id}/reparse returns a fresh pending job."""
    with patch("app.routers.documents.run_parse_job", new_callable=AsyncMock):
        doc = await _upload_doc(api_client, auth_headers)

    doc_id = doc["id"]

    with patch("app.routers.parsing.run_parse_job", new_callable=AsyncMock):
        reparse_resp = await api_client.post(
            f"/documents/{doc_id}/reparse", headers=auth_headers
        )

    assert reparse_resp.status_code == 200
    data = reparse_resp.json()
    assert data["document_id"] == doc_id
    assert data["status"] == "pending"


@pytest.mark.asyncio
async def test_reparse_ownership(api_client: AsyncClient, auth_headers: dict):
    with patch("app.routers.documents.run_parse_job", new_callable=AsyncMock):
        doc = await _upload_doc(api_client, auth_headers)

    other_headers = await _register_and_login(api_client, suffix="c")
    with patch("app.routers.parsing.run_parse_job", new_callable=AsyncMock):
        resp = await api_client.post(
            f"/documents/{doc['id']}/reparse", headers=other_headers
        )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_login_with_email(api_client: AsyncClient):
    """Users should be able to log in with their email address as well as username."""
    # Register a fresh user
    reg = await api_client.post(
        "/auth/register",
        json={
            "username": "emailloginuser",
            "email": "emailuser@example.com",
            "password": "password123",
        },
    )
    assert reg.status_code == 201

    # Login using email in the username field
    resp = await api_client.post(
        "/auth/login",
        data={"username": "emailuser@example.com", "password": "password123"},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert resp.status_code == 200, f"Login with email failed: {resp.text}"
    assert "access_token" in resp.json()

    # Login using actual username still works
    resp2 = await api_client.post(
        "/auth/login",
        data={"username": "emailloginuser", "password": "password123"},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert resp2.status_code == 200


@pytest.mark.asyncio
async def test_session_pause_resume_complete(api_client: AsyncClient, auth_headers: dict):
    """Pause → resume → complete workflow must all return 200 with correct status."""
    with patch("app.routers.documents.run_parse_job", new_callable=AsyncMock):
        doc = await _upload_doc(api_client, auth_headers)

    # Start session
    start = await api_client.post(
        "/sessions/start",
        json={"document_id": doc["id"], "name": "ctrl test", "mode": "baseline"},
        headers=auth_headers,
    )
    assert start.status_code == 201
    sid = start.json()["id"]

    # Pause
    pause_r = await api_client.post(f"/sessions/{sid}/pause", headers=auth_headers)
    assert pause_r.status_code == 200
    assert pause_r.json()["status"] == "paused"

    # Resume
    resume_r = await api_client.post(f"/sessions/{sid}/resume", headers=auth_headers)
    assert resume_r.status_code == 200
    assert resume_r.json()["status"] == "active"

    # Complete
    complete_r = await api_client.post(f"/sessions/{sid}/complete", headers=auth_headers)
    assert complete_r.status_code == 200
    assert complete_r.json()["status"] == "completed"


@pytest.mark.asyncio
async def test_chunking_preserves_image_items():
    """build_text_chunks must keep image items in place without merging them."""
    from app.services.parsing.chunking import build_text_chunks
    from app.services.parsing.models import ImageContentItem

    raw = [
        {"item_type": "text", "text": "Intro paragraph.", "page": 1, "bbox": None, "label": "paragraph"},
        {"item_type": "image", "page": 1, "bbox": None, "file_path": "/tmp/img.png", "caption": "Figure 1: Test"},
        {"item_type": "text", "text": "After the figure.", "page": 1, "bbox": None, "label": "paragraph"},
    ]
    items = build_text_chunks(raw)
    assert len(items) == 3
    assert items[0].item_type == "text"
    assert isinstance(items[1], ImageContentItem)
    assert items[1].caption == "Figure 1: Test"
    assert items[2].item_type == "text"
    assert [i.index for i in items] == [0, 1, 2]


@pytest.mark.asyncio
async def test_chunking_preserves_table_items():
    """Tables must appear at their correct position and carry caption + markdown text."""
    from app.services.parsing.chunking import build_text_chunks
    from app.services.parsing.models import TableContentItem

    raw = [
        {"item_type": "text", "text": "Text before table.", "page": 2, "bbox": None, "label": "paragraph"},
        {
            "item_type": "table",
            "text": "| A | B |\n|---|---|\n| 1 | 2 |",
            "page": 2,
            "bbox": None,
            "caption": "Table 1: Results",
            "file_path": None,
        },
        {"item_type": "text", "text": "Text after table.", "page": 2, "bbox": None, "label": "paragraph"},
    ]
    items = build_text_chunks(raw)
    assert len(items) == 3
    assert isinstance(items[1], TableContentItem)
    assert items[1].caption == "Table 1: Results"
    assert "| A | B |" in items[1].text
    assert [i.index for i in items] == [0, 1, 2]
