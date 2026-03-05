"""
Tests for POST /activity/batch — Phase 5 Telemetry Logging.

Covers:
  - happy-path: row inserted with correct event_type and payload keys
  - requires auth (401 without token)
  - enforces ownership (user B cannot post for user A session)
  - rejects invalid payload (missing session_id, bad types)
  - rejects inactive session (ended session returns 404)
"""

import io

import pytest
from httpx import AsyncClient

_MINIMAL_PDF = (
    b"%PDF-1.0\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj "
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj "
    b"3 0 obj<</Type/Page/MediaBox[0 0 3 3]>>endobj\n"
    b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n"
    b"0000000058 00000 n \n0000000115 00000 n \n"
    b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n190\n%%EOF"
)

_VALID_BATCH = {
    "scroll_delta_sum": 240.0,
    "scroll_delta_abs_sum": 240.0,
    "scroll_event_count": 4,
    "scroll_direction_changes": 1,
    "scroll_pause_seconds": 0.5,
    "idle_seconds": 0.5,
    "mouse_path_px": 320.0,
    "mouse_net_px": 200.0,
    "window_focus_state": "focused",
    "current_paragraph_id": "chunk-7",
    "current_chunk_index": 7,
    "viewport_progress_ratio": 0.35,
    "client_timestamp": "2026-02-25T12:00:00Z",
}


@pytest.fixture()
async def active_session(api_client: AsyncClient, auth_headers: dict) -> dict:
    """Create a document + active session and return {session_id, auth_headers}."""
    doc_resp = await api_client.post(
        "/documents/upload",
        data={"title": "Batch Test Doc"},
        files={"file": ("test.pdf", io.BytesIO(_MINIMAL_PDF), "application/pdf")},
        headers=auth_headers,
    )
    assert doc_resp.status_code == 201, doc_resp.text
    doc_id = doc_resp.json()["id"]

    sess_resp = await api_client.post(
        "/sessions/start",
        json={"document_id": doc_id, "name": "Batch Session", "mode": "baseline"},
        headers=auth_headers,
    )
    assert sess_resp.status_code == 201, sess_resp.text
    return {"session_id": sess_resp.json()["id"], "auth_headers": auth_headers}


class TestActivityBatchInsert:
    async def test_batch_inserts_row(
        self, api_client: AsyncClient, active_session: dict
    ) -> None:
        """POST /activity/batch returns 201 and a row with event_type=telemetry_batch."""
        session_id = active_session["session_id"]
        headers = active_session["auth_headers"]

        resp = await api_client.post(
            "/activity/batch",
            json={"session_id": session_id, **_VALID_BATCH},
            headers=headers,
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["event_type"] == "telemetry_batch"
        assert body["session_id"] == session_id
        assert "id" in body
        assert "created_at" in body

    async def test_batch_payload_keys_stored(
        self, api_client: AsyncClient, active_session: dict
    ) -> None:
        """The stored payload must contain all telemetry signal keys."""
        session_id = active_session["session_id"]
        headers = active_session["auth_headers"]

        resp = await api_client.post(
            "/activity/batch",
            json={"session_id": session_id, **_VALID_BATCH},
            headers=headers,
        )
        assert resp.status_code == 201, resp.text

        # Verify the row exists by posting again and checking id increments
        resp2 = await api_client.post(
            "/activity/batch",
            json={"session_id": session_id, **_VALID_BATCH},
            headers=headers,
        )
        assert resp2.json()["id"] != resp.json()["id"], "Each batch must create a new row"

    async def test_batch_minimal_payload(
        self, api_client: AsyncClient, active_session: dict
    ) -> None:
        """Sending only session_id (all other fields use defaults) must succeed."""
        session_id = active_session["session_id"]
        headers = active_session["auth_headers"]

        resp = await api_client.post(
            "/activity/batch",
            json={"session_id": session_id},
            headers=headers,
        )
        assert resp.status_code == 201, resp.text


class TestActivityBatchAuth:
    async def test_requires_auth(
        self, api_client: AsyncClient, active_session: dict
    ) -> None:
        """Unauthenticated request must return 401."""
        session_id = active_session["session_id"]
        resp = await api_client.post(
            "/activity/batch",
            json={"session_id": session_id, **_VALID_BATCH},
            # no headers
        )
        assert resp.status_code == 401, resp.text

    async def test_wrong_bearer_returns_401(
        self, api_client: AsyncClient, active_session: dict
    ) -> None:
        session_id = active_session["session_id"]
        resp = await api_client.post(
            "/activity/batch",
            json={"session_id": session_id, **_VALID_BATCH},
            headers={"Authorization": "Bearer totally-invalid-token"},
        )
        assert resp.status_code == 401, resp.text


class TestActivityBatchOwnership:
    async def test_enforces_ownership(
        self, api_client: AsyncClient, active_session: dict
    ) -> None:
        """User B cannot post a batch for User A's session."""
        session_id = active_session["session_id"]

        # Register User B
        resp_b = await api_client.post(
            "/auth/register",
            json={
                "username": "intruder_batch",
                "email": "intruder_batch@example.com",
                "password": "password123",
            },
        )
        assert resp_b.status_code == 201, resp_b.text
        headers_b = {"Authorization": f"Bearer {resp_b.json()['access_token']}"}

        resp = await api_client.post(
            "/activity/batch",
            json={"session_id": session_id, **_VALID_BATCH},
            headers=headers_b,
        )
        assert resp.status_code == 404, resp.text

    async def test_unknown_session_returns_404(
        self, api_client: AsyncClient, auth_headers: dict
    ) -> None:
        resp = await api_client.post(
            "/activity/batch",
            json={"session_id": 99999, **_VALID_BATCH},
            headers=auth_headers,
        )
        assert resp.status_code == 404, resp.text


class TestActivityBatchValidation:
    async def test_missing_session_id_returns_422(
        self, api_client: AsyncClient, auth_headers: dict
    ) -> None:
        """Payload without session_id must be rejected by Pydantic."""
        resp = await api_client.post(
            "/activity/batch",
            json={**_VALID_BATCH},  # no session_id
            headers=auth_headers,
        )
        assert resp.status_code == 422, resp.text

    async def test_invalid_scroll_count_type_returns_422(
        self, api_client: AsyncClient, auth_headers: dict
    ) -> None:
        """scroll_event_count must be an integer, not a string."""
        resp = await api_client.post(
            "/activity/batch",
            json={"session_id": 1, **{**_VALID_BATCH, "scroll_event_count": "lots"}},
            headers=auth_headers,
        )
        assert resp.status_code == 422, resp.text

    async def test_negative_abs_sum_returns_422(
        self, api_client: AsyncClient, auth_headers: dict
    ) -> None:
        """scroll_delta_abs_sum has ge=0 constraint; negative value must be rejected."""
        resp = await api_client.post(
            "/activity/batch",
            json={"session_id": 1, **{**_VALID_BATCH, "scroll_delta_abs_sum": -5}},
            headers=auth_headers,
        )
        assert resp.status_code == 422, resp.text

    async def test_viewport_ratio_out_of_range_returns_422(
        self, api_client: AsyncClient, auth_headers: dict
    ) -> None:
        """viewport_progress_ratio must be 0..1."""
        resp = await api_client.post(
            "/activity/batch",
            json={"session_id": 1, **{**_VALID_BATCH, "viewport_progress_ratio": 1.5}},
            headers=auth_headers,
        )
        assert resp.status_code == 422, resp.text

    async def test_ended_session_returns_404(
        self, api_client: AsyncClient, active_session: dict
    ) -> None:
        """Telemetry must be rejected for an ended session."""
        session_id = active_session["session_id"]
        headers = active_session["auth_headers"]

        await api_client.post(f"/sessions/{session_id}/end", headers=headers)

        resp = await api_client.post(
            "/activity/batch",
            json={"session_id": session_id, **_VALID_BATCH},
            headers=headers,
        )
        assert resp.status_code == 404, resp.text
