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


@pytest.fixture()
async def doc_id(api_client: AsyncClient, auth_headers: dict) -> int:
    resp = await api_client.post(
        "/documents/upload",
        data={"title": "Session Test Doc"},
        files={"file": ("test.pdf", io.BytesIO(_MINIMAL_PDF), "application/pdf")},
        headers=auth_headers,
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


class TestStartSession:
    async def test_start_baseline_session(
        self, api_client: AsyncClient, auth_headers: dict, doc_id: int
    ):
        resp = await api_client.post(
            "/sessions/start",
            json={"document_id": doc_id, "name": "Study block 1", "mode": "baseline"},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["status"] == "active"
        assert body["mode"] == "baseline"
        assert body["ended_at"] is None
        assert body["duration_seconds"] is None

    async def test_start_adaptive_session(
        self, api_client: AsyncClient, auth_headers: dict, doc_id: int
    ):
        resp = await api_client.post(
            "/sessions/start",
            json={"document_id": doc_id, "name": "Adaptive block", "mode": "adaptive"},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        assert resp.json()["mode"] == "adaptive"

    async def test_start_session_invalid_mode(
        self, api_client: AsyncClient, auth_headers: dict, doc_id: int
    ):
        resp = await api_client.post(
            "/sessions/start",
            json={"document_id": doc_id, "name": "Bad mode", "mode": "turbo"},
            headers=auth_headers,
        )
        assert resp.status_code == 422

    async def test_start_session_document_not_found(
        self, api_client: AsyncClient, auth_headers: dict
    ):
        resp = await api_client.post(
            "/sessions/start",
            json={"document_id": 99999, "name": "Ghost", "mode": "baseline"},
            headers=auth_headers,
        )
        assert resp.status_code == 404


class TestPauseResume:
    @pytest.fixture()
    async def active_session_id(
        self, api_client: AsyncClient, auth_headers: dict, doc_id: int
    ) -> int:
        resp = await api_client.post(
            "/sessions/start",
            json={"document_id": doc_id, "name": "PR Test", "mode": "baseline"},
            headers=auth_headers,
        )
        return resp.json()["id"]

    async def test_pause_active_session(
        self, api_client: AsyncClient, auth_headers: dict, active_session_id: int
    ):
        resp = await api_client.post(
            f"/sessions/{active_session_id}/pause", headers=auth_headers
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "paused"

    async def test_pause_already_paused_fails(
        self, api_client: AsyncClient, auth_headers: dict, active_session_id: int
    ):
        await api_client.post(f"/sessions/{active_session_id}/pause", headers=auth_headers)
        resp = await api_client.post(
            f"/sessions/{active_session_id}/pause", headers=auth_headers
        )
        assert resp.status_code == 400

    async def test_resume_paused_session(
        self, api_client: AsyncClient, auth_headers: dict, active_session_id: int
    ):
        await api_client.post(f"/sessions/{active_session_id}/pause", headers=auth_headers)
        resp = await api_client.post(
            f"/sessions/{active_session_id}/resume", headers=auth_headers
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "active"

    async def test_resume_active_session_fails(
        self, api_client: AsyncClient, auth_headers: dict, active_session_id: int
    ):
        resp = await api_client.post(
            f"/sessions/{active_session_id}/resume", headers=auth_headers
        )
        assert resp.status_code == 400


class TestEndSession:
    @pytest.fixture()
    async def active_session_id(
        self, api_client: AsyncClient, auth_headers: dict, doc_id: int
    ) -> int:
        resp = await api_client.post(
            "/sessions/start",
            json={"document_id": doc_id, "name": "End Test", "mode": "baseline"},
            headers=auth_headers,
        )
        return resp.json()["id"]

    async def test_end_session(
        self, api_client: AsyncClient, auth_headers: dict, active_session_id: int
    ):
        resp = await api_client.post(
            f"/sessions/{active_session_id}/end", headers=auth_headers
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ended"
        assert body["ended_at"] is not None
        assert body["duration_seconds"] is not None
        assert body["duration_seconds"] >= 0

    async def test_end_already_ended_fails(
        self, api_client: AsyncClient, auth_headers: dict, active_session_id: int
    ):
        await api_client.post(f"/sessions/{active_session_id}/end", headers=auth_headers)
        resp = await api_client.post(
            f"/sessions/{active_session_id}/end", headers=auth_headers
        )
        assert resp.status_code == 400


class TestListAndGet:
    async def test_list_sessions_empty(self, api_client: AsyncClient, auth_headers: dict):
        resp = await api_client.get("/sessions", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    async def test_list_sessions(
        self, api_client: AsyncClient, auth_headers: dict, doc_id: int
    ):
        for name in ("S1", "S2"):
            await api_client.post(
                "/sessions/start",
                json={"document_id": doc_id, "name": name, "mode": "baseline"},
                headers=auth_headers,
            )
        resp = await api_client.get("/sessions", headers=auth_headers)
        assert resp.json()["total"] == 2

    async def test_get_session(
        self, api_client: AsyncClient, auth_headers: dict, doc_id: int
    ):
        create_resp = await api_client.post(
            "/sessions/start",
            json={"document_id": doc_id, "name": "Solo", "mode": "adaptive"},
            headers=auth_headers,
        )
        sid = create_resp.json()["id"]
        resp = await api_client.get(f"/sessions/{sid}", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["id"] == sid

    async def test_get_session_not_found(self, api_client: AsyncClient, auth_headers: dict):
        resp = await api_client.get("/sessions/99999", headers=auth_headers)
        assert resp.status_code == 404
