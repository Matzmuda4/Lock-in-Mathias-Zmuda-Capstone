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
async def active_session_id(api_client: AsyncClient, auth_headers: dict) -> int:
    doc_resp = await api_client.post(
        "/documents/upload",
        data={"title": "Activity Test Doc"},
        files={"file": ("test.pdf", io.BytesIO(_MINIMAL_PDF), "application/pdf")},
        headers=auth_headers,
    )
    doc_id = doc_resp.json()["id"]
    session_resp = await api_client.post(
        "/sessions/start",
        json={"document_id": doc_id, "name": "Activity Session", "mode": "baseline"},
        headers=auth_headers,
    )
    return session_resp.json()["id"]


class TestPostActivity:
    async def test_post_scroll_forward(
        self, api_client: AsyncClient, auth_headers: dict, active_session_id: int
    ):
        resp = await api_client.post(
            "/activity",
            json={
                "session_id": active_session_id,
                "event_type": "scroll_forward",
                "payload": {"delta_px": 120, "page": 1},
            },
            headers=auth_headers,
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["event_type"] == "scroll_forward"
        assert body["payload"]["delta_px"] == 120
        assert body["session_id"] == active_session_id
        assert "id" in body
        assert "created_at" in body

    async def test_post_all_valid_event_types(
        self, api_client: AsyncClient, auth_headers: dict, active_session_id: int
    ):
        valid_types = [
            "scroll_forward",
            "scroll_backward",
            "idle",
            "blur",
            "focus",
            "heartbeat",
        ]
        for event_type in valid_types:
            resp = await api_client.post(
                "/activity",
                json={"session_id": active_session_id, "event_type": event_type},
                headers=auth_headers,
            )
            assert resp.status_code == 201, f"Failed for event_type={event_type}: {resp.text}"

    async def test_post_invalid_event_type(
        self, api_client: AsyncClient, auth_headers: dict, active_session_id: int
    ):
        resp = await api_client.post(
            "/activity",
            json={"session_id": active_session_id, "event_type": "look_away"},
            headers=auth_headers,
        )
        assert resp.status_code == 422

    async def test_post_activity_empty_payload(
        self, api_client: AsyncClient, auth_headers: dict, active_session_id: int
    ):
        resp = await api_client.post(
            "/activity",
            json={"session_id": active_session_id, "event_type": "heartbeat"},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        assert resp.json()["payload"] == {}

    async def test_post_activity_to_unknown_session(
        self, api_client: AsyncClient, auth_headers: dict
    ):
        resp = await api_client.post(
            "/activity",
            json={"session_id": 99999, "event_type": "heartbeat"},
            headers=auth_headers,
        )
        assert resp.status_code == 404

    async def test_post_activity_to_other_users_session(
        self, api_client: AsyncClient, auth_headers: dict, active_session_id: int
    ):
        resp_b = await api_client.post(
            "/auth/register",
            json={"username": "hacker2", "email": "h2@example.com", "password": "password123"},
        )
        headers_b = {"Authorization": f"Bearer {resp_b.json()['access_token']}"}

        resp = await api_client.post(
            "/activity",
            json={"session_id": active_session_id, "event_type": "heartbeat"},
            headers=headers_b,
        )
        assert resp.status_code == 404

    async def test_post_activity_to_ended_session(
        self, api_client: AsyncClient, auth_headers: dict, active_session_id: int
    ):
        await api_client.post(f"/sessions/{active_session_id}/end", headers=auth_headers)
        resp = await api_client.post(
            "/activity",
            json={"session_id": active_session_id, "event_type": "heartbeat"},
            headers=auth_headers,
        )
        assert resp.status_code == 404

    async def test_post_activity_requires_auth(
        self, api_client: AsyncClient, active_session_id: int
    ):
        resp = await api_client.post(
            "/activity",
            json={"session_id": active_session_id, "event_type": "heartbeat"},
        )
        assert resp.status_code == 401
