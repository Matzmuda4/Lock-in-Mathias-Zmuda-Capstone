import io

from httpx import AsyncClient

# Minimal valid PDF bytes — enough for content validation without a real PDF lib
_MINIMAL_PDF = (
    b"%PDF-1.0\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj "
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj "
    b"3 0 obj<</Type/Page/MediaBox[0 0 3 3]>>endobj\n"
    b"xref\n0 4\n"
    b"0000000000 65535 f \n"
    b"0000000009 00000 n \n"
    b"0000000058 00000 n \n"
    b"0000000115 00000 n \n"
    b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n190\n%%EOF"
)


def _pdf_file(filename: str = "test.pdf") -> tuple:
    return (filename, io.BytesIO(_MINIMAL_PDF), "application/pdf")


class TestUpload:
    async def test_upload_success(self, api_client: AsyncClient, auth_headers: dict):
        resp = await api_client.post(
            "/documents/upload",
            data={"title": "My Paper"},
            files={"file": _pdf_file()},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["title"] == "My Paper"
        assert body["filename"] == "test.pdf"
        assert body["file_size"] > 0
        assert "id" in body

    async def test_upload_non_pdf_rejected(self, api_client: AsyncClient, auth_headers: dict):
        resp = await api_client.post(
            "/documents/upload",
            data={"title": "Bad file"},
            files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
            headers=auth_headers,
        )
        assert resp.status_code == 400

    async def test_upload_requires_auth(self, api_client: AsyncClient):
        resp = await api_client.post(
            "/documents/upload",
            data={"title": "No auth"},
            files={"file": _pdf_file()},
        )
        assert resp.status_code == 401


class TestListDocuments:
    async def test_list_empty(self, api_client: AsyncClient, auth_headers: dict):
        resp = await api_client.get("/documents", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 0
        assert body["documents"] == []

    async def test_list_after_upload(self, api_client: AsyncClient, auth_headers: dict):
        await api_client.post(
            "/documents/upload",
            data={"title": "Doc A"},
            files={"file": _pdf_file("a.pdf")},
            headers=auth_headers,
        )
        await api_client.post(
            "/documents/upload",
            data={"title": "Doc B"},
            files={"file": _pdf_file("b.pdf")},
            headers=auth_headers,
        )
        resp = await api_client.get("/documents", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["total"] == 2

    async def test_list_isolated_between_users(self, api_client: AsyncClient, auth_headers: dict):
        """Documents uploaded by user A must not appear in user B's list."""
        await api_client.post(
            "/documents/upload",
            data={"title": "User A doc"},
            files={"file": _pdf_file()},
            headers=auth_headers,
        )
        # Register a second user
        resp_b = await api_client.post(
            "/auth/register",
            json={"username": "userB", "email": "userb@example.com", "password": "password123"},
        )
        headers_b = {"Authorization": f"Bearer {resp_b.json()['access_token']}"}

        resp = await api_client.get("/documents", headers=headers_b)
        assert resp.json()["total"] == 0


class TestGetDocumentFile:
    async def test_download_success(self, api_client: AsyncClient, auth_headers: dict):
        upload = await api_client.post(
            "/documents/upload",
            data={"title": "Downloadable"},
            files={"file": _pdf_file()},
            headers=auth_headers,
        )
        doc_id = upload.json()["id"]
        resp = await api_client.get(f"/documents/{doc_id}/file", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/pdf"

    async def test_download_not_found(self, api_client: AsyncClient, auth_headers: dict):
        resp = await api_client.get("/documents/99999/file", headers=auth_headers)
        assert resp.status_code == 404


class TestDeleteDocument:
    async def test_delete_success(self, api_client: AsyncClient, auth_headers: dict):
        upload = await api_client.post(
            "/documents/upload",
            data={"title": "To delete"},
            files={"file": _pdf_file()},
            headers=auth_headers,
        )
        doc_id = upload.json()["id"]
        resp = await api_client.delete(f"/documents/{doc_id}", headers=auth_headers)
        assert resp.status_code == 204

        # Confirm it's gone
        list_resp = await api_client.get("/documents", headers=auth_headers)
        assert list_resp.json()["total"] == 0

    async def test_delete_other_users_document_fails(
        self, api_client: AsyncClient, auth_headers: dict
    ):
        upload = await api_client.post(
            "/documents/upload",
            data={"title": "Protected"},
            files={"file": _pdf_file()},
            headers=auth_headers,
        )
        doc_id = upload.json()["id"]

        resp_b = await api_client.post(
            "/auth/register",
            json={"username": "hacker", "email": "hacker@example.com", "password": "password123"},
        )
        headers_b = {"Authorization": f"Bearer {resp_b.json()['access_token']}"}

        resp = await api_client.delete(f"/documents/{doc_id}", headers=headers_b)
        assert resp.status_code == 404
