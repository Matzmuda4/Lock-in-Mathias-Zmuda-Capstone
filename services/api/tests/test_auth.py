import pytest
from httpx import AsyncClient


class TestRegister:
    async def test_register_success(self, api_client: AsyncClient):
        resp = await api_client.post(
            "/auth/register",
            json={"username": "alice", "email": "alice@example.com", "password": "secret123"},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert "access_token" in body
        assert body["token_type"] == "bearer"

    async def test_register_duplicate_username(self, api_client: AsyncClient):
        payload = {"username": "bob", "email": "bob@example.com", "password": "secret123"}
        await api_client.post("/auth/register", json=payload)
        resp = await api_client.post(
            "/auth/register",
            json={"username": "bob", "email": "different@example.com", "password": "secret123"},
        )
        assert resp.status_code == 409

    async def test_register_duplicate_email(self, api_client: AsyncClient):
        await api_client.post(
            "/auth/register",
            json={"username": "charlie", "email": "shared@example.com", "password": "secret123"},
        )
        resp = await api_client.post(
            "/auth/register",
            json={"username": "charlie2", "email": "shared@example.com", "password": "secret123"},
        )
        assert resp.status_code == 409

    async def test_register_password_too_short(self, api_client: AsyncClient):
        resp = await api_client.post(
            "/auth/register",
            json={"username": "dave", "email": "dave@example.com", "password": "short"},
        )
        assert resp.status_code == 422

    async def test_register_invalid_email(self, api_client: AsyncClient):
        resp = await api_client.post(
            "/auth/register",
            json={"username": "eve", "email": "not-an-email", "password": "secret123"},
        )
        assert resp.status_code == 422


class TestLogin:
    @pytest.fixture(autouse=True)
    async def _create_user(self, api_client: AsyncClient):
        await api_client.post(
            "/auth/register",
            json={"username": "frank", "email": "frank@example.com", "password": "password123"},
        )

    async def test_login_success(self, api_client: AsyncClient):
        resp = await api_client.post(
            "/auth/login", data={"username": "frank", "password": "password123"}
        )
        assert resp.status_code == 200
        assert "access_token" in resp.json()

    async def test_login_wrong_password(self, api_client: AsyncClient):
        resp = await api_client.post(
            "/auth/login", data={"username": "frank", "password": "wrongpassword"}
        )
        assert resp.status_code == 401

    async def test_login_unknown_user(self, api_client: AsyncClient):
        resp = await api_client.post(
            "/auth/login", data={"username": "nobody", "password": "password123"}
        )
        assert resp.status_code == 401


class TestGetMe:
    async def test_get_me_success(self, api_client: AsyncClient, auth_headers: dict):
        resp = await api_client.get("/auth/me", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["username"] == "testuser"
        assert body["email"] == "testuser@example.com"
        assert "id" in body
        assert "created_at" in body

    async def test_get_me_no_token(self, api_client: AsyncClient):
        resp = await api_client.get("/auth/me")
        assert resp.status_code == 401

    async def test_get_me_invalid_token(self, api_client: AsyncClient):
        resp = await api_client.get(
            "/auth/me", headers={"Authorization": "Bearer not.a.real.token"}
        )
        assert resp.status_code == 401
