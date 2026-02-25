"""
FastAPI smoke tests — verify the app starts, CORS is configured,
and the /health endpoint returns the expected payload.
"""

import pytest


class TestHealthEndpoint:
    async def test_health_returns_200(self, api_client):
        response = await api_client.get("/health")
        assert response.status_code == 200, (
            f"Expected 200 from /health, got {response.status_code}"
        )

    async def test_health_payload_has_status_ok(self, api_client):
        response = await api_client.get("/health")
        body = response.json()
        assert body.get("status") == "ok", (
            f"Expected status='ok' in health response, got: {body}"
        )

    async def test_health_payload_has_version(self, api_client):
        response = await api_client.get("/health")
        body = response.json()
        assert "version" in body, f"Missing 'version' key in health response: {body}"
        assert isinstance(body["version"], str) and len(body["version"]) > 0

    async def test_docs_endpoint_accessible(self, api_client):
        """Swagger UI should be reachable in all environments."""
        response = await api_client.get("/docs")
        assert response.status_code == 200, (
            f"Expected 200 from /docs, got {response.status_code}"
        )

    async def test_openapi_schema_accessible(self, api_client):
        """OpenAPI JSON schema must be valid and include our app title."""
        response = await api_client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert schema.get("info", {}).get("title") == "Lock-In API", (
            f"Unexpected OpenAPI title: {schema.get('info', {}).get('title')}"
        )

    async def test_cors_header_present_for_tauri_origin(self, api_client):
        """
        The Tauri dev server runs at http://localhost:1420.
        A preflight OPTIONS request from that origin must receive the
        Access-Control-Allow-Origin header.
        """
        response = await api_client.options(
            "/health",
            headers={
                "Origin": "http://localhost:1420",
                "Access-Control-Request-Method": "GET",
            },
        )
        # FastAPI responds 200 to OPTIONS when CORS middleware is active
        assert response.status_code == 200, (
            f"OPTIONS /health returned {response.status_code} — CORS middleware may be missing"
        )
        allow_origin = response.headers.get("access-control-allow-origin", "")
        assert allow_origin in ("http://localhost:1420", "*"), (
            f"Expected CORS allow-origin for localhost:1420, got: '{allow_origin}'"
        )
