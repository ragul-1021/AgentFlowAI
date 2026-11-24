from __future__ import annotations

from httpx import ASGITransport, AsyncClient

from app.config import Settings
from app.main import create_app
from tests.fakes import FakeDatabase


async def test_health_reports_configuration(client: AsyncClient) -> None:
    response = await client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "agentflow"
    assert body["environment"] == "development"
    assert body["version"] == "0.1.0"
    assert body["mock_mode"] is True
    assert body["providers"] == {"gemini": True, "groq": False, "tavily": False}
    assert body["database"] == {"status": "up", "revision": "0001", "latency_ms": 1}

    # Secrets must never be reflected back to the client.
    assert "test-secret" not in response.text
    assert "test-gemini-key" not in response.text
    assert "postgresql" not in response.text


async def test_health_returns_503_when_database_is_down(
    settings: Settings,
) -> None:
    app = create_app(settings, database=FakeDatabase(status="down"))
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://agentflow.test") as http:
        response = await http.get("/api/health")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["database"] == {"status": "down", "revision": None, "latency_ms": 1}


async def test_health_returns_503_without_a_database(settings: Settings) -> None:
    app = create_app(settings)
    app.state.database = None
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://agentflow.test") as http:
        response = await http.get("/api/health")

    assert response.status_code == 503
    assert response.json()["database"]["status"] == "down"


async def test_health_has_security_headers(client: AsyncClient) -> None:
    response = await client.get("/api/health")

    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["X-Request-ID"]


async def test_request_id_is_echoed(client: AsyncClient) -> None:
    response = await client.get("/api/health", headers={"X-Request-ID": "req-test-123"})
    assert response.headers["X-Request-ID"] == "req-test-123"


async def test_invalid_request_id_is_replaced(client: AsyncClient) -> None:
    response = await client.get(
        "/api/health",
        headers={"X-Request-ID": "x" * 200},
    )
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] != "x" * 200
    assert len(response.headers["X-Request-ID"]) == 32


async def test_cors_allows_dev_origin(client: AsyncClient) -> None:
    response = await client.get(
        "/api/health",
        headers={"Origin": "http://localhost:5173"},
    )
    assert (
        response.headers.get("access-control-allow-origin")
        == "http://localhost:5173"
    )


async def test_cors_rejects_unknown_origin(client: AsyncClient) -> None:
    response = await client.get(
        "/api/health",
        headers={"Origin": "http://evil.example"},
    )
    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


async def test_root_route(client: AsyncClient) -> None:
    response = await client.get("/")
    assert response.status_code == 200
    assert response.json()["health"] == "/api/health"


async def test_unknown_route_returns_404(client: AsyncClient) -> None:
    response = await client.get("/api/does-not-exist")
    assert response.status_code == 404
