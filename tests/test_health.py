"""
Tests for GET /health.
"""


def test_health_returns_200(client):
    r = client.get("/health")
    assert r.status_code == 200


def test_health_schema(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert isinstance(body["db_connected"], bool)
    assert "version" in body


def test_health_db_connected(client):
    """The DB must be reachable when tests run."""
    body = client.get("/health").json()
    assert body["db_connected"] is True, (
        "Database is not connected — check DATABASE_URL in .env"
    )
