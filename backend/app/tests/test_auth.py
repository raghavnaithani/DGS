import sqlite3
import pytest
from app.tests.fixtures.auth_fixtures import auth_headers

def test_unauthenticated_401(client):
    """P0.4: GET /v1/profile without JWT returns 401"""
    response = client.get("/v1/profile")
    assert response.status_code == 401
    assert "Authorization header" in response.json()["detail"] or "token" in response.json()["detail"].lower()

def test_profile_404_before_onboarding(client, auth_headers):
    """P0.3: GET /v1/profile with valid JWT returns 404 (profile not created yet)"""
    response = client.get("/v1/profile", headers=auth_headers)
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()

def test_schema_user_profiles_exists(client, test_db_path):
    """P0.7: New user_profiles table exists in SQLite after server start"""
    with sqlite3.connect(test_db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_profiles'").fetchone()
        assert row is not None

def test_schema_sessions_columns(client, test_db_path):
    """P0.8: New sessions columns (user_id, status, domain, node_count) exist"""
    with sqlite3.connect(test_db_path) as conn:
        conn.row_factory = sqlite3.Row
        columns = [row["name"] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()]
        assert "user_id" in columns
        assert "status" in columns
        assert "domain" in columns
        assert "node_count" in columns
