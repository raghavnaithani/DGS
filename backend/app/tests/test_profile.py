import pytest
import sqlite3

def test_profile_upsert_sets_onboarding_complete(client, auth_headers, test_db_path):
    payload = {
        "expertise_level": "beginner",
        "risk_tolerance": 8,
        "values": ["Financial growth", "Freedom"],
        "life_situation": "Testing profile upsert"
    }
    
    res = client.post("/v1/profile", json=payload, headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["onboarding_complete"] is True
    assert data["expertise_level"] == "beginner"

def test_profile_roundtrip(client, auth_headers):
    payload = {
        "expertise_level": "expert",
        "risk_tolerance": 2,
        "values": ["Stability", "Family"],
        "life_situation": "Stable job"
    }
    client.post("/v1/profile", json=payload, headers=auth_headers)
    
    res = client.get("/v1/profile", headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["expertise_level"] == "expert"
    assert data["risk_tolerance"] == 2
    assert data["values"] == ["Stability", "Family"]
    assert data["life_situation"] == "Stable job"

def test_patch_partial_update(client, auth_headers):
    payload = {
        "expertise_level": "intermediate",
        "risk_tolerance": 5,
        "values": [],
        "life_situation": ""
    }
    client.post("/v1/profile", json=payload, headers=auth_headers)
    
    res = client.patch("/v1/profile", json={"risk_tolerance": 9}, headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["risk_tolerance"] == 9
    assert data["expertise_level"] == "intermediate"
