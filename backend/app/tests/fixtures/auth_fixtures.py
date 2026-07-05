import jwt
import pytest
from app.config import settings
from datetime import datetime, timezone, timedelta

def create_mock_jwt(user_id: str, email: str, expires_in_minutes: int = 60) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "email": email,
        "aud": "authenticated",
        "iat": now,
        "exp": now + timedelta(minutes=expires_in_minutes)
    }
    return jwt.encode(payload, settings.supabase_jwt_secret, algorithm="HS256")

@pytest.fixture
def auth_headers():
    token = create_mock_jwt("user-123", "test@example.com")
    return {"Authorization": f"Bearer {token}"}

@pytest.fixture
def auth_headers_pro():
    token = create_mock_jwt("pro-123", "pro@example.com")
    return {"Authorization": f"Bearer {token}"}
