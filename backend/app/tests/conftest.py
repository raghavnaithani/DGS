import os
import tempfile
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from .fixtures.auth_fixtures import auth_headers, auth_headers_pro, create_mock_jwt

@pytest.fixture(scope="session")
def test_db_path():
    import os
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass

@pytest.fixture(scope="session")
def client(test_db_path):
    from pathlib import Path
    from unittest.mock import patch
    # Patch DEFAULT_SQLITE_PATH in jobs_store and connection before importing main
    with patch("app.database.connection.DEFAULT_SQLITE_PATH", Path(test_db_path)), \
         patch("app.database.jobs_store.DEFAULT_SQLITE_PATH", Path(test_db_path)):
        from app.config import settings
        settings.lancedb_path = test_db_path + "_lance"
        settings.supabase_jwt_secret = "test-secret-12345"
        settings.stripe_secret_key = "sk_test_123"
        settings.stripe_pro_price_id = "price_test_123"
        settings.stripe_webhook_secret = "whsec_test"
        
        from app.main import app
        with TestClient(app) as test_client:
            yield test_client
