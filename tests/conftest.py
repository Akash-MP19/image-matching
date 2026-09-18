import os
import sys
import pytest
import asyncio
from pathlib import Path
from httpx import AsyncClient, ASGITransport

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure test uses sqlite in-memory or test db
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test_image_matching.db"
os.environ["OFFLINE_MODE"] = "true"

from src.main import app
from src.database import init_db, engine, Base, AsyncSessionLocal
from src.services.batch_processor import BatchProcessor

@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"

@pytest.fixture(autouse=True)
async def setup_test_database():
    await init_db()
    async with AsyncSessionLocal() as session:
        await BatchProcessor.process_corpus(session)
    yield

@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
