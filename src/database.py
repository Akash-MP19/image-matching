import logging
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

from src.config import settings

logger = logging.getLogger(__name__)

Base = declarative_base()

# Determine database URL: check environment DATABASE_URL first
db_url = settings.DATABASE_URL
# If set to default postgres url, check if postgresql is used
if "postgresql" in db_url:
    connect_args = {}
else:
    connect_args = {"check_same_thread": False} if "sqlite" in db_url else {}

try:
    engine = create_async_engine(
        db_url,
        echo=False,
        future=True,
        connect_args=connect_args
    )
except Exception as e:
    logger.warning(f"Could not connect to {db_url}: {e}. Falling back to SQLite.")
    db_url = "sqlite+aiosqlite:///./image_matching.db"
    engine = create_async_engine(
        db_url,
        echo=False,
        future=True,
        connect_args={"check_same_thread": False}
    )

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables verified/created successfully.")
