import json

from contextlib import asynccontextmanager
from sqlalchemy import inspect, Column, String, Boolean, BigInteger, DateTime, select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from datetime import datetime
from typing import Optional
from pydantic import BaseModel

from config import ASYNC_DATABASE_URL, MOSCOW_TZ


@asynccontextmanager
async def get_session():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


engine = create_async_engine(
    ASYNC_DATABASE_URL,
    echo=False,
    pool_size=10,  # realistic default for async apps
    max_overflow=20,  # allow up to 30 concurrent (10 + 20 overflow)
    pool_timeout=60,  # seconds to wait before raising TimeoutError
    pool_recycle=1800,  # recycle every 30 mins to avoid stale sockets
    pool_pre_ping=True,  # ensures dropped connections are refreshed
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
    class_=AsyncSession,
)


class BaseModelMixin:
    """Adds universal to_dict() and to_json() methods to SQLAlchemy models."""

    def to_dict(self) -> dict:
        """Convert the SQLAlchemy object to a clean dictionary."""
        return {
            c.key: getattr(self, c.key)
            for c in inspect(self).mapper.column_attrs
        }

    def to_json(self) -> str:
        """Convert the SQLAlchemy object to a JSON string."""
        return json.dumps(self.to_dict(), default=str)


Base = declarative_base(cls=BaseModelMixin)
