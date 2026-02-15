from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.blocked_links.model import BlockedLink


async def add_blocked_link(db: AsyncSession, url: str) -> bool:
    normalized = url.strip().lower()
    result = await db.execute(select(BlockedLink).where(BlockedLink.url == normalized))
    if result.scalars().first():
        return False

    db.add(BlockedLink(url=normalized))
    await db.commit()
    return True


async def remove_blocked_link(db: AsyncSession, url: str) -> bool:
    normalized = url.strip().lower()
    result = await db.execute(select(BlockedLink).where(BlockedLink.url == normalized))
    db_link = result.scalars().first()
    if not db_link:
        return False

    await db.delete(db_link)
    await db.commit()
    return True


async def get_blocked_links(db: AsyncSession) -> set[str]:
    result = await db.execute(select(BlockedLink.url))
    return set(result.scalars().all())
