from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import MOSCOW_TZ
from src.chat_user.model import ChatUser
from src.chat_user.schema import ChatUserUpdate, ChatUserCreate


async def get_chat_user(db: AsyncSession, user_id: int) -> Optional[ChatUser]:
    """Получить пользователя по id."""
    result = await db.execute(select(ChatUser).where(ChatUser.id == user_id))
    return result.scalars().first()


async def get_chat_users(
        db: AsyncSession,
        skip: int = 0,
        limit: int = 100,
) -> list[ChatUser]:
    """Получить список пользователей с пагинацией."""
    result = await db.execute(
        select(ChatUser)
        .order_by(ChatUser.id)
        .offset(skip)
        .limit(limit)
    )
    return list(result.scalars().all())


# ========== CREATE / UPDATE / UPSERT ==========


async def create_chat_user(db: AsyncSession, user_in: ChatUserCreate) -> ChatUser:
    """Создать нового ChatUser на основе ChatUserCreate."""
    db_user = ChatUser(
        id=user_in.id,
        full_name=user_in.full_name,
        username=user_in.username,
        passed_poll=user_in.passed_poll,
        whitelist=user_in.whitelist,
        muted_until=user_in.muted_until,
        times_muted=user_in.times_muted,
        banned_until=user_in.banned_until,
        times_banned=user_in.times_banned,
        messages_sent=user_in.messages_sent,
        times_reported=user_in.times_reported,
        accused_spam=user_in.accused_spam,
        last_accused_text=user_in.last_accused_text,
        poll_attempts=user_in.poll_attempts,
        poll_active=user_in.poll_active,
        poll_message_id=user_in.poll_message_id,
        poll_chat_id=user_in.poll_chat_id,
        poll_id=user_in.poll_id,
        poll_correct_option_id=user_in.poll_correct_option_id,
    )
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    return db_user


async def update_chat_user(
        db: AsyncSession,
        user_id: int,
        user_in: ChatUserUpdate,
) -> Optional[ChatUser]:
    """
    Частичное обновление пользователя по id.

    ВАЖНО: интерфейс именно (db, user_id, ChatUserUpdate),
    как ожидает твой chat.py.
    """
    db_user = await get_chat_user(db, user_id)
    if db_user is None:
        return None

    # Pydantic v2: model_dump; если у тебя v1 – можно заменить на .dict()
    data = user_in.model_dump(exclude_unset=True)

    for field, value in data.items():
        setattr(db_user, field, value)

    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    return db_user


async def upsert_chat_user(db: AsyncSession, user_in: ChatUserCreate) -> ChatUser:
    """
    Если пользователь есть — обновляем его полями из ChatUserCreate,
    если нет — создаём.
    """
    db_user = await get_chat_user(db, user_in.id)
    if db_user is None:
        return await create_chat_user(db, user_in)

    # все поля, кроме id, в ChatUserUpdate
    update_data = ChatUserUpdate(
        **user_in.model_dump(exclude={"id"})
    )
    updated = await update_chat_user(db, user_in.id, update_data)
    # updated не None, т.к. db_user уже был
    return updated  # type: ignore[return-value]


async def delete_chat_user(db: AsyncSession, user_id: int) -> bool:
    """Удалить пользователя по id."""
    db_user = await get_chat_user(db, user_id)
    if db_user is None:
        return False
    await db.delete(db_user)
    await db.commit()
    return True


# ========== SINGLE-FIELD HELPERS (ОБЁРТКИ НАД update_chat_user) ==========


async def set_whitelist(
        db: AsyncSession,
        user_id: int,
        value: bool = True,
) -> Optional[ChatUser]:
    """Установить/снять whitelist."""
    return await update_chat_user(
        db,
        user_id,
        ChatUserUpdate(whitelist=value),
    )


async def set_passed_poll(
        db: AsyncSession,
        user_id: int,
        value: bool = True,
) -> Optional[ChatUser]:
    """Отметить, что пользователь прошёл / не прошёл капчу."""
    return await update_chat_user(
        db,
        user_id,
        ChatUserUpdate(passed_poll=value),
    )


async def set_muted_until(
        db: AsyncSession,
        user_id: int,
        muted_until: Optional[datetime],
) -> Optional[ChatUser]:
    """Обновить дату окончания мута."""
    return await update_chat_user(
        db,
        user_id,
        ChatUserUpdate(muted_until=muted_until),
    )


async def set_banned_until(
        db: AsyncSession,
        user_id: int,
        banned_until: Optional[datetime],
) -> Optional[ChatUser]:
    """Обновить дату окончания бана."""
    return await update_chat_user(
        db,
        user_id,
        ChatUserUpdate(banned_until=banned_until),
    )


# ========== COUNTERS (INCREMENT-ХЕЛПЕРЫ) ==========


async def increment_messages_sent(
        db: AsyncSession,
        user_id: int,
        delta: int = 1,
) -> Optional[ChatUser]:
    """Увеличить messages_sent на delta."""
    db_user = await get_chat_user(db, user_id)
    if db_user is None:
        return None

    new_value = (db_user.messages_sent or 0) + delta
    return await update_chat_user(
        db,
        user_id,
        ChatUserUpdate(messages_sent=new_value),
    )


async def increment_times_reported(
        db: AsyncSession,
        user_id: int,
        delta: int = 1,
        accused_spam: Optional[bool] = None,
) -> Optional[ChatUser]:
    """Увеличить счётчик жалоб, опционально обновить accused_spam."""
    db_user = await get_chat_user(db, user_id)
    if db_user is None:
        return None

    new_value = (db_user.times_reported or 0) + delta

    update_data = ChatUserUpdate(times_reported=new_value)
    if accused_spam is not None:
        update_data.accused_spam = accused_spam

    return await update_chat_user(db, user_id, update_data)


async def increment_times_muted(
        db: AsyncSession,
        user_id: int,
        delta: int = 1,
) -> Optional[ChatUser]:
    """Увеличить times_muted на delta."""
    db_user = await get_chat_user(db, user_id)
    if db_user is None:
        return None

    new_value = (db_user.times_muted or 0) + delta
    return await update_chat_user(
        db,
        user_id,
        ChatUserUpdate(times_muted=new_value),
    )


async def increment_times_banned(
        db: AsyncSession,
        user_id: int,
        delta: int = 1,
) -> Optional[ChatUser]:
    """Увеличить times_banned на delta."""
    db_user = await get_chat_user(db, user_id)
    if db_user is None:
        return None

    new_value = (db_user.times_banned or 0) + delta
    return await update_chat_user(
        db,
        user_id,
        ChatUserUpdate(times_banned=new_value),
    )


# ========== SELECTION HELPERS ==========


async def get_users_with_active_mute(
        db: AsyncSession,
        now: Optional[datetime] = None,
) -> list[ChatUser]:
    """
    Получить всех пользователей, у которых сейчас активный мут.
    Можно использовать в фоновой задаче, чтобы синхронизировать права.
    """
    if now is None:
        now = datetime.now(tz=MOSCOW_TZ)

    result = await db.execute(
        select(ChatUser).where(
            ChatUser.muted_until.is_not(None),
            ChatUser.muted_until > now,
            )
    )
    return list(result.scalars().all())
