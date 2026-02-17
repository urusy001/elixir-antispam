import asyncio
import logging

from datetime import datetime
from typing import Optional
from aiogram import Bot
from aiogram.types import Message

from src.bot.permissions import NEW_USER, USER_PASSED
from src.database import get_session
from src.database.chat_user import update_chat_user, ChatUserUpdate


async def safe_delete_message(bot: Bot, chat_id: int, message_id: int) -> None:
    try:
        await bot.delete_message(chat_id, message_id)
    except Exception:
        pass


async def delete_later(bot: Bot, chat_id: int, message_id: int, delay: int = 31) -> None:
    await asyncio.sleep(delay)
    try:
        await bot.delete_message(chat_id, message_id)
    except Exception as e:
        logging.getLogger("main").warning(f"Couldnt delete message with id {message_id} in chat {chat_id}: {e.__class__.__name__}")


async def send_ephemeral_message(bot: Bot, chat_id: int, text: str, *, thread_id: Optional[int] = None, parse_mode: str = "HTML"):
    kwargs = {"parse_mode": parse_mode}
    if thread_id is not None:
        kwargs["message_thread_id"] = thread_id
    msg = await bot.send_message(chat_id, text, **kwargs)
    asyncio.create_task(delete_later(bot, chat_id, msg.message_id, 31))
    return msg


async def answer_ephemeral(message: Message, text: str):
    msg = await message.answer(text, parse_mode="HTML")
    asyncio.create_task(delete_later(message.bot, message.chat.id, message.message_id, 31))
    asyncio.create_task(delete_later(message.bot, message.chat.id, msg.message_id, 31))
    return msg


async def safe_restrict(bot: Bot, chat_id: int, user_id: int, permissions, *, until_date: Optional[datetime] = None) -> bool:
    try: member = await bot.get_chat_member(chat_id, user_id)
    except Exception: return False
    if member.status in ("left", "kicked"): return False
    try:
        await bot.restrict_chat_member(chat_id, user_id, permissions, until_date=until_date)
        return True
    except Exception: return False


async def safe_ban_user(
    bot: Bot,
    chat_id: int,
    user_id: int,
    *,
    until_date: Optional[datetime] = None,
    revoke_messages: bool = True,
) -> bool:
    _ = revoke_messages
    return await safe_restrict(bot, chat_id, user_id, NEW_USER, until_date=until_date)


async def safe_unrestrict(bot: Bot, chat_id: int, user_id: int) -> bool:
    return await safe_restrict(bot, chat_id, user_id, USER_PASSED)



async def pass_user(_chat_id: int, user_id: int, _bot: Bot, timer: Optional[float] = 24 * 60 * 60):
    await asyncio.sleep(timer)
    async with get_session() as session:
        await update_chat_user(session, user_id, ChatUserUpdate(muted_until=None))
