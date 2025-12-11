import asyncio
import csv

from logging import Logger
from aiogram import Bot
from aiogram.enums import ChatMemberStatus
from aiogram.types import Message

from config import CSV_PATH, ELIXIR_CHAT_ID

_csv_lock = asyncio.Lock()  # чтобы несколько хендлеров не писали одновременно

async def append_message_to_csv(text: str, label: int | str = 0) -> None:
    """
    Добавляет строку в messages.csv.
    csv.writer сам экранирует запятые, кавычки и переносы строк.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")  # чуть-чуть нормализуем

    async with _csv_lock:
        file_exists = CSV_PATH.exists()
        with CSV_PATH.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(
                f,
                delimiter=",",
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL,  # можно csv.QUOTE_ALL если хочешь всегда в кавычках
            )
            if not file_exists: writer.writerow(["Message", "Label"])
            writer.writerow([text, label])

async def _notify_user(message: Message, text: str, timer: float | None = None, logger: Logger = None) -> None:
    if logger: logger.info("Notify user %s | text_preview=%r | timer=%s", message.from_user.id, text[:100], timer)
    x = await message.answer(text, parse_mode="HTML")
    if timer:
        await asyncio.sleep(timer)
        await x.delete()
        if logger: logger.debug("Deleted notification message for user %s", message.from_user.id)

async def CHAT_ADMIN_FILTER(message: Message, bot: Bot) -> bool:
    if getattr(message.chat, "id") not in [ELIXIR_CHAT_ID]: return False
    if message.sender_chat and message.sender_chat.id == message.chat.id: return True

    if message.from_user:
        member = await bot.get_chat_member(message.chat.id, message.from_user.id)
        return member.status in ("administrator", "creator")

    return False


async def CHAT_NOT_BANNED_FILTER(user_id: int) -> bool:
    from src.bot.main import bot
    try:
        member = await bot.get_chat_member(ELIXIR_CHAT_ID, user_id)
        if member.status in [ChatMemberStatus.KICKED]: return False
        else: return True
    except: return True
