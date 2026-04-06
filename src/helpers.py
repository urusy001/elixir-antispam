import asyncio
import csv

from logging import Logger
from pathlib import Path
from typing import Optional
from aiogram import Bot
from aiogram.enums import ChatMemberStatus
from aiogram.types import Message

from config import ELIXIR_CHAT_ID, LABELED_CSV_PATH, STATS_CSV_PATH

STATS_CSV_HEADER = ["Message", "PredictedProba", "TrueLabel"]
LEGACY_LABELED_CSV_HEADER = ["Message", "Label"]

_csv_lock = asyncio.Lock()


def normalize_message_for_stats(text: str) -> str:
    return " ".join(str(text).strip().split())


def _read_csv_header(path: Path) -> Optional[list[str]]:
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            return next(reader)
        except StopIteration:
            return []


def _write_stats_csv_header() -> None:
    STATS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STATS_CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL).writerow(STATS_CSV_HEADER)


def ensure_stats_csv_ready() -> None:
    STATS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    stats_header = _read_csv_header(STATS_CSV_PATH)

    if stats_header == LEGACY_LABELED_CSV_HEADER:
        if LABELED_CSV_PATH.exists():
            raise RuntimeError(
                f"Cannot archive legacy dataset: {LABELED_CSV_PATH} already exists while "
                f"{STATS_CSV_PATH} is still using the old labeled schema."
            )
        STATS_CSV_PATH.replace(LABELED_CSV_PATH)
        stats_header = None

    if stats_header is None or stats_header == []:
        _write_stats_csv_header()
        return

    if stats_header != STATS_CSV_HEADER:
        raise RuntimeError(
            f"Unexpected stats CSV header in {STATS_CSV_PATH}: {stats_header!r}. "
            f"Expected {STATS_CSV_HEADER!r}."
        )


async def append_message_stats_to_csv(text: str, predicted_proba: float, true_label: Optional[int] = None) -> None:
    text = normalize_message_for_stats(text)
    async with _csv_lock:
        ensure_stats_csv_ready()
        with STATS_CSV_PATH.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([text, predicted_proba, "" if true_label is None else int(true_label)])

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
