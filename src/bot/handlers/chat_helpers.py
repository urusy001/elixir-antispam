import asyncio

from datetime import datetime, timedelta
from typing import Optional
from aiogram.enums import MessageEntityType
from aiogram.types import Message

from config import ELIXIR_CHAT_ID
from src.image import extract_text_from_image
from src.database.blocked_links import extract_base_domain
from src.database.chat_user import ChatUserCreate, ChatUserUpdate, update_chat_user

PERMANENT_RESTRICTION_DAYS = 365 * 100
PERMANENT_BAN_THRESHOLD_DAYS = 365 * 10


def CHAT_USER_FILTER(obj) -> bool:
    return getattr(obj.chat, "id", 0) in [-1003182914098, ELIXIR_CHAT_ID]


def far_future(now: datetime) -> datetime:
    return now + timedelta(days=PERMANENT_RESTRICTION_DAYS)


def is_permanently_banned(user, now: datetime) -> bool:
    if user is None:
        return False
    return bool(user.banned_until and user.banned_until > now + timedelta(days=PERMANENT_BAN_THRESHOLD_DAYS))


def build_chat_user_create(user_id: int, full_name: str, username: Optional[str], *, passed_poll: bool, messages_sent: int) -> ChatUserCreate:
    return ChatUserCreate(id=user_id, full_name=full_name, username=username, passed_poll=passed_poll, whitelist=False, muted_until=None, times_muted=0, banned_until=None, times_banned=0, messages_sent=messages_sent, times_reported=0, accused_spam=False, last_accused_text=None, poll_attempts=0, poll_active=False, poll_message_id=None, poll_chat_id=None, poll_id=None, poll_correct_option_id=None)


def compute_ai_user_risk(messages_sent: int, reports: int, mutes: int) -> float:
    risk = 1.0
    risk += min(0.25, reports * 0.08)
    risk += min(0.25, mutes * 0.10)
    if messages_sent <= 3:
        risk += 0.10
    if messages_sent >= 50 and reports == 0 and mutes == 0:
        risk -= 0.10
    return min(1.35, max(0.75, risk))


def mute_label(mute_delta: timedelta) -> str:
    if mute_delta.days >= 30:
        return "на 1 месяц"
    if mute_delta.days >= 7:
        return "на 1 неделю"
    if mute_delta.days >= 1:
        return "на 1 день"
    return "временно"


def resolve_spam_mute_delta(times_muted: int) -> Optional[timedelta]:
    if times_muted == 2:
        return timedelta(days=1)
    if times_muted == 3:
        return timedelta(weeks=1)
    if times_muted == 4:
        return timedelta(days=30)
    return None


def extract_entity_domains(message: Message) -> set[str]:
    domains: set[str] = set()
    for entity in (message.entities or []) + (message.caption_entities or []):
        if entity.type != MessageEntityType.TEXT_LINK or not entity.url:
            continue
        entity_domain = extract_base_domain(entity.url)
        if entity_domain:
            domains.add(entity_domain)
    return domains


async def extract_message_text(message: Message) -> str:
    text_parts: list[str] = []
    if message.text and message.text.strip():
        text_parts.append(message.text.strip())
    if message.caption and message.caption.strip():
        text_parts.append(message.caption.strip())
    if message.photo:
        largest_photo = message.photo[-1]
        file = await message.bot.get_file(largest_photo.file_id)
        image_bytes = await message.bot.download_file(file.file_path)
        ocr_text = await asyncio.to_thread(extract_text_from_image, image_bytes)
        print(ocr_text)
        if ocr_text and ocr_text.strip():
            text_parts.append(ocr_text.strip())
    return "\n".join(text_parts).strip()


async def apply_permanent_restriction(user_id: int, chat_user, session, now: datetime, text: str, times_muted: Optional[int] = None) -> None:
    payload = {"times_reported": (chat_user.times_reported if chat_user else 0) + 1, "accused_spam": True, "last_accused_text": text[:1024], "banned_until": None, "muted_until": far_future(now), "times_banned": (chat_user.times_banned if chat_user else 0) + 1}
    if times_muted is not None:
        payload["times_muted"] = times_muted
    await update_chat_user(session, user_id, ChatUserUpdate(**payload))
