from datetime import datetime
from aiogram import Router, Bot
from aiogram.filters import ChatMemberUpdatedFilter, JOIN_TRANSITION
from aiogram.types import Message, ChatMemberUpdated, PollAnswer

from config import MOSCOW_TZ, CAPTCHA_MAX_ATTEMPTS
from src.helpers import append_message_stats_to_csv, CHAT_ADMIN_FILTER
from src.test_classifier import predict_spam_proba
from src.database import get_session
from src.database.chat_user import update_chat_user, get_chat_user, ChatUserUpdate, upsert_chat_user
from src.bot.handlers.chat_helpers import CHAT_USER_FILTER, far_future, build_chat_user_create, extract_message_text, is_command_message
from src.bot.handlers.chat_shared import send_ephemeral_message, safe_ban_user, safe_delete_message
from src.bot.handlers.chat_captcha import POLL_THREADS, start_captcha

router = Router(name="chat")


@router.chat_member(CHAT_USER_FILTER, ChatMemberUpdatedFilter(JOIN_TRANSITION))
async def handle_new_member(event: ChatMemberUpdated):
    user = event.new_chat_member.user
    if user.is_bot: return
    async with get_session() as session:
        chat_user = await get_chat_user(session, user.id)
        if chat_user is None:
            await upsert_chat_user(session, build_chat_user_create(user.id, full_name=user.full_name or "", username=user.username, passed_poll=False, messages_sent=0))
            return None

        await update_chat_user(session, user.id, ChatUserUpdate(full_name=user.full_name or "", username=user.username, passed_poll=False, poll_attempts=0, poll_active=False, poll_message_id=None, poll_chat_id=None, poll_id=None, poll_correct_option_id=None))
    return None


@router.poll_answer()
async def handle_poll_answer(answer: PollAnswer, bot: Bot):
    user_id = answer.user.id
    poll_id = answer.poll_id
    chosen = answer.option_ids[0] if answer.option_ids else None
    now = datetime.now(tz=MOSCOW_TZ)

    async with get_session() as session:
        user = await get_chat_user(session, user_id)
        if not user or user.passed_poll: return POLL_THREADS.pop(poll_id, None)
        if not user.poll_active or not user.poll_id or user.poll_id != poll_id: return POLL_THREADS.pop(poll_id, None)

        chat_id = user.poll_chat_id
        msg_id = user.poll_message_id
        thread_id = POLL_THREADS.get(poll_id)
        if chat_id and msg_id: await safe_delete_message(bot, chat_id, msg_id)

        correct_id = user.poll_correct_option_id
        if chosen is not None and correct_id is not None and chosen == correct_id:
            await update_chat_user(session, user_id, ChatUserUpdate(passed_poll=True, poll_attempts=user.poll_attempts, poll_active=False, poll_chat_id=None, poll_message_id=None, poll_id=None, poll_correct_option_id=None))
            if chat_id: await send_ephemeral_message(bot, chat_id, f"{answer.user.mention_html()}, проверка пройдена.\nТеперь вы можете отправлять сообщения в чат.", thread_id=thread_id)
            return POLL_THREADS.pop(poll_id, None)


        attempts = (user.poll_attempts or 0) + 1
        if attempts >= CAPTCHA_MAX_ATTEMPTS:
            await update_chat_user(session, user_id, ChatUserUpdate(poll_attempts=attempts, poll_active=False, poll_chat_id=None, poll_message_id=None, poll_id=None, poll_correct_option_id=None, muted_until=far_future(now), banned_until=None, times_banned=(user.times_banned or 0) + 1))
            if chat_id:
                await safe_ban_user(bot, chat_id, user_id)
                await send_ephemeral_message(bot, chat_id, f"{answer.user.mention_html()}, проверка не пройдена.\nКоличество попыток исчерпано. Права на отправку сообщений ограничены до решения администратора.", thread_id=thread_id)

        else:
            await update_chat_user(session, user_id, ChatUserUpdate(poll_attempts=attempts, poll_active=False, poll_chat_id=None, poll_message_id=None, poll_id=None, poll_correct_option_id=None))
            if chat_id:
                left = CAPTCHA_MAX_ATTEMPTS - attempts
                await send_ephemeral_message(bot, chat_id, f"{answer.user.mention_html()}, ответ неверный.\nОсталось попыток: {left}. Для новой попытки отправьте любое сообщение в чат.", thread_id=thread_id)

        return POLL_THREADS.pop(poll_id, None)


@router.message(CHAT_USER_FILTER, lambda message: getattr(message, "message_thread_id", None) in [None, 27775] and not is_command_message(message))
async def handle_chat_message(message: Message):
    text = await extract_message_text(message)
    if not text: return None
    user = message.from_user
    if not user: return None

    trusted_user = await CHAT_ADMIN_FILTER(message, message.bot)
    passed_poll = True

    async with get_session() as session:
        chat_user = await get_chat_user(session, user.id)
        if chat_user is None:
            await upsert_chat_user(session, build_chat_user_create(user.id, full_name=user.full_name or "", username=user.username, passed_poll=True, messages_sent=1))
        else:
            trusted_user = trusted_user or bool(chat_user.whitelist)
            passed_poll = trusted_user or bool(chat_user.passed_poll)
            new_messages_sent = (chat_user.messages_sent or 0) + 1
            await update_chat_user(session, user.id, ChatUserUpdate(messages_sent=new_messages_sent))

    if not passed_poll:
        # New joiners must pass the quiz before any message is accepted or scored.
        await safe_delete_message(message.bot, message.chat.id, message.message_id)
        await start_captcha(message.bot, message.chat.id, user.id, message.message_thread_id)
        return None

    proba_spam = await predict_spam_proba(text)
    return await append_message_stats_to_csv(text, proba_spam)
