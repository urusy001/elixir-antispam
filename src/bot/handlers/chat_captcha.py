import asyncio
import random

from datetime import datetime
from typing import Optional
from aiogram import Bot

from config import MOSCOW_TZ, CAPTCHA_MAX_ATTEMPTS, POLL_TIMEOUT_BUFFER, POLL_TIMEOUT_SECONDS
from src.database import get_session
from src.poll_questions import POLL_QUESTIONS_RU, PollQuestion
from src.database.chat_user import get_chat_user, update_chat_user, upsert_chat_user, ChatUserUpdate
from src.bot.handlers.chat_helpers import far_future, is_permanently_banned, build_chat_user_create
from src.bot.handlers.chat_shared import send_ephemeral_message, safe_ban_user, safe_delete_message

POLL_THREADS: dict[str, Optional[int]] = {}


async def captcha_timeout_worker(bot: Bot, poll_id: str, user_id: int, chat_id: int, thread_id: Optional[int], timeout: int = POLL_TIMEOUT_SECONDS + POLL_TIMEOUT_BUFFER) -> None:
    await asyncio.sleep(timeout)
    now = datetime.now(tz=MOSCOW_TZ)
    async with get_session() as session:
        user = await get_chat_user(session, user_id)
        if not user:
            POLL_THREADS.pop(poll_id, None)
            return
        if user.passed_poll or not user.poll_active or not user.poll_id or user.poll_id != poll_id:
            POLL_THREADS.pop(poll_id, None)
            return

        chat_id_db = user.poll_chat_id or chat_id
        msg_id = user.poll_message_id
        if chat_id_db and msg_id:
            await safe_delete_message(bot, chat_id_db, msg_id)

        attempts = (user.poll_attempts or 0) + 1
        if attempts >= CAPTCHA_MAX_ATTEMPTS:
            await update_chat_user(session, user_id, ChatUserUpdate(poll_attempts=attempts, poll_active=False, poll_chat_id=None, poll_message_id=None, poll_id=None, poll_correct_option_id=None, muted_until=far_future(now), banned_until=None, times_banned=(user.times_banned or 0) + 1))
            if chat_id_db:
                await safe_ban_user(bot, chat_id_db, user_id)
                await send_ephemeral_message(bot, chat_id_db, f'<a href="tg://user?id={user_id}">Пользователь</a> не прошёл проверку.\nКоличество попыток исчерпано. Права на отправку сообщений ограничены до решения администратора.', thread_id=thread_id)
        else:
            await update_chat_user(session, user_id, ChatUserUpdate(poll_attempts=attempts, poll_active=False, poll_chat_id=None, poll_message_id=None, poll_id=None, poll_correct_option_id=None))
            if chat_id_db:
                left = CAPTCHA_MAX_ATTEMPTS - attempts
                await send_ephemeral_message(bot, chat_id_db, f'<a href="tg://user?id={user_id}">Пользователь</a> не успел ответить на проверочный вопрос.\nОсталось попыток: {left}. Для новой попытки нужно отправить сообщение в чат.', thread_id=thread_id)

    POLL_THREADS.pop(poll_id, None)


async def start_captcha(bot: Bot, chat_id: int, user_id: int, thread_id: Optional[int]) -> None:
    now = datetime.now(tz=MOSCOW_TZ)
    async with get_session() as session:
        user = await get_chat_user(session, user_id)
        if user is None:
            await upsert_chat_user(session, build_chat_user_create(user_id, full_name="", username=None, passed_poll=False, messages_sent=0))
            user = await get_chat_user(session, user_id)

        if is_permanently_banned(user, now):
            return

        if user.poll_attempts >= CAPTCHA_MAX_ATTEMPTS and not user.passed_poll:
            await send_ephemeral_message(bot, chat_id, f'<a href="tg://user?id={user_id}">Пользователь</a> не прошёл проверку.\nПрава на отправку сообщений ограничены до решения администратора.', thread_id=thread_id)
            return

        if user.poll_active and user.poll_chat_id and user.poll_message_id:
            await send_ephemeral_message(bot, chat_id, f'<a href="tg://user?id={user_id}">Пользователь</a>, у вас уже есть активный вопрос выше. Сначала ответьте на него.', thread_id=thread_id)
            return

        poll_question: PollQuestion = random.choice(POLL_QUESTIONS_RU)
        question = poll_question.text
        options, correct_option_id = poll_question.options(True)
        await send_ephemeral_message(bot, chat_id, "Для отправки сообщений в чат необходимо пройти простую проверку.\nОтветьте на вопрос ниже. Всего доступно три попытки.", thread_id=thread_id)

        kwargs = {}
        if thread_id is not None: kwargs["message_thread_id"] = thread_id
        poll_message = await bot.send_poll(chat_id=chat_id, question=question, options=options, type="quiz", correct_option_id=correct_option_id, is_anonymous=False, open_period=POLL_TIMEOUT_SECONDS, **kwargs)

        await update_chat_user(session, user_id, ChatUserUpdate(poll_active=True, poll_chat_id=chat_id, poll_message_id=poll_message.message_id, poll_id=poll_message.poll.id, poll_correct_option_id=correct_option_id))
        POLL_THREADS[poll_message.poll.id] = thread_id
        asyncio.create_task(captcha_timeout_worker(bot, poll_message.poll.id, user_id, chat_id, thread_id, timeout=POLL_TIMEOUT_SECONDS + POLL_TIMEOUT_BUFFER))
