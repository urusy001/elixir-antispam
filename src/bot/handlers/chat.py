import asyncio
import logging
import random

from datetime import datetime, timedelta
from typing import Optional
from aiogram import Router, Bot
from aiogram.filters import ChatMemberUpdatedFilter, JOIN_TRANSITION
from aiogram.types import Message, ChatMemberUpdated, PollAnswer

from config import MOSCOW_TZ, ELIXIR_CHAT_ID
from src.bot.permissions import NEW_USER, USER_PASSED
from src.image import extract_text_from_image
from src.poll_questions import POLL_QUESTIONS_RU, PollQuestion
from src.test_classifier import is_spam
from src.helpers import append_message_to_csv, CHAT_ADMIN_FILTER
from src.database import get_session
from src.chat_user import update_chat_user, get_chat_user, ChatUserUpdate, ChatUserCreate, upsert_chat_user

router = Router(name="chat")

CHAT_USER_FILTER = lambda obj: getattr(obj.chat, "id", 0) in [-1003182914098, ELIXIR_CHAT_ID]

CAPTCHA_MAX_ATTEMPTS = 3
POLL_TIMEOUT_SECONDS = 30
POLL_TIMEOUT_BUFFER = 3  # чуть больше, чем open_period

POLL_THREADS: dict[str, Optional[int]] = {}

async def delete_later(bot: Bot, chat_id: int, message_id: int, delay: int = 31) -> None:
    """Удалить сообщение через delay секунд."""
    await asyncio.sleep(delay)
    try: await bot.delete_message(chat_id, message_id)
    except Exception as e: logging.getLogger("main").warning(f"Couldnt delete message with id {message_id} in chat {chat_id}: {e.__class__.__name__}")

async def send_ephemeral_message(bot: Bot, chat_id: int, text: str, *, thread_id: Optional[int] = None, parse_mode: str = "HTML",):
    """Отправить сообщение и удалить его через ttl секунд."""
    kwargs = {"parse_mode": parse_mode}
    if thread_id is not None: kwargs["message_thread_id"] = thread_id

    msg = await bot.send_message(chat_id, text, **kwargs)
    asyncio.create_task(delete_later(bot, chat_id, msg.message_id, 31))
    return msg

async def answer_ephemeral(message: Message, text: str,):
    """Ответить на сообщение и удалить ответ через ttl секунд."""
    msg = await message.answer(text, parse_mode="HTML")
    asyncio.create_task(delete_later(message.bot, message.chat.id, message.message_id, 31))
    asyncio.create_task(delete_later(message.bot, message.chat.id, msg.message_id, 31))
    return msg

async def safe_restrict(bot: Bot, chat_id: int, user_id: int, permissions) -> bool:
    try: member = await bot.get_chat_member(chat_id, user_id)
    except Exception: return False
    if member.status in ("left", "kicked"): return False
    try:
        await bot.restrict_chat_member(chat_id, user_id, permissions)
        return True
    except Exception: return False

async def safe_unrestrict(bot: Bot, chat_id: int, user_id: int) -> bool: return await safe_restrict(bot, chat_id, user_id, USER_PASSED)

async def pass_user(chat_id: int, user_id: int, bot: Bot, timer: Optional[float] = 24 * 60 * 60):
    """Авто-снятие мута по таймеру."""
    await asyncio.sleep(timer)
    await safe_unrestrict(bot, chat_id, user_id)
    async with get_session() as session: await update_chat_user(session, user_id, ChatUserUpdate(muted_until=None))

async def captcha_timeout_worker(bot: Bot, poll_id: str, user_id: int, chat_id: int, thread_id: Optional[int], timeout: int = POLL_TIMEOUT_SECONDS + POLL_TIMEOUT_BUFFER) -> None:
    """
    Ждём timeout секунд после создания опроса.
    Если за это время пользователь не прошел капчу (passed_poll=False)
    и этот poll всё ещё активен у юзера — считаем как "не успел ответить".
    """
    await asyncio.sleep(timeout)
    now = datetime.now(tz=MOSCOW_TZ)

    async with get_session() as session:
        user = await get_chat_user(session, user_id)
        if not user:
            POLL_THREADS.pop(poll_id, None)
            return

        if (
                user.passed_poll
                or not user.poll_active
                or not user.poll_id
                or user.poll_id != poll_id
        ):
            POLL_THREADS.pop(poll_id, None)
            return

        chat_id_db = user.poll_chat_id or chat_id
        msg_id = user.poll_message_id

        # Удаляем сообщение с опросом
        if chat_id_db and msg_id:
            try: await bot.delete_message(chat_id_db, msg_id)
            except Exception: pass

        attempts = (user.poll_attempts or 0) + 1
        if attempts >= CAPTCHA_MAX_ATTEMPTS:
            far_future = now + timedelta(days=365 * 100)
            await update_chat_user(
                session,
                user_id,
                ChatUserUpdate(
                    poll_attempts=attempts,
                    poll_active=False,
                    poll_chat_id=None,
                    poll_message_id=None,
                    poll_id=None,
                    poll_correct_option_id=None,
                    muted_until=far_future,
                    banned_until=None,
                    times_banned=(user.times_banned or 0) + 1,
                ),
            )

            if chat_id_db:
                await safe_restrict(bot, chat_id_db, user_id, NEW_USER)
                text = (
                    f'<a href="tg://user?id={user_id}">Пользователь</a> не прошёл проверку.\n'
                    "Количество попыток исчерпано. Права на отправку сообщений ограничены до решения администратора."
                )
                await send_ephemeral_message(bot, chat_id_db, text, thread_id=thread_id)
        else:
            await update_chat_user(
                session,
                user_id,
                ChatUserUpdate(
                    poll_attempts=attempts,
                    poll_active=False,
                    poll_chat_id=None,
                    poll_message_id=None,
                    poll_id=None,
                    poll_correct_option_id=None,
                ),
            )
            left = CAPTCHA_MAX_ATTEMPTS - attempts
            if chat_id_db:
                text = (
                    f'<a href="tg://user?id={user_id}">Пользователь</a> не успел ответить на проверочный вопрос.\n'
                    f"Осталось попыток: {left}. Для новой попытки нужно отправить сообщение в чат."
                )
                await send_ephemeral_message(bot, chat_id_db, text, thread_id=thread_id)

    POLL_THREADS.pop(poll_id, None)

async def start_captcha(bot: Bot, chat_id: int, user_id: int, thread_id: Optional[int]):
    """
    Старт / повтор капчи.
    Всегда отправляем сообщения и опрос в тот же thread_id (если он есть).
    """
    now = datetime.now(tz=MOSCOW_TZ)
    async with get_session() as session:
        user = await get_chat_user(session, user_id)
        if user is None:
            await upsert_chat_user(
                session,
                ChatUserCreate(
                    id=user_id,
                    full_name="",
                    username=None,
                    passed_poll=False,
                    whitelist=False,
                    muted_until=None,
                    times_muted=0,
                    banned_until=None,
                    times_banned=0,
                    messages_sent=0,
                    times_reported=0,
                    accused_spam=False,
                    last_accused_text=None,
                    poll_attempts=0,
                    poll_active=False,
                    poll_message_id=None,
                    poll_chat_id=None,
                    poll_id=None,
                    poll_correct_option_id=None,
                ),
            )
            user = await get_chat_user(session, user_id)

        # Уже бессрочно забанен
        if user.banned_until and user.banned_until > now + timedelta(days=365 * 10):
            return

        # Лимит попыток исчерпан
        if user.poll_attempts >= CAPTCHA_MAX_ATTEMPTS and not user.passed_poll:
            text = (
                f'<a href="tg://user?id={user_id}">Пользователь</a> не прошёл проверку.\n'
                "Права на отправку сообщений ограничены до решения администратора."
            )
            await send_ephemeral_message(bot, chat_id, text, thread_id=thread_id)
            return

        # Уже есть активный опрос
        if user.poll_active and user.poll_chat_id and user.poll_message_id:
            text = (
                f'<a href="tg://user?id={user_id}">Пользователь</a>, '
                "у вас уже есть активный вопрос выше. Сначала ответьте на него."
            )
            await send_ephemeral_message(bot, chat_id, text, thread_id=thread_id)
            return

        # Новый опрос
        poll_question: PollQuestion = random.choice(POLL_QUESTIONS_RU)
        question = poll_question.text
        options, correct_option_id = poll_question.options(True)

        info_text = (
            "Для отправки сообщений в чат необходимо пройти простую проверку.\n"
            "Ответьте на вопрос ниже. Всего доступно три попытки."
        )
        await send_ephemeral_message(bot, chat_id, info_text, thread_id=thread_id)

        kwargs = {}
        if thread_id is not None:
            kwargs["message_thread_id"] = thread_id

        poll_message = await bot.send_poll(
            chat_id=chat_id,
            question=question,
            options=options,
            type="quiz",
            correct_option_id=correct_option_id,
            is_anonymous=False,
            open_period=POLL_TIMEOUT_SECONDS,
            **kwargs,
        )

        # Сохраняем состояние капчи в БД
        await update_chat_user(
            session,
            user_id,
            ChatUserUpdate(
                poll_active=True,
                poll_chat_id=chat_id,
                poll_message_id=poll_message.message_id,
                poll_id=poll_message.poll.id,
                poll_correct_option_id=correct_option_id,
            ),
        )

        # Сохраняем соответствие poll_id -> thread_id в памяти
        POLL_THREADS[poll_message.poll.id] = thread_id

        # Запускаем таймаут-воркер
        asyncio.create_task(
            captcha_timeout_worker(
                bot,
                poll_message.poll.id,
                user_id,
                chat_id,
                thread_id,
                timeout=POLL_TIMEOUT_SECONDS + POLL_TIMEOUT_BUFFER,
            )
        )

@router.chat_member(CHAT_USER_FILTER, ChatMemberUpdatedFilter(JOIN_TRANSITION))
async def handle_new_member(event: ChatMemberUpdated):
    user = event.new_chat_member.user
    if user.is_bot: return

    async with get_session() as session:
        await upsert_chat_user(
            session,
            ChatUserCreate(
                id=user.id,
                full_name=user.full_name or "",
                username=user.username,
                passed_poll=False,
                whitelist=False,
                muted_until=None,
                times_muted=0,
                banned_until=None,
                times_banned=0,
                messages_sent=0,
                times_reported=0,
                accused_spam=False,
                last_accused_text=None,
                poll_attempts=0,
                poll_active=False,
                poll_message_id=None,
                poll_chat_id=None,
                poll_id=None,
                poll_correct_option_id=None,
            ),
        )

@router.poll_answer()
async def handle_poll_answer(answer: PollAnswer, bot: Bot):
    user_id = answer.user.id
    poll_id = answer.poll_id
    chosen = answer.option_ids[0] if answer.option_ids else None

    now = datetime.now(tz=MOSCOW_TZ)
    async with get_session() as session:
        user = await get_chat_user(session, user_id)
        if not user or user.passed_poll:
            POLL_THREADS.pop(poll_id, None)
            return

        # не его активный опрос → игнор
        if not user.poll_active or not user.poll_id or user.poll_id != poll_id:
            POLL_THREADS.pop(poll_id, None)
            return

        chat_id = user.poll_chat_id
        msg_id = user.poll_message_id
        thread_id = POLL_THREADS.get(poll_id)

        # Удаляем сообщение с опросом
        if chat_id and msg_id:
            try:
                await bot.delete_message(chat_id, msg_id)
            except Exception:
                pass

        correct_id = user.poll_correct_option_id

        # ---- ВЕРНЫЙ ОТВЕТ ----
        if chosen is not None and correct_id is not None and chosen == correct_id:
            await update_chat_user(
                session,
                user_id,
                ChatUserUpdate(
                    passed_poll=True,
                    poll_attempts=user.poll_attempts,  # не трогаем
                    poll_active=False,
                    poll_chat_id=None,
                    poll_message_id=None,
                    poll_id=None,
                    poll_correct_option_id=None,
                ),
            )

            if chat_id:
                text = (
                    f"{answer.user.mention_html()}, проверка пройдена.\n"
                    "Теперь вы можете отправлять сообщения в чат."
                )
                await send_ephemeral_message(bot, chat_id, text, thread_id=thread_id)

            POLL_THREADS.pop(poll_id, None)
            return

        # ---- НЕВЕРНЫЙ ОТВЕТ ----
        attempts = (user.poll_attempts or 0) + 1

        if attempts >= CAPTCHA_MAX_ATTEMPTS:
            far_future = now + timedelta(days=365 * 100)
            await update_chat_user(
                session,
                user_id,
                ChatUserUpdate(
                    poll_attempts=attempts,
                    poll_active=False,
                    poll_chat_id=None,
                    poll_message_id=None,
                    poll_id=None,
                    poll_correct_option_id=None,
                    muted_until=far_future,
                    banned_until=None,
                    times_banned=(user.times_banned or 0) + 1,
                ),
            )

            if chat_id:
                await safe_restrict(bot, chat_id, user_id, NEW_USER)
                text = (
                    f"{answer.user.mention_html()}, проверка не пройдена.\n"
                    "Количество попыток исчерпано. Права на отправку сообщений ограничены до решения администратора."
                )
                await send_ephemeral_message(bot, chat_id, text, thread_id=thread_id)
        else:
            await update_chat_user(
                session,
                user_id,
                ChatUserUpdate(
                    poll_attempts=attempts,
                    poll_active=False,
                    poll_chat_id=None,
                    poll_message_id=None,
                    poll_id=None,
                    poll_correct_option_id=None,
                ),
            )
            left = CAPTCHA_MAX_ATTEMPTS - attempts
            if chat_id:
                text = (
                    f"{answer.user.mention_html()}, ответ неверный.\n"
                    f"Осталось попыток: {left}. Для новой попытки отправьте любое сообщение в чат."
                )
                await send_ephemeral_message(bot, chat_id, text, thread_id=thread_id)

        POLL_THREADS.pop(poll_id, None)

@router.message(CHAT_USER_FILTER, lambda message: getattr(message, "message_thread_id", None) is None)
async def handle_chat_message(message: Message):
    text_parts: list[str] = []

    if message.text and message.text.strip(): text_parts.append(message.text.strip())

    if message.caption and message.caption.strip(): text_parts.append(message.caption.strip())

    if message.photo:
        largest_photo = message.photo[-1]
        file = await message.bot.get_file(largest_photo.file_id)
        image_bytes = await message.bot.download_file(file.file_path)
        ocr_text = await asyncio.to_thread(extract_text_from_image, image_bytes)
        print(ocr_text)
        if ocr_text and ocr_text.strip(): text_parts.append(ocr_text.strip())

    text = "\n".join(text_parts).strip()
    if not text: return None

    user = message.from_user
    if not user: return None

    now = datetime.now(tz=MOSCOW_TZ)
    whitelist = await CHAT_ADMIN_FILTER(message, message.bot)

    async with get_session() as session:
        chat_user = await get_chat_user(session, user.id)

        if chat_user is None:
            await upsert_chat_user(
                session,
                ChatUserCreate(
                    id=user.id,
                    full_name=user.full_name or "",
                    username=user.username,
                    passed_poll=True,
                    whitelist=False,
                    muted_until=None,
                    times_muted=0,
                    banned_until=None,
                    times_banned=0,
                    messages_sent=1,
                    times_reported=0,
                    accused_spam=False,
                    last_accused_text=None,
                    poll_attempts=0,
                    poll_active=False,
                    poll_message_id=None,
                    poll_chat_id=None,
                    poll_id=None,
                    poll_correct_option_id=None,
                ),
            )
            chat_user = await get_chat_user(session, user.id)
            passed_poll = True
        else:
            whitelist = whitelist or bool(chat_user.whitelist)
            passed_poll = bool(chat_user.passed_poll)
            new_messages_sent = (chat_user.messages_sent or 0) + 1
            await update_chat_user(
                session,
                user.id,
                ChatUserUpdate(messages_sent=new_messages_sent),
            )

        if chat_user.banned_until and chat_user.banned_until > now + timedelta(days=365 * 10):
            await safe_restrict(message.bot, message.chat.id, user.id, NEW_USER)
            try: await message.delete()
            except Exception: pass
            return None

    if whitelist:
        return await append_message_to_csv(text, 0)


    if not passed_poll:
        await start_captcha(
            message.bot,
            message.chat.id,
            user.id,
            message.message_thread_id,
        )
        try: await message.delete()
        except Exception: pass
        return None

    result, p = await is_spam(text)
    print(result, p, text)

    if result:
        async with get_session() as session:
            chat_user = await get_chat_user(session, user.id)
            times_reported = (chat_user.times_reported if chat_user else 0) + 1
            if p >= 0.8:
                far_future = now + timedelta(days=365 * 100)
                await update_chat_user(
                    session,
                    user.id,
                    ChatUserUpdate(
                        times_reported=times_reported,
                        accused_spam=True,
                        last_accused_text=text[:1024],
                        banned_until=None,
                        muted_until=far_future,
                        times_banned=(chat_user.times_banned if chat_user else 0) + 1,
                    ),
                )

                await safe_restrict(message.bot, message.chat.id, user.id, NEW_USER)
                info_text = (
                    "Сообщение с очень высокой вероятностью является спамом.\n"
                    f"Пользователь {user.mention_html()} ограничен в отправке сообщений <b>без срока</b>."
                )
                await answer_ephemeral(message, info_text)

            else:
                new_count = (chat_user.times_muted if chat_user else 0) + 1

                if new_count == 1:
                    await update_chat_user(
                        session,
                        user.id,
                        ChatUserUpdate(
                            times_reported=times_reported,
                            accused_spam=True,
                            last_accused_text=text[:1024],
                            times_muted=new_count,
                        ),
                    )
                    info_text = (
                        "Сообщение похоже на спам.\n"
                        "Сообщение удалено. Это первое предупреждение, ограничения не выданы."
                    )
                    await answer_ephemeral(message, info_text)
                else:
                    if new_count == 2:
                        mute_delta = timedelta(days=1)
                    elif new_count == 3:
                        mute_delta = timedelta(weeks=1)
                    elif new_count == 4:
                        mute_delta = timedelta(days=30)
                    else:
                        mute_delta = None  # дальше бессрочно

                    if mute_delta is None:
                        far_future = now + timedelta(days=365 * 100)
                        await update_chat_user(
                            session,
                            user.id,
                            ChatUserUpdate(
                                times_reported=times_reported,
                                accused_spam=True,
                                last_accused_text=text[:1024],
                                banned_until=None,
                                muted_until=far_future,
                                times_banned=(chat_user.times_banned if chat_user else 0) + 1,
                                times_muted=new_count,
                            ),
                        )
                        await safe_restrict(message.bot, message.chat.id, user.id, NEW_USER)

                        info_text = (
                            "Сообщение похоже на спам.\n"
                            f"Пользователь {user.mention_html()} ограничен в отправке сообщений <b>без срока</b> "
                            "из-за повторяющегося спама."
                        )
                        await answer_ephemeral(message, info_text)
                    else:
                        mute_until = now + mute_delta
                        await update_chat_user(
                            session,
                            user.id,
                            ChatUserUpdate(
                                times_reported=times_reported,
                                accused_spam=True,
                                last_accused_text=text[:1024],
                                muted_until=mute_until,
                                times_muted=new_count,
                            ),
                        )
                        await safe_restrict(message.bot, message.chat.id, user.id, NEW_USER)

                        if mute_delta.days >= 30:
                            label = "на 1 месяц"
                        elif mute_delta.days >= 7:
                            label = "на 1 неделю"
                        elif mute_delta.days >= 1:
                            label = "на 1 день"
                        else:
                            label = "временно"

                        info_text = (
                            "Сообщение похоже на спам.\n"
                            f"Пользователь {user.mention_html()} автоматически ограничен в правах {label}.\n"
                            f"Для досрочного возвращения прав используйте команду <code>/unmute {user.id}</code>"
                        )
                        await answer_ephemeral(message, info_text)
                        asyncio.create_task(
                            pass_user(
                                message.chat.id,
                                user.id,
                                message.bot,
                                mute_delta.total_seconds(),
                            )
                        )

        try: await message.delete()
        except Exception: pass

    return await append_message_to_csv(text, int(result))