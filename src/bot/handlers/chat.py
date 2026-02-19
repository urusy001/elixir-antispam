import asyncio

from datetime import datetime
from aiogram import Router, Bot
from aiogram.filters import ChatMemberUpdatedFilter, JOIN_TRANSITION
from aiogram.types import Message, ChatMemberUpdated, PollAnswer

from config import MOSCOW_TZ, CAPTCHA_MAX_ATTEMPTS
from src.bot.permissions import NEW_USER
from src.helpers import append_message_to_csv, CHAT_ADMIN_FILTER
from src.test_classifier import is_spam
from src.database import get_session
from src.database.blocked_links import get_blocked_links, extract_blocked_targets_from_text
from src.database.chat_user import update_chat_user, get_chat_user, ChatUserUpdate, upsert_chat_user
from src.bot.handlers.chat_helpers import CHAT_USER_FILTER, far_future, is_permanently_banned, build_chat_user_create, mute_label, resolve_spam_mute_delta, extract_entity_domains, extract_message_text, apply_permanent_restriction
from src.bot.handlers.chat_shared import send_ephemeral_message, answer_ephemeral, safe_restrict, safe_ban_user, pass_user, safe_delete_message
from src.bot.handlers.chat_captcha import POLL_THREADS, start_captcha

router = Router(name="chat")


@router.chat_member(CHAT_USER_FILTER, ChatMemberUpdatedFilter(JOIN_TRANSITION))
async def handle_new_member(event: ChatMemberUpdated):
    user = event.new_chat_member.user
    if user.is_bot: return
    async with get_session() as session: await upsert_chat_user(session, build_chat_user_create(user.id, full_name=user.full_name or "", username=user.username, passed_poll=False, messages_sent=0))


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


@router.message(CHAT_USER_FILTER, lambda message: getattr(message, "message_thread_id", None) is None)
async def handle_chat_message(message: Message):
    text = await extract_message_text(message)
    if not text: return None
    blocked_targets_in_message = extract_blocked_targets_from_text(text) | extract_entity_domains(message)
    user = message.from_user
    if not user: return None

    now = datetime.now(tz=MOSCOW_TZ)
    whitelist = await CHAT_ADMIN_FILTER(message, message.bot)
    passed_poll = True
    matched_blocked_target = None

    async with get_session() as session:
        chat_user = await get_chat_user(session, user.id)
        if chat_user is None:
            await upsert_chat_user(session, build_chat_user_create(user.id, full_name=user.full_name or "", username=user.username, passed_poll=True, messages_sent=1))
            chat_user = await get_chat_user(session, user.id)

        else:
            whitelist = whitelist or bool(chat_user.whitelist)
            passed_poll = bool(chat_user.passed_poll)
            new_messages_sent = (chat_user.messages_sent or 0) + 1
            await update_chat_user(session, user.id, ChatUserUpdate(messages_sent=new_messages_sent))

        if is_permanently_banned(chat_user, now):
            await safe_ban_user(message.bot, message.chat.id, user.id)
            await safe_delete_message(message.bot, message.chat.id, message.message_id)
            return None

        if not whitelist and blocked_targets_in_message:
            blocked_values = await get_blocked_links(session)
            matched_blocked_target = next((target for target in blocked_targets_in_message if target in blocked_values), None)
            if matched_blocked_target: await apply_permanent_restriction(user.id, chat_user, session, now, text)

    if matched_blocked_target:
        await safe_ban_user(message.bot, message.chat.id, user.id)
        await safe_delete_message(message.bot, message.chat.id, message.message_id)
        await answer_ephemeral(message, "Вы были ограничены в правах за рекламу.")
        return await append_message_to_csv(text, 1)

    if whitelist: return await append_message_to_csv(text, 0)
    if not passed_poll:
        await start_captcha(message.bot, message.chat.id, user.id, message.message_thread_id)
        await safe_delete_message(message.bot, message.chat.id, message.message_id)
        return None

    result, p = await is_spam(text)
    print(result, p, text)

    if result:
        async with get_session() as session:
            chat_user = await get_chat_user(session, user.id)
            times_reported = (chat_user.times_reported if chat_user else 0) + 1
            if p >= 0.8:
                await apply_permanent_restriction(user.id, chat_user, session, now, text)
                await safe_ban_user(message.bot, message.chat.id, user.id)
                await answer_ephemeral(message, f"Сообщение с очень высокой вероятностью является спамом.\nПользователь {user.mention_html()} ограничен в отправке сообщений <b>без срока</b>.")

            else:
                new_count = (chat_user.times_muted if chat_user else 0) + 1
                if new_count == 1:
                    await update_chat_user(session, user.id, ChatUserUpdate(times_reported=times_reported, accused_spam=True, last_accused_text=text[:1024], times_muted=new_count))
                    await answer_ephemeral(message, "Сообщение похоже на спам.\nСообщение удалено. Это первое предупреждение, ограничения не выданы.")

                else:
                    mute_delta = resolve_spam_mute_delta(new_count)
                    if mute_delta is None:
                        await apply_permanent_restriction(user.id, chat_user, session, now, text, times_muted=new_count)
                        await safe_ban_user(message.bot, message.chat.id, user.id)
                        await answer_ephemeral(message, f"Сообщение похоже на спам.\nПользователь {user.mention_html()} ограничен в отправке сообщений <b>без срока</b> из-за повторяющегося спама.")

                    else:
                        mute_until = now + mute_delta
                        await update_chat_user(session, user.id, ChatUserUpdate(times_reported=times_reported, accused_spam=True, last_accused_text=text[:1024], muted_until=mute_until, times_muted=new_count))
                        await safe_restrict(message.bot, message.chat.id, user.id, NEW_USER, until_date=mute_until)
                        await answer_ephemeral(message, f"Сообщение похоже на спам.\nПользователь {user.mention_html()} автоматически ограничен в правах {mute_label(mute_delta)}.\nДля досрочного возвращения прав используйте команду <code>/unmute {user.id}</code>")
                        asyncio.create_task(pass_user(message.chat.id, user.id, message.bot, mute_delta.total_seconds()))

        await safe_delete_message(message.bot, message.chat.id, message.message_id)
    return await append_message_to_csv(text, int(result))
