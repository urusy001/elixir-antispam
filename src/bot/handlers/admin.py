import asyncio

from datetime import timedelta, datetime
from typing import Optional
from aiogram import Router
from aiogram.enums import ChatType
from aiogram.filters import Command
from aiogram.types import Message

from config import MOSCOW_TZ
from src.bot.handlers.chat_shared import answer_ephemeral, safe_unrestrict, safe_restrict, pass_user
from src.bot.permissions import NEW_USER
from src.image import extract_text_from_image
from src.test_classifier import is_spam
from src.helpers import CHAT_ADMIN_FILTER, _notify_user, append_message_to_csv
from src.database import get_session
from src.chat_user import update_chat_user, get_chat_user, ChatUserUpdate
from src.blocked_links import add_blocked_link, remove_blocked_link, extract_base_domain

router = Router(name="admin")
router.message.filter(CHAT_ADMIN_FILTER)
router.callback_query.filter(CHAT_ADMIN_FILTER)


@router.message(Command("block_link"))
async def handle_block_link(message: Message):
    parts = (message.text or "").strip().split(maxsplit=1)
    raw_link = parts[1].strip() if len(parts) > 1 else ""
    if not raw_link: return await answer_ephemeral(message, "<b>Ошибка команды:</b> указывайте ссылку, которую нужно заблокировать.\n\n<b>Пример:</b> <code>/block_link https://spam.example.com/promo</code>")

    domain = extract_base_domain(raw_link)
    if not domain: return await answer_ephemeral(message, "<b>Ошибка команды:</b> не удалось распознать домен в ссылке.")

    async with get_session() as session: added = await add_blocked_link(session, domain)
    if added: return await answer_ephemeral(message, f"Домен <code>{domain}</code> добавлен в блок-лист.")
    return await answer_ephemeral(message, f"Домен <code>{domain}</code> уже находится в блок-листе.")


@router.message(Command("unblock_link"))
async def handle_unblock_link(message: Message):
    parts = (message.text or "").strip().split(maxsplit=1)
    raw_link = parts[1].strip() if len(parts) > 1 else ""
    if not raw_link: return await answer_ephemeral(message, "<b>Ошибка команды:</b> указывайте ссылку, которую нужно разблокировать.\n\n<b>Пример:</b> <code>/unblock_link https://spam.example.com/promo</code>")

    domain = extract_base_domain(raw_link)
    if not domain: return await answer_ephemeral(message, "<b>Ошибка команды:</b> не удалось распознать домен в ссылке.")

    async with get_session() as session: removed = await remove_blocked_link(session, domain)
    if removed:return await answer_ephemeral(message, f"Домен <code>{domain}</code> удален из блок-листа.")
    return await answer_ephemeral(message, f"Домен <code>{domain}</code> не найден в блок-листе.")

@router.message(Command("spam"))
async def handle_spam(message: Message):
    if not message.reply_to_message: return await answer_ephemeral(message,"<b>Ошибка команды: </b>отвечайте командой на нужное сообщение")
    if not message.reply_to_message.text or message.reply_to_message.photo: return await answer_ephemeral(message, "Сообщение без текста, пометить как спам нельзя.")

    target = message.reply_to_message.from_user
    spam_text = message.reply_to_message.text.strip()
    await append_message_to_csv(spam_text, 1)

    async with get_session() as session:
        user = await get_chat_user(session, target.id)
        times_reported = (user.times_reported if user else 0) + 1
        await update_chat_user(session, target.id, ChatUserUpdate(times_reported=times_reported, accused_spam=True, last_accused_text=spam_text))

    await safe_restrict(message.bot, message.chat.id, target.id, NEW_USER)
    asyncio.create_task(_notify_user(message, f"Сообщение маркировано как <b>спам</b>, пользователь {target.mention_html()} <b>ограничен</b>\nДля возвращения прав используйте команду <code>/unmute {target.id}</code>", 300))

    try: await message.reply_to_message.delete()
    except Exception: pass
    return None


@router.message(Command("mute"))
async def handle_mute(message: Message):
    if not message.reply_to_message: return await answer_ephemeral(message,"<b>Ошибка команды: </b>отвечайте командой на нужное сообщение")

    chat_id = message.chat.id
    target = message.reply_to_message.from_user
    minutes_str = message.text.strip().removeprefix("/mute").strip()

    timer = float(minutes_str) * 60 if minutes_str.isdigit() else None
    if timer:
        mute_until = datetime.now(tz=MOSCOW_TZ) + timedelta(seconds=timer)
        async with get_session() as session:
            user = await get_chat_user(session, target.id)
            times_muted = (user.times_muted if user else 0) + 1
            await update_chat_user(session, target.id, ChatUserUpdate(muted_until=mute_until, times_muted=times_muted))

        asyncio.create_task(pass_user(chat_id, target.id, message.bot, timer))

    await safe_restrict(message.bot, chat_id, target.id, NEW_USER)
    label = "" if not timer else f" на {minutes_str} минут"
    return asyncio.create_task(_notify_user(message, f"Пользователь {target.mention_html()} успешно <b>ограничен в правах{label}</b>\nДля возвращения прав используйте команду <code>/unmute {target.id}</code>", 300))


@router.message(Command("whitelist"))
async def handle_whitelist(message: Message):
    text = message.text or ""
    args = text.strip().removeprefix("/whitelist").strip().split()
    if not args: return await answer_ephemeral(message, "<b>Использование:</b>\n<code>/whitelist add [user_id]</code> — добавить в белый список\n<code>/whitelist remove user_id</code> — убрать из белого списка\n\nМожно указать <code>user_id</code> или ответить командой на сообщение пользователя.")

    action = args[0].lower()
    if action in ("add", "on", "+"): value = True
    elif action in ("remove", "rm", "off", "del", "-"): value = False
    else: return await answer_ephemeral(message, "<b>Ошибка команды:</b> неизвестное действие.\nИспользуйте <code>add</code> или <code>remove</code>.")

    user_id: Optional[int] = None
    if len(args) >= 2 and args[1].isdigit(): user_id = int(args[1])
    elif message.reply_to_message and message.reply_to_message.from_user: user_id = message.reply_to_message.from_user.id

    if not user_id: return await answer_ephemeral(message,"Укажите <code>user_id</code> или ответьте командой на сообщение пользователя.\n\n<b>Пример:</b> <code>/whitelist add 123456789</code>")
    async with get_session() as session: user = await update_chat_user(session, user_id, ChatUserUpdate(whitelist=value))

    if not user: return await answer_ephemeral(message, "Пользователь не найден в базе.\nОн должен хотя бы один раз написать в чат, чтобы бот его сохранил.")
    status = "добавлен в <b>белый список</b>" if value else "убран из <b>белого списка</b>"
    return await answer_ephemeral(message, f"Пользователь с <code>user_id={user_id}</code> {status}.")


@router.message(Command("unmute"))
async def handle_unmute(message: Message):
    text = message.text or ""
    user_id_str = text.strip().removeprefix("/unmute").strip()

    user_id = int(user_id_str) if user_id_str.isdigit() else None
    if not user_id and message.reply_to_message: user_id = message.reply_to_message.from_user.id
    if not user_id: return await answer_ephemeral(message, "Либо укажите user_id пользователя, либо ответьте командой на его сообщение\n\n<i>Напишите @ShostakovIV в ТГ, если не знаете как получить user_id</i>")

    ok = await safe_unrestrict(message.bot, message.chat.id, user_id)
    async with get_session() as session: await update_chat_user(session, user_id, ChatUserUpdate(muted_until=None))
    return await answer_ephemeral(message, "Пользователю успешно возвращены права" if ok else "Не удалось вернуть права: пользователь не найден или уже покинул чат")


@router.message(Command("get_thread"))
async def handle_get_id(message: Message): return await answer_ephemeral(message, f"{message.message_thread_id}")

@router.message(lambda message: message.chat.type == ChatType.PRIVATE)
async def handle_private(message: Message):
    text = ""
    if message.text and message.text.strip(): text += '\n'+message.text.strip()
    if message.caption and message.caption.strip(): text += '\n'+message.caption.strip()
    if message.photo:
        largest_photo = message.photo[-1]
        file = await message.bot.get_file(largest_photo.file_id)
        image_bytes = await message.bot.download_file(file.file_path)
        ocr_text = await asyncio.to_thread(extract_text_from_image, image_bytes)
        print(ocr_text)
        if ocr_text and ocr_text.strip(): text += '\n'+ocr_text.strip()

    if not text: return await message.answer("Нужен текст")

    is_spam_flag, prob = await is_spam(text.strip())
    percent = f"{prob * 100:.2f}%"   # например, 12.34%
    verdict = "Спам" if is_spam_flag else "Не спам"
    return await message.reply(f"{verdict}: {percent}")
