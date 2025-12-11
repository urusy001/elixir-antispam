import asyncio
from datetime import datetime
from aiogram import Dispatcher, Bot

from config import ANTISPAM_BOT_TOKEN, MOSCOW_TZ, ELIXIR_CHAT_ID
from src.bot.handlers.chat import pass_user
from src.bot.handlers import *
from src.database import get_session
from src.chat_user import get_users_with_active_mute

bot = Bot(ANTISPAM_BOT_TOKEN)
dp = Dispatcher()
dp.include_routers(admin_router, user_router, chat_router)

async def restore_mutes(chat_id: int):
    now = datetime.now(tz=MOSCOW_TZ)
    async with get_session() as session: users = await get_users_with_active_mute(session, now)
    for user in users:
        timer = (user.muted_until - now).total_seconds()
        if timer < 0: timer = 0
        asyncio.create_task(pass_user(chat_id, user.id, bot, timer))

async def run_bot():
    await bot.delete_webhook(False)
    await restore_mutes(chat_id=ELIXIR_CHAT_ID)
    await dp.start_polling(bot)
