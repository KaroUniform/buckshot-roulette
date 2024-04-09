import asyncio
import logging
from logging.handlers import TimedRotatingFileHandler
import os
from aiogram import Bot, Dispatcher
from config import config
import handlers
from aiogram.methods import DeleteWebhook
from utils.storage import STORAGE


async def main() -> None:
    path = os.path.abspath(os.path.dirname(__file__))
    os.makedirs(path + "/log/", exist_ok=True)
    time_rotating_handler = TimedRotatingFileHandler(
        path + "/log/buchshot.log", when="midnight", interval=7, backupCount=30
    )
    logging.basicConfig(
        format="[%(asctime)s][%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
        datefmt="%d.%m.%Y %H:%M:%S",
    )
    bot = Bot(token=config.bot_token.get_secret_value())
    dp = Dispatcher(name='main', storage=STORAGE)
    # dp.message.middleware(Debug())

    dp.include_routers(
        handlers.rooms_manager.router,
        handlers.start.router,
        handlers.echo.router,
        handlers.base.router,  # Make sure it's the last handler
    )

    await bot(DeleteWebhook(drop_pending_updates=True))
    await dp.start_polling(
        bot,
        allowed_updates=dp.resolve_used_update_types(),
    )


if __name__ == "__main__":
    asyncio.run(main())
