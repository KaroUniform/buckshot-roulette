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
    logger = logging.getLogger(__name__)

    # Optional HTTPS proxy for hosts that can't reach api.telegram.org
    # directly. compose.yml passes `BOT_PROXY` through as `HTTPS_PROXY`;
    # if it's unset we fall back to aiogram's default aiohttp transport.
    # httpx is used (see utils/httpx_session) because aiohttp's handling
    # of TLS-tunnel proxies is inconsistent.
    proxy_url = os.getenv("HTTPS_PROXY") or None
    session = None
    if proxy_url:
        from utils.httpx_session import HttpxSession
        session = HttpxSession(proxy=proxy_url)
        # Strip credentials before logging — the proxy URL typically
        # carries `user:password@host:port`.
        host_tail = proxy_url.split("@")[-1] if "@" in proxy_url else proxy_url
        logger.info("HTTPS proxy enabled: %s", host_tail)

    bot = Bot(token=config.bot_token.get_secret_value(), session=session)
    dp = Dispatcher(name='main', storage=STORAGE)
    # dp.message.middleware(Debug())

    dp.include_routers(
        handlers.ai_game.router,     # AI-mode messages — state-filtered
        handlers.ai_stats.router,    # /stats — state-agnostic, public read
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
