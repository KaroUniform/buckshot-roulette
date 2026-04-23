import logging

from aiogram import Bot


logger = logging.getLogger(__name__)

WIN_STICKER_ID = "CAACAgIAAxkBAAEBuf1p6kaACU2hkSZ0LlN9J_XaRVPWOwACSTEAAoTfaEtmyI9fpwx3RzsE"


async def send_winner_sticker(bot: Bot, chat_id: int):
    if not WIN_STICKER_ID:
        return

    try:
        await bot.send_sticker(chat_id, sticker=WIN_STICKER_ID)
    except Exception:
        logger.exception("Failed to send winner sticker to chat_id=%s", chat_id)
