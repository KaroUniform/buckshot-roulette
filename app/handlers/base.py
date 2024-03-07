from aiogram import Router, F
from aiogram.types import Message
import logging

router = Router()
logger = logging.getLogger()


@router.message()
async def cmd_start(message: Message):
    logger.info(
        f"chat: {message.chat.username} | user: {message.from_user.username} | mess: {message.text}"
    )
    return
