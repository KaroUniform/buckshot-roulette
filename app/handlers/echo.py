from aiogram import Router, F
from aiogram.filters import Command
from aiogram.types import Message


router = Router()


@router.message(Command("echo"))
async def cmd_start(message: Message):
    await message.answer(
        text=str(message),
    )
