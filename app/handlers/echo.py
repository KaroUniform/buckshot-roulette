from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message


router = Router()


@router.message(Command("echo"))
async def cmd_start(message: Message):
    await message.answer(
        text=str(message),
    )


@router.message(lambda message: message.sticker is not None)
async def show_sticker_id(message: Message):
    await message.answer(
        text=(
            "Sticker file_id:\n"
            f"`{message.sticker.file_id}`\n\n"
            "Вставь это значение в `WIN_STICKER_ID` в `app/handlers/rooms_manager.py`."
        ),
        parse_mode="Markdown",
    )
