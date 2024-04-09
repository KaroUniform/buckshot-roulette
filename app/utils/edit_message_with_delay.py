import asyncio
from aiogram.types import Message

async def edit_message(msg1: Message, new_text="❔", sleep=4):
    if(sleep>=100):
        return
    await asyncio.sleep(sleep)
    await msg1.edit_text(new_text)
