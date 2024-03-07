from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message


router = Router()


@router.message(Command("start"))
@router.message(Command("help"))
async def new_round(message: Message):
    text = """
Welcome to Buckshot Roulette! Here's a quick guide to the game:

1. *Create/Join a Room:*
   - To start, create a room with `/join` command.
   - If you have a code, simply use `/join 123456` to join an existing room.

2. *Initiate the Game:*
   - The game kicks off with a coin toss to determine who goes first.
   - The objective is to eliminate your opponent before they eliminate you!

3. *Health Points and ammo:*
   - Both players start with an equal amount of ⚡️HP⚡️, randomly generated.
   - At the beginning of each round, a random mix of 💥live💥 and 🫧blank🫧 rounds is loaded into the shotgun.

4. *Gameplay:*
   - Players take turns making moves.
   - You can either 🔼shoot your opponent🔼, passing the turn to them, or 🔽shoot yourself🔽, gaining the next move if the shot is 🫧blank🫧.

5. *Items for Strategy:*
   - Use items strategically before shooting:
      - *🔗Handcuffs:* Force your opponent to skip their next move.
      - *🚬Cigarette:* Restore one HP to yourself.
      - *🍺Beer:* Remove one loaded cartridge from the shotgun.
      - *🪚Saw:* Double the shotgun's damage for one turn.
      - *🔍Magnifying Glass:* Identify one currently loaded cartridge in the shotgun.

Keep a keen eye, plan your moves wisely, and aim for victory in Buckshot Roulette!
   """.strip()

    await message.answer(text, parse_mode="Markdown")
