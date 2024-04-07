import random
from core.shotgun import Shotgun
from core.models.turn_model import Dealer
from core.player import Player
from core.models.items import ItemType

def use_item(
        item: ItemType,
        player: Player,
        target: Player,
        dealer: Dealer,
        shotgun: Shotgun,
    ):
        """
        Processes a player's use of a specific item during their turn.

        Args:
        - item (ItemType): The item to use. Valid items include:
            - "handsaw": Use a handsaw to double damage.
            - "beer": Consume beer to throw away the last round from the shotgun.
            - "smoke": Use smoke to increase player's health.
            - "handcuff": Use handcuffs to immobilize the opponent temporarily.
            - "magnifying_glass": Inspect the shotgun with a magnifying glass.

        Returns:
        - Tuple[Union[str, Tuple[str, str]], str]:
            - If the item usage is successful, a tuple containing:
                - The result of the item usage for the active player.
                - The result of the item usage for the passive player.
            - If the item usage fails, a string indicating the failure reason.
        """

        if player.get_number_of_item(item) == 0:
            return (f"You don't have {item}", None)

        if not dealer.use_items(item):
            return "❌Only 1 item per turn", None

        match item:
            case ItemType.HANDSAW:
                player.inventory.handsaw -= 1
                shotgun.damage *= 2
                return (
                    "🧨Careful, the weapon now deals x2 damage",
                    f"{player.data.name} uses🪚",
                )
            case ItemType.BEER:
                player.inventory.beer -= 1
                round = shotgun.shaking()
                return (
                    f"⤴️ The {round} flew out of the shotgun",
                    f"{player.data.name} uses 🍺. The ⤴️{round} flew out of the shotgun ",
                )
            case ItemType.SMOKE:
                player.inventory.smoke -= 1
                player.smoke()
                return (
                    f"🚬You now have {player.data.hp} hp",
                    f"{player.data.name} uses 🚬",
                )
            case ItemType.HANDCUFF:
                if target.tied > 0:
                    return ("You cannot re-link a linked player", None)
                player.inventory.handcuff -= 1
                target.tied = 3
                dealer.extend_turn()
                return (
                    f"🔗{target.data.name} is tied for 1 turn",
                    f"🔗 {player.data.name} has tied you up",
                )
            case ItemType.GLASS:
                player.inventory.magnifying_glass -= 1
                return (
                    f"🔍 You see the {str(shotgun.inspect())} inside",
                    f"{player.data.name} uses 🔍. Very interesting...",
                )
            case ItemType.PHONE:
                player.inventory.phone -= 1
                words = {
                    1: 'FIRST',
                    2: 'SECOND',
                    3: 'THIRD',
                    4: 'FOURTH',
                    5: 'FIFTH',
                    6: 'SIXTH',
                    7: 'SEVENTH',
                    8: 'EIGHTH'
                }
                pos = random.randint(0, len(shotgun.rounds)-1)
                return (
                    f"📞 ...The {words[pos+1]} bullet\nis {'🫧BLANK🫧' if shotgun.rounds[::-1][pos].value == '🫧' else '💥LIVE💥'}...",
                    f"📞{player.data.name} calling an unknown number...",
                )
            case ItemType.PILLS:
                player.inventory.pills -= 1
                if random.randint(1, 100) <= 40:
                    player.smoke(2)
                    return (
                        f"💊You're trying spoiled pills. They're working. You are being healed for 2⚡️",
                        f"💊{player.data.name} tries the spoiled pills. Lucky. Restored 2⚡️.",
                    )
                else:
                    player.take_damage(1)
                    return (
                        f"💊🤢You're trying spoiled pills. It was a bad pack. You lose 1⚡️",
                        f"💊🤢{player.data.name} tries the spoiled pills. It was a bad pack. Lose 1⚡️",
                    )
            case ItemType.INVERTER:
                player.inventory.inverter -= 1
                shotgun.invert()
                return (
                        f"🔀You start the inverter. ALL the live bullets became blank bullets. ALL the blank have become live.",
                        f"🔀{player.data.name} start the inverter. ALL the live bullets became blank bullets. ALL the blank have become live.",
                    )
            case _:
                raise ValueError(f"An unexpected item was received: {item}")