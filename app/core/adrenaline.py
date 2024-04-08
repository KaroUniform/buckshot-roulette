from core.shotgun import Shotgun
from core.models.turn_model import Dealer
from core.player import Player
from core.models.items import ItemType

def adrenaline(func):
    def wrapper(
        item: ItemType,
        player: Player,
        target: Player,
        dealer: Dealer,
        shotgun: Shotgun,
    ):
        # Проверяем, было ли поле adrenaline у player равно True перед вызовом функции
        if player.adrenaline:
            result = func(item, player, target, dealer, shotgun)  # Вызов оригинальной функции
            player.stop_adrenaline_effect()  # Вызов метода stop_adrenaline после выполнения функции
            return result
        else:
            return func(item, player, target, dealer, shotgun)  # Просто вызов функции, если adrenaline не был True
    return wrapper