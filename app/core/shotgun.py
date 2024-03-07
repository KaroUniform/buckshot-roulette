import random
from .player import Player
from .models.ammo import AmmoType
import numpy as np


class Shotgun:

    rounds: list[AmmoType]
    damage = 1
    mean_num_rounds = 5
    std_dev_num_rounds = 1.5
    mean_live_rounds = 2.5
    std_dev_live_rounds = 1.5

    def __init__(self) -> None:
        self.magazine = []

    def recharge(self):

        num_rounds = int(
            np.random.normal(self.mean_num_rounds, self.std_dev_num_rounds)
        )
        num_rounds = max(2, min(8, num_rounds))
        live_rounds_count = int(
            np.random.normal(self.mean_live_rounds, self.std_dev_live_rounds)
        )
        live_rounds_count = max(
            1, min(num_rounds - 1, live_rounds_count)
        )  # Ограничиваем значение от 1 до num_rounds - 1

        # Генерация общего списка патронов
        rounds = [AmmoType.LIVE] * live_rounds_count + [AmmoType.BLANK] * (
            num_rounds - live_rounds_count
        )
        random.shuffle(rounds)
        self.rounds = rounds
        res = [r.value for r in rounds]
        res.sort()
        return ", ".join(res)

    def shot(self, player: Player):
        if not self.rounds:
            return AmmoType.BLANK.value

        round = self.rounds.pop()
        if round == AmmoType.LIVE:
            player.take_damage(self.damage)
        self.damage = 1
        return round.value

    def need_recharge(self):
        return not bool(self.rounds)

    def shaking(self):
        return self.rounds.pop().value

    def inspect(self):
        if not self.rounds:
            return AmmoType.BLANK
        return self.rounds[-1].value

    def get_ammo_sorted(self):
        res = [r.value for r in self.rounds]
        res.sort()
        return ", ".join(res)
