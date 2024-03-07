import random
from typing import List
from .player import Player
from .models.ammo import AmmoType
import numpy as np


class Shotgun:
    """Class representing a shotgun."""

    def __init__(
        self,
        damage: int = 1,  # Determines the damage strength of the shotgun shot.
        mean_num_rounds: float = 5.0,  # Sets the average number of rounds in the magazine during reload.
        std_dev_num_rounds: float = 1.5,  # Defines the standard deviation in the distribution of rounds during reload.
        mean_live_rounds: float = 2.5,  # Specifies the average number of live rounds (causing damage) during reload.
        std_dev_live_rounds: float = 1.5,  # Sets the standard deviation in the distribution of live rounds during reload.
    ) -> None:
        """Initialize a Shotgun instance with optional parameters."""
        self.magazine: List[AmmoType] = []
        self.rounds: List[AmmoType] = []
        self.damage = damage
        self.mean_num_rounds = mean_num_rounds
        self.std_dev_num_rounds = std_dev_num_rounds
        self.mean_live_rounds = mean_live_rounds
        self.std_dev_live_rounds = std_dev_live_rounds

    def recharge(self) -> str:
        """Recharge the shotgun with random rounds."""
        total_rounds = int(
            np.random.normal(self.mean_num_rounds, self.std_dev_num_rounds)
        )
        total_rounds = max(2, total_rounds)

        min_live_rounds = int(0.3 * total_rounds)

        live_rounds_count = int(
            np.random.normal(self.mean_live_rounds, self.std_dev_live_rounds)
        )
        live_rounds_count = max(
            min_live_rounds, min(total_rounds - 1, live_rounds_count)
        )

        blank_rounds_count = total_rounds - live_rounds_count

        rounds = [AmmoType.LIVE] * live_rounds_count + [
            AmmoType.BLANK
        ] * blank_rounds_count

        random.shuffle(rounds)
        self.rounds = rounds
        res = [r.value for r in rounds]
        res.sort()
        return ", ".join(res)

    def shot(self, target: Player) -> str:
        """Perform a shot with the shotgun to."""
        if not self.rounds:
            return AmmoType.BLANK.value

        round = self.rounds.pop()
        if round == AmmoType.LIVE:
            target.take_damage(self.damage)
        self.damage = 1
        return round.value

    def need_recharge(self) -> bool:
        """Check if the shotgun needs to be recharged."""
        return not bool(self.rounds)

    def shaking(self) -> str:
        """Shake the shotgun and return the value of the shaken round."""
        return self.rounds.pop().value

    def inspect(self) -> str:
        """Inspect the last round in the shotgun."""
        if not self.rounds:
            return AmmoType.BLANK.value
        return self.rounds[-1].value

    def get_ammo_sorted(self) -> str:
        """Get a sorted string representation of the shotgun's rounds."""
        res = [r.value for r in self.rounds]
        res.sort()
        return ", ".join(res)
