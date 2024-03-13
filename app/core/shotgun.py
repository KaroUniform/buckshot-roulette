import random
from typing import List

import random
from .player import Player
from .models.ammo import AmmoType


class Shotgun:
    """Class representing a shotgun."""

    def __init__(
        self,
        damage: int = 1,
        mean_num_rounds: float = 5.0,
        std_dev_num_rounds: float = 1.5,
        mean_live_ratio: float = 0.5,
        std_dev_live_ratio: float = 0.2,
    ) -> None:
        """Initialize a Shotgun instance with optional parameters."""
        self.magazine: List[AmmoType] = []
        self.rounds: List[AmmoType] = []
        self.damage = damage
        self.mean_num_rounds = mean_num_rounds
        self.std_dev_num_rounds = std_dev_num_rounds
        self.mean_live_ratio = mean_live_ratio
        self.std_dev_live_ratio = std_dev_live_ratio

    def recharge(self) -> str:
        """Recharge the shotgun with random rounds."""
        rounds = self.generate_normal_distribution()

        live = rounds / 2 + rounds / 2 * random.uniform(-0.2, 0.3)
        blank = rounds - live
        live, blank = max(1, round(live)), max(1, round(blank))
        rounds = [AmmoType.LIVE] * live + [AmmoType.BLANK] * blank

        random.shuffle(rounds)
        self.rounds = rounds
        res = [r.value for r in rounds]
        return ", ".join(res)

    def generate_normal_distribution(
        self,
        mean: float = 5,
        std_deviation: float = 1,
        lower_bound: int = 2,
        upper_bound: int = 7,
    ) -> int:
        """
        Generates a random number from a normal distribution with the specified mean and standard deviation.

        Args:
        - mean (float): The mean of the normal distribution (default is 4.0).
        - std_deviation (float): The standard deviation of the normal distribution (default is 2.0).
        - lower_bound (int): The lower bound for the generated number (default is 2).
        - upper_bound (int): The upper bound for the generated number (default is 7).

        Returns:
        int: A rounded and bounded random number between the specified lower and upper bounds.
        """
        generated_number = random.normalvariate(mean, std_deviation)
        generated_number = max(lower_bound, min(upper_bound, round(generated_number)))
        return generated_number

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
