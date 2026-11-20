from __future__ import annotations

import random
from typing import Literal


class SensorSimulator:
    """Random-walk generator for AI4I-shaped readings.

    Most ticks stay near a healthy operating point. Every 20th cycle includes a
    short high-wear, high-torque excursion so the demo stream can show
    recommendation changes.
    """

    def __init__(
        self,
        seed: int = 42,
        zscore_baseline: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self._rng = random.Random(seed)
        self.zscore_baseline = zscore_baseline
        self.tick = 0
        self.air = 300.0
        self.process = 310.5
        self.speed = 1600.0
        self.torque = 38.0
        self.wear = 40.0

    def next_reading(
        self,
        mode: Literal["normal", "abnormal", "mixed"] = "mixed",
    ) -> dict[str, float | str | bool]:
        self.tick += 1
        self.air = _walk(self._rng, self.air, 0.12, 297.0, 303.0)
        self.process = _walk(self._rng, self.process, 0.12, 308.0, 314.0)
        self.speed = _walk(self._rng, self.speed, 6.0, 1450.0, 1900.0)
        self.torque = _walk(self._rng, self.torque, 0.25, 28.0, 48.0)
        self.wear += self._rng.uniform(0.4, 1.6)
        if self.wear >= 190.0:
            self.wear = 35.0

        abnormal = (
            mode == "abnormal"
            or (mode == "mixed" and 14 <= (self.tick % 20) <= 17)
        )
        if abnormal:
            torque_baseline = (
                self.zscore_baseline.get("Torque [Nm]")
                if self.zscore_baseline
                else None
            )
            if torque_baseline:
                mean = float(torque_baseline["mean"])
                std = float(torque_baseline["std"])
                if std > 0.0:
                    self.torque = min(300.0, max(0.0, mean + 4.0 * std))
                else:
                    self.torque = min(300.0, max(0.0, mean + 1.0))
            else:
                self.torque = min(68.0, self.torque + 10.0)
            self.speed = max(1180.0, self.speed - 180.0)
            self.wear = min(230.0, self.wear + 25.0)
            self.process = max(self.air + 6.5, self.process - 1.2)

        if self.process <= self.air:
            self.process = self.air + 8.8

        product_type = ("L", "M", "H")[self.tick % 3]
        return {
            "simulated": True,
            "simulation_mode": "abnormal" if abnormal else "normal",
            "product_type": product_type,
            "air_temperature": round(self.air, 2),
            "process_temperature": round(self.process, 2),
            "rotational_speed": round(self.speed, 2),
            "torque": round(self.torque, 2),
            "tool_wear": round(self.wear, 2),
        }


def _walk(
    rng: random.Random,
    value: float,
    step: float,
    lower: float,
    upper: float,
) -> float:
    updated = value + rng.uniform(-step, step)
    return min(upper, max(lower, updated))
