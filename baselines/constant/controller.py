"""
baselines/constant/controller.py
------------------------------------
Baseline 3: constant laser power for the whole episode — no state feedback
whatsoever. Swept over the training grid {100, 110, ..., 400} W (31 values,
matching multilayer_random_v2.m's LP_values). The headline "Constant Policy"
number is the MEAN return across this entire sweep (a genuinely naive
baseline by construction — not the best-in-hindsight constant); the
per-constant breakdown is also reported since it's an informative sanity
check on the reward landscape's shape.
"""

import numpy as np

CONSTANT_GRID = np.arange(100.0, 401.0, 10.0, dtype=np.float32)   # 31 values


class ConstantController:
    def __init__(self, power_W: float):
        self.power_W = float(power_W)

    def act(self, ctx) -> float:
        return self.power_W


def sweep_constant_controllers(grid: np.ndarray = CONSTANT_GRID):
    """Yields (power_W, ConstantController) for every value in the grid."""
    for w in grid:
        yield float(w), ConstantController(float(w))
