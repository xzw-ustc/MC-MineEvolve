"""Progress-score and stagnation detection (paper Eq. 2 and Eq. 3).

Eq. (2):
    p_i = λ_x * Var(x_{i,1:T}) / (ε_x + Var(x_{i,1:T}))
        + λ_v * ||Δv_i||_1 / (ε_v + ||Δv_i||_1)
        + λ_g * p^goal_i

Eq. (3):
    ℓ_i = I[p_i < ε_p ∧ y_i = 0]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class ProgressWeights:
    lambda_x: float = 0.4    # spatial (coord variance) weight
    lambda_v: float = 0.4    # inventory L1 weight
    lambda_g: float = 0.2    # task-specific goal progress weight
    eps_x: float = 1.0
    eps_v: float = 1.0
    eps_p: float = 0.10      # stagnation threshold


def coord_variance(coords: Sequence[Sequence[float]]) -> float:
    """Sum of per-axis variances across the trajectory window."""

    if not coords:
        return 0.0
    n = len(coords)
    if n < 2:
        return 0.0
    dim = max((len(c) for c in coords), default=0)
    if dim == 0:
        return 0.0
    total = 0.0
    for axis in range(dim):
        vals = [float(c[axis]) for c in coords if len(c) > axis]
        if len(vals) < 2:
            continue
        mean = sum(vals) / len(vals)
        total += sum((v - mean) ** 2 for v in vals) / len(vals)
    return total


def inventory_l1(delta_v: Mapping[str, int] | Iterable[int]) -> float:
    """L1 norm of the inventory delta dict or iterable."""

    if isinstance(delta_v, Mapping):
        return float(sum(abs(int(x)) for x in delta_v.values()))
    return float(sum(abs(int(x)) for x in delta_v))


def progress_score(
    coords: Sequence[Sequence[float]],
    delta_v: Mapping[str, int],
    p_goal: float = 0.0,
    weights: ProgressWeights | None = None,
) -> float:
    """Compute p_i in [0, 1] (paper Eq. 2)."""

    w = weights or ProgressWeights()
    var_x = coord_variance(coords)
    inv_l1 = inventory_l1(delta_v)
    spatial_term = var_x / (w.eps_x + var_x) if (var_x + w.eps_x) > 0 else 0.0
    inventory_term = inv_l1 / (w.eps_v + inv_l1) if (inv_l1 + w.eps_v) > 0 else 0.0
    goal_term = max(0.0, min(1.0, float(p_goal)))
    return float(w.lambda_x * spatial_term + w.lambda_v * inventory_term + w.lambda_g * goal_term)


def detect_stagnation(
    progress: float,
    success: int,
    weights: ProgressWeights | None = None,
) -> bool:
    """Eq. (3): low-progress AND not successful."""

    w = weights or ProgressWeights()
    return bool(progress < w.eps_p and int(success) == 0)
