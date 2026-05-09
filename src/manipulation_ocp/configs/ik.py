from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DualEEIKConfig:
    """
    Lightweight dual-end-effector IK parameters.

    Defaults follow the previous Test-OCP dual-EE IK setup.
    """

    max_iters: int = 40
    tol: float = 1e-3
    step_size: float = 0.35
    damping: float = 1e-4
    clip_to_limits: bool = True
    patience: int = 6
    min_progress_ratio: float = 0.05
    min_progress_abs: float = 1e-4
    fallback_to_seed_if_poor_progress: bool = True
    weight_left: float = 1.0
    weight_right: float = 1.0
    verbose: bool = False