from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InitialGuessConfig:
    """
    Initial guess generation parameters.

    Convention:
        n_intervals = 50
        num_nodes = 51
        dt = 0.04
        total_time = 2.0
    """

    n_intervals: int = 50
    dt: float = 0.04
    default_v_limit_if_invalid: float = 10.0
    clip_v_to_limits: bool = True
    clip_u_to_limits: bool = True

    @property
    def num_nodes(self) -> int:
        return self.n_intervals + 1

    @property
    def total_time(self) -> float:
        return self.n_intervals * self.dt