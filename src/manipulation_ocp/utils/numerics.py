from __future__ import annotations

from typing import Sequence

import numpy as np


def as_vector(
    x: Sequence[float] | np.ndarray,
    *,
    size: int | None = None,
    name: str = "x",
) -> np.ndarray:
    """
    Convert input to a 1D float vector.

    Parameters
    ----------
    x:
        Input array-like.

    size:
        Expected vector size. If None, only flatten is applied.

    name:
        Name used in error messages.
    """
    arr = np.asarray(x, dtype=float).reshape(-1)

    if size is not None and arr.size != size:
        raise ValueError(
            f"{name} must have shape ({size},), got shape ({arr.size},)"
        )

    return arr


def clip_to_bounds(
    x: Sequence[float] | np.ndarray,
    lower: Sequence[float] | np.ndarray,
    upper: Sequence[float] | np.ndarray,
    *,
    finite_only: bool = True,
) -> np.ndarray:
    """
    Clip a vector to lower/upper bounds.

    If finite_only=True, entries with non-finite bounds are left unchanged.
    """
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    lower_arr = np.asarray(lower, dtype=float).reshape(x_arr.shape)
    upper_arr = np.asarray(upper, dtype=float).reshape(x_arr.shape)

    clipped = x_arr.copy()

    if finite_only:
        valid = (
            np.isfinite(lower_arr)
            & np.isfinite(upper_arr)
            & (lower_arr <= upper_arr)
        )
        clipped[valid] = np.clip(
            clipped[valid],
            lower_arr[valid],
            upper_arr[valid],
        )
    else:
        clipped = np.clip(clipped, lower_arr, upper_arr)

    return clipped


def finite_difference_first_order(
    y: Sequence[Sequence[float]] | np.ndarray,
    dt: float,
    *,
    axis: int = 0,
) -> np.ndarray:
    """
    First-order finite difference with the same shape as input.

    Default convention:
        y shape = (num_nodes, dim)
        axis = 0

    Boundary:
        forward difference at the first node
        backward difference at the last node

    Interior:
        central difference
    """
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0, got {dt}")

    arr = np.asarray(y, dtype=float)

    if arr.ndim != 2:
        raise ValueError(f"y must be 2D, got ndim={arr.ndim}")

    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")

    if axis == 1:
        arr_work = arr.T
    else:
        arr_work = arr

    num_nodes, _ = arr_work.shape
    dy = np.zeros_like(arr_work)

    if num_nodes == 1:
        out = dy
    elif num_nodes == 2:
        slope = (arr_work[1] - arr_work[0]) / dt
        dy[0] = slope
        dy[1] = slope
        out = dy
    else:
        dy[0] = (arr_work[1] - arr_work[0]) / dt
        dy[-1] = (arr_work[-1] - arr_work[-2]) / dt
        dy[1:-1] = (arr_work[2:] - arr_work[:-2]) / (2.0 * dt)
        out = dy

    if axis == 1:
        return out.T

    return out


def linear_interpolation(
    start: Sequence[float] | np.ndarray,
    goal: Sequence[float] | np.ndarray,
    num_nodes: int,
) -> np.ndarray:
    """
    Build a linear trajectory from start to goal.

    Returns
    -------
    traj:
        shape = (num_nodes, dim)
    """
    if num_nodes < 2:
        raise ValueError(f"num_nodes must be >= 2, got {num_nodes}")

    start_arr = np.asarray(start, dtype=float).reshape(-1)
    goal_arr = np.asarray(goal, dtype=float).reshape(start_arr.shape)

    alpha = np.linspace(0.0, 1.0, num_nodes)

    return (
        (1.0 - alpha[:, None]) * start_arr[None, :]
        + alpha[:, None] * goal_arr[None, :]
    )