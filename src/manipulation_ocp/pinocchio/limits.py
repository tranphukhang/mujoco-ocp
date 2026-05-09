from __future__ import annotations

from typing import Sequence

import numpy as np
import pinocchio as pin

from manipulation_ocp.utils.numerics import (
    as_vector,
    clip_to_bounds,
)


def get_position_limits(model: pin.Model) -> tuple[np.ndarray, np.ndarray]:
    """Return Pinocchio position limits."""
    lower = np.asarray(model.lowerPositionLimit, dtype=float).reshape(model.nq)
    upper = np.asarray(model.upperPositionLimit, dtype=float).reshape(model.nq)
    return lower, upper


def get_velocity_limits(
    model: pin.Model,
    *,
    default_if_invalid: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return velocity lower/upper limits.

    Pinocchio may provide inf/nan/zero velocity limits depending on the model.
    Invalid values are replaced by +/- default_if_invalid.
    """
    if default_if_invalid <= 0.0:
        raise ValueError(
            f"default_if_invalid must be > 0, got {default_if_invalid}"
        )

    raw = np.asarray(model.velocityLimit, dtype=float).reshape(model.nv)
    vmax = raw.copy()

    invalid = (~np.isfinite(vmax)) | (vmax <= 0.0)
    vmax[invalid] = default_if_invalid

    return -vmax, vmax


def get_effort_limits(model: pin.Model) -> tuple[np.ndarray, np.ndarray]:
    """
    Return effort/torque lower/upper limits.

    Pinocchio stores positive effort limits:
        |tau_i| <= effortLimit_i
    """
    umax = np.asarray(model.effortLimit, dtype=float).reshape(model.nv)

    lower = -umax.copy()
    upper = umax.copy()

    return lower, upper


def clip_q_to_limits(
    model: pin.Model,
    q: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Clip q to finite position limits."""
    q_vec = as_vector(q, size=model.nq, name="q")
    q_lower, q_upper = get_position_limits(model)

    return clip_to_bounds(q_vec, q_lower, q_upper, finite_only=True)


def clip_v_to_limits(
    model: pin.Model,
    v: Sequence[float] | np.ndarray,
    *,
    default_if_invalid: float = 10.0,
) -> np.ndarray:
    """Clip v to effective velocity limits."""
    v_vec = as_vector(v, size=model.nv, name="v")
    v_lower, v_upper = get_velocity_limits(
        model,
        default_if_invalid=default_if_invalid,
    )

    return clip_to_bounds(v_vec, v_lower, v_upper, finite_only=True)


def clip_u_to_limits(
    model: pin.Model,
    u: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Clip torque input to effort limits."""
    u_vec = as_vector(u, size=model.nv, name="u")
    u_lower, u_upper = get_effort_limits(model)

    return clip_to_bounds(u_vec, u_lower, u_upper, finite_only=True)


def clip_trajectory_v_to_limits(
    model: pin.Model,
    v_traj: np.ndarray,
    *,
    default_if_invalid: float = 10.0,
) -> np.ndarray:
    """
    Clip velocity trajectory.

    Expected shape:
        v_traj = (num_nodes, nv)
    """
    arr = np.asarray(v_traj, dtype=float)

    if arr.ndim != 2 or arr.shape[1] != model.nv:
        raise ValueError(
            f"v_traj must have shape (num_nodes, {model.nv}), "
            f"got {arr.shape}"
        )

    clipped = arr.copy()

    for k in range(clipped.shape[0]):
        clipped[k] = clip_v_to_limits(
            model,
            clipped[k],
            default_if_invalid=default_if_invalid,
        )

    return clipped


def clip_trajectory_u_to_limits(
    model: pin.Model,
    u_traj: np.ndarray,
) -> np.ndarray:
    """
    Clip torque trajectory.

    Expected shape:
        u_traj = (num_nodes, nv)
    """
    arr = np.asarray(u_traj, dtype=float)

    if arr.ndim != 2 or arr.shape[1] != model.nv:
        raise ValueError(
            f"u_traj must have shape (num_nodes, {model.nv}), "
            f"got {arr.shape}"
        )

    clipped = arr.copy()

    for k in range(clipped.shape[0]):
        clipped[k] = clip_u_to_limits(model, clipped[k])

    return clipped