from __future__ import annotations

from typing import Literal, Sequence

import numpy as np


ArmSide = Literal["left", "right"]


LEFT_ARM_SLICE = slice(3, 10)
RIGHT_ARM_SLICE = slice(10, 17)


def minimum_jerk_profile(s: np.ndarray) -> np.ndarray:
    """
    Minimum-jerk profile.

    s should be normalized time in [0, 1].

    profile:
        10 s^3 - 15 s^4 + 6 s^5
    """
    s = np.asarray(s, dtype=float)

    return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5


def minimum_jerk_profile_derivative(s: np.ndarray) -> np.ndarray:
    """
    Derivative of minimum-jerk profile with respect to normalized time s.

    d/ds:
        30 s^2 - 60 s^3 + 30 s^4
    """
    s = np.asarray(s, dtype=float)

    return 30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4


def build_time_grid(
    *,
    duration: float,
    dt: float,
    include_endpoint: bool = True,
) -> np.ndarray:
    """
    Build a time grid from 0 to duration.

    The returned grid always starts at 0.
    If include_endpoint=True, the last sample is exactly duration.
    """
    if duration <= 0.0:
        raise ValueError(f"duration must be > 0, got {duration}")

    if dt <= 0.0:
        raise ValueError(f"dt must be > 0, got {dt}")

    if include_endpoint:
        num_intervals = int(np.ceil(duration / dt))
        return np.linspace(0.0, duration, num_intervals + 1)

    num_steps = int(np.ceil(duration / dt))
    return np.arange(num_steps, dtype=float) * dt


def build_minimum_jerk_joint_trajectory(
    *,
    q_start: Sequence[float] | np.ndarray,
    q_goal: Sequence[float] | np.ndarray,
    duration: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a minimum-jerk joint trajectory.

    Parameters
    ----------
    q_start, q_goal:
        Shape (dim,).

    duration:
        Trajectory duration in seconds.

    dt:
        Sampling time.

    Returns
    -------
    q_traj:
        Shape (num_nodes, dim)

    v_traj:
        Shape (num_nodes, dim)
    """
    q0 = np.asarray(q_start, dtype=float).reshape(-1)
    q1 = np.asarray(q_goal, dtype=float).reshape(-1)

    if q0.shape != q1.shape:
        raise ValueError(
            f"q_start and q_goal must have same shape, got {q0.shape} and {q1.shape}"
        )

    time = build_time_grid(duration=duration, dt=dt)
    s = time / duration

    alpha = minimum_jerk_profile(s)
    alpha_dot_s = minimum_jerk_profile_derivative(s)

    dq = q1 - q0

    q_traj = q0[None, :] + alpha[:, None] * dq[None, :]

    # d alpha / dt = d alpha / ds * ds / dt = alpha_dot_s / duration
    v_traj = (alpha_dot_s[:, None] / duration) * dq[None, :]

    return q_traj, v_traj


def _arm_slice(side: ArmSide) -> slice:
    if side == "left":
        return LEFT_ARM_SLICE

    if side == "right":
        return RIGHT_ARM_SLICE

    raise ValueError(f"side must be 'left' or 'right', got {side!r}")


def build_arm_return_trajectory(
    *,
    q_current: Sequence[float] | np.ndarray,
    q_initial: Sequence[float] | np.ndarray,
    side: ArmSide,
    duration: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a joint-space trajectory that returns only one arm to its initial q.

    Joint convention:
        left arm  = q[3:10]
        right arm = q[10:17]

    Other joints are held fixed at q_current.

    Parameters
    ----------
    q_current:
        Current OCP q, shape (17,).

    q_initial:
        Initial/home OCP q, shape (17,).

    side:
        "left" or "right".

    duration:
        Return duration.

    dt:
        Sampling time.

    Returns
    -------
    q_traj:
        Shape (num_nodes, 17)

    v_traj:
        Shape (num_nodes, 17)
    """
    q_cur = np.asarray(q_current, dtype=float).reshape(-1)
    q_home = np.asarray(q_initial, dtype=float).reshape(-1)

    if q_cur.size != 17:
        raise ValueError(f"q_current must have size 17, got {q_cur.size}")

    if q_home.size != 17:
        raise ValueError(f"q_initial must have size 17, got {q_home.size}")

    idx = _arm_slice(side)

    q_arm_traj, v_arm_traj = build_minimum_jerk_joint_trajectory(
        q_start=q_cur[idx],
        q_goal=q_home[idx],
        duration=duration,
        dt=dt,
    )

    num_nodes = q_arm_traj.shape[0]

    q_traj = np.tile(q_cur.reshape(1, 17), (num_nodes, 1))
    v_traj = np.zeros((num_nodes, 17), dtype=float)

    q_traj[:, idx] = q_arm_traj
    v_traj[:, idx] = v_arm_traj

    return q_traj, v_traj


def build_gripper_command_trajectory(
    *,
    command_start: float,
    command_goal: float,
    duration: float,
    dt: float,
) -> np.ndarray:
    """
    Build a smooth normalized gripper command trajectory.

    Command convention:
        0.0 = open
        1.0 = close

    Returns
    -------
    command_traj:
        Shape (num_nodes,)
    """
    c0 = float(command_start)
    c1 = float(command_goal)

    if c0 < 0.0 or c0 > 1.0:
        raise ValueError(f"command_start must be in [0, 1], got {c0}")

    if c1 < 0.0 or c1 > 1.0:
        raise ValueError(f"command_goal must be in [0, 1], got {c1}")

    time = build_time_grid(duration=duration, dt=dt)
    s = time / duration

    alpha = minimum_jerk_profile(s)

    command_traj = c0 + alpha * (c1 - c0)

    return np.clip(command_traj, 0.0, 1.0)