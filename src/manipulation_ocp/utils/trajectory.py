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


from dataclasses import dataclass


@dataclass(frozen=True)
class ResampledTrajectory:
    """
    Trajectory resampled to a fixed dt grid.

    time:
        Shape (num_nodes,)

    q:
        Shape (num_nodes, nq)

    v:
        Shape (num_nodes, nv)

    u:
        Optional, shape (num_nodes, nu)
    """

    time: np.ndarray
    q: np.ndarray
    v: np.ndarray
    u: np.ndarray | None

    dt: float
    original_tf: float
    resampled_tf: float


def build_fixed_dt_grid_from_tf(
    *,
    tf: float,
    dt: float,
    close_to_integer_tol: float = 1e-2,
) -> np.ndarray:
    """
    Build a fixed-dt time grid that does not exceed tf significantly.

    This is useful when OCP returns tf slightly different from an exact
    multiple of dt, for example:
        tf = 0.400032
        dt = 0.02

    In that case this returns:
        [0.00, 0.02, ..., 0.40]

    Parameters
    ----------
    tf:
        Original trajectory duration.

    dt:
        Desired fixed sampling time.

    close_to_integer_tol:
        Tolerance on tf / dt for snapping to the nearest integer number
        of intervals.
    """
    tf = float(tf)
    dt = float(dt)

    if tf <= 0.0:
        raise ValueError(f"tf must be > 0, got {tf}")

    if dt <= 0.0:
        raise ValueError(f"dt must be > 0, got {dt}")

    ratio = tf / dt
    nearest_intervals = int(round(ratio))

    if nearest_intervals <= 0:
        nearest_intervals = 1

    if abs(ratio - nearest_intervals) <= close_to_integer_tol:
        num_intervals = nearest_intervals
    else:
        # Avoid extrapolating beyond the OCP trajectory duration.
        num_intervals = int(np.floor(ratio))
        num_intervals = max(num_intervals, 1)

    return np.arange(num_intervals + 1, dtype=float) * dt


def interpolate_vector_trajectory(
    *,
    values: np.ndarray,
    source_time: np.ndarray,
    target_time: np.ndarray,
    name: str = "values",
) -> np.ndarray:
    """
    Linearly interpolate a vector trajectory.

    values:
        shape = (num_source_nodes, dim)

    source_time:
        shape = (num_source_nodes,)

    target_time:
        shape = (num_target_nodes,)
    """
    values_arr = np.asarray(values, dtype=float)
    source_time_arr = np.asarray(source_time, dtype=float).reshape(-1)
    target_time_arr = np.asarray(target_time, dtype=float).reshape(-1)

    if values_arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {values_arr.shape}")

    if values_arr.shape[0] != source_time_arr.size:
        raise ValueError(
            f"{name} rows must match source_time size. "
            f"Got {values_arr.shape[0]} and {source_time_arr.size}"
        )

    if source_time_arr[0] != 0.0:
        raise ValueError("source_time must start at 0.0")

    if np.any(np.diff(source_time_arr) <= 0.0):
        raise ValueError("source_time must be strictly increasing")

    if target_time_arr[0] != 0.0:
        raise ValueError("target_time must start at 0.0")

    if np.any(np.diff(target_time_arr) <= 0.0):
        raise ValueError("target_time must be strictly increasing")

    if target_time_arr[-1] > source_time_arr[-1] + 1e-9:
        raise ValueError(
            "target_time exceeds source_time duration. "
            f"target end={target_time_arr[-1]}, source end={source_time_arr[-1]}"
        )

    out = np.zeros((target_time_arr.size, values_arr.shape[1]), dtype=float)

    for j in range(values_arr.shape[1]):
        out[:, j] = np.interp(
            target_time_arr,
            source_time_arr,
            values_arr[:, j],
        )

    return out


def resample_qv_trajectory_to_dt(
    *,
    q_nodes: np.ndarray,
    v_nodes: np.ndarray,
    tf: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resample q and v trajectories to a fixed dt grid.

    Returns
    -------
    time:
        Shape (num_resampled_nodes,)

    q_ref:
        Shape (num_resampled_nodes, nq)

    v_ref:
        Shape (num_resampled_nodes, nv)
    """
    q_arr = np.asarray(q_nodes, dtype=float)
    v_arr = np.asarray(v_nodes, dtype=float)

    if q_arr.ndim != 2:
        raise ValueError(f"q_nodes must be 2D, got shape {q_arr.shape}")

    if v_arr.ndim != 2:
        raise ValueError(f"v_nodes must be 2D, got shape {v_arr.shape}")

    if q_arr.shape != v_arr.shape:
        raise ValueError(
            f"q_nodes and v_nodes must have same shape. "
            f"Got {q_arr.shape} and {v_arr.shape}"
        )

    source_time = np.linspace(0.0, float(tf), q_arr.shape[0])
    target_time = build_fixed_dt_grid_from_tf(tf=tf, dt=dt)

    q_ref = interpolate_vector_trajectory(
        values=q_arr,
        source_time=source_time,
        target_time=target_time,
        name="q_nodes",
    )

    v_ref = interpolate_vector_trajectory(
        values=v_arr,
        source_time=source_time,
        target_time=target_time,
        name="v_nodes",
    )

    return target_time, q_ref, v_ref


def resample_ocp_solution_to_dt(
    *,
    solution,
    dt: float,
    include_u: bool = True,
) -> ResampledTrajectory:
    """
    Resample an OCP solution to a fixed dt grid.

    This is the function planner/executor should call before passing
    reference trajectory to RL or MuJoCo executor.
    """
    time, q_ref, v_ref = resample_qv_trajectory_to_dt(
        q_nodes=solution.q_nodes,
        v_nodes=solution.v_nodes,
        tf=solution.tf,
        dt=dt,
    )

    u_ref = None

    if include_u:
        source_time = np.linspace(
            0.0,
            float(solution.tf),
            solution.U_nodes.shape[0],
        )

        u_ref = interpolate_vector_trajectory(
            values=solution.U_nodes,
            source_time=source_time,
            target_time=time,
            name="U_nodes",
        )

    return ResampledTrajectory(
        time=time,
        q=q_ref,
        v=v_ref,
        u=u_ref,
        dt=float(dt),
        original_tf=float(solution.tf),
        resampled_tf=float(time[-1]),
    )