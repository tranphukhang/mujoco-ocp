from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np


ArmSide = Literal["left", "right"]

LEFT_ARM_SLICE = slice(3, 10)
RIGHT_ARM_SLICE = slice(10, 17)

OCP_Q_DIM = 17


def _as_1d_float_array(
    values: Sequence[float] | np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """
    Convert values to a finite 1D float array.
    """
    arr = np.asarray(values, dtype=float).reshape(-1)

    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values, got {arr}")

    return arr


def _as_2d_float_array(
    values: Sequence[Sequence[float]] | np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """
    Convert values to a finite 2D float array.
    """
    arr = np.asarray(values, dtype=float)

    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape}")

    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values")

    return arr


def _validate_positive_scalar(
    value: float,
    *,
    name: str,
) -> float:
    """
    Validate value is finite and strictly positive.
    """
    value_float = float(value)

    if not np.isfinite(value_float):
        raise ValueError(f"{name} must be finite, got {value}")

    if value_float <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value_float}")

    return value_float


def _validate_nonnegative_tolerance(
    value: float,
    *,
    name: str,
) -> float:
    """
    Validate tolerance is finite and non-negative.
    """
    value_float = float(value)

    if not np.isfinite(value_float):
        raise ValueError(f"{name} must be finite, got {value}")

    if value_float < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value_float}")

    return value_float


def minimum_jerk_profile(s: np.ndarray) -> np.ndarray:
    """
    Minimum-jerk position profile.

    s:
        Normalized time in [0, 1].

    profile:
        10 s^3 - 15 s^4 + 6 s^5
    """
    s_arr = np.asarray(s, dtype=float)

    return 10.0 * s_arr**3 - 15.0 * s_arr**4 + 6.0 * s_arr**5


def minimum_jerk_profile_derivative(s: np.ndarray) -> np.ndarray:
    """
    Derivative of minimum-jerk profile with respect to normalized time s.

    d/ds:
        30 s^2 - 60 s^3 + 30 s^4
    """
    s_arr = np.asarray(s, dtype=float)

    return 30.0 * s_arr**2 - 60.0 * s_arr**3 + 30.0 * s_arr**4


def build_time_grid(
    *,
    duration: float,
    dt: float,
    include_endpoint: bool = True,
) -> np.ndarray:
    """
    Build a time grid from 0 to duration.

    If include_endpoint=True:
        The last sample is exactly duration.
        If duration is not an exact multiple of dt, the final interval can be
        shorter than dt.

    If include_endpoint=False:
        The grid starts at 0 and uses fixed dt samples that stay before
        duration.
    """
    duration_float = _validate_positive_scalar(duration, name="duration")
    dt_float = _validate_positive_scalar(dt, name="dt")

    if include_endpoint:
        time = np.arange(0.0, duration_float + 1e-12, dt_float)

        if time.size == 0:
            time = np.array([0.0], dtype=float)

        if time[-1] < duration_float - 1e-12:
            time = np.append(time, duration_float)
        else:
            time[-1] = duration_float

        return time

    num_steps = int(np.ceil(duration_float / dt_float))
    num_steps = max(num_steps, 1)

    return np.arange(num_steps, dtype=float) * dt_float


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
        Shape (num_nodes, dim).

    v_traj:
        Shape (num_nodes, dim).
    """
    duration_float = _validate_positive_scalar(duration, name="duration")
    dt_float = _validate_positive_scalar(dt, name="dt")

    q0 = _as_1d_float_array(q_start, name="q_start")
    q1 = _as_1d_float_array(q_goal, name="q_goal")

    if q0.shape != q1.shape:
        raise ValueError(
            f"q_start and q_goal must have same shape, "
            f"got {q0.shape} and {q1.shape}"
        )

    time = build_time_grid(
        duration=duration_float,
        dt=dt_float,
        include_endpoint=True,
    )

    s = time / duration_float

    alpha = minimum_jerk_profile(s)
    alpha_dot_s = minimum_jerk_profile_derivative(s)

    dq = q1 - q0

    q_traj = q0[None, :] + alpha[:, None] * dq[None, :]

    # d alpha / dt = d alpha / ds * ds / dt = alpha_dot_s / duration
    v_traj = (alpha_dot_s[:, None] / duration_float) * dq[None, :]

    return q_traj, v_traj


def _arm_slice(side: ArmSide) -> slice:
    """
    Return OCP q slice for one arm.

    Joint convention:
        left arm  = q[3:10]
        right arm = q[10:17]
    """
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
        Shape (num_nodes, 17).

    v_traj:
        Shape (num_nodes, 17).
    """
    q_cur = _as_1d_float_array(q_current, name="q_current")
    q_home = _as_1d_float_array(q_initial, name="q_initial")

    if q_cur.size != OCP_Q_DIM:
        raise ValueError(f"q_current must have size {OCP_Q_DIM}, got {q_cur.size}")

    if q_home.size != OCP_Q_DIM:
        raise ValueError(f"q_initial must have size {OCP_Q_DIM}, got {q_home.size}")

    idx = _arm_slice(side)

    q_arm_traj, v_arm_traj = build_minimum_jerk_joint_trajectory(
        q_start=q_cur[idx],
        q_goal=q_home[idx],
        duration=duration,
        dt=dt,
    )

    num_nodes = q_arm_traj.shape[0]

    q_traj = np.tile(q_cur.reshape(1, OCP_Q_DIM), (num_nodes, 1))
    v_traj = np.zeros((num_nodes, OCP_Q_DIM), dtype=float)

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
        Shape (num_nodes,).
    """
    duration_float = _validate_positive_scalar(duration, name="duration")
    dt_float = _validate_positive_scalar(dt, name="dt")

    c0 = float(command_start)
    c1 = float(command_goal)

    if not np.isfinite(c0):
        raise ValueError(f"command_start must be finite, got {command_start}")

    if not np.isfinite(c1):
        raise ValueError(f"command_goal must be finite, got {command_goal}")

    if c0 < 0.0 or c0 > 1.0:
        raise ValueError(f"command_start must be in [0, 1], got {c0}")

    if c1 < 0.0 or c1 > 1.0:
        raise ValueError(f"command_goal must be in [0, 1], got {c1}")

    time = build_time_grid(
        duration=duration_float,
        dt=dt_float,
        include_endpoint=True,
    )

    s = time / duration_float
    alpha = minimum_jerk_profile(s)

    command_traj = c0 + alpha * (c1 - c0)

    return np.clip(command_traj, 0.0, 1.0)


@dataclass(frozen=True)
class ResampledTrajectory:
    """
    Trajectory resampled to a fixed dt grid.

    time:
        Shape (num_nodes,).

    q:
        Shape (num_nodes, nq).

    v:
        Shape (num_nodes, nv).

    u:
        Optional, shape (num_nodes, nu).
    """

    time: np.ndarray
    q: np.ndarray
    v: np.ndarray
    u: np.ndarray | None

    dt: float
    original_tf: float
    resampled_tf: float

    def __post_init__(self) -> None:
        time = _as_1d_float_array(self.time, name="time")
        q = _as_2d_float_array(self.q, name="q")
        v = _as_2d_float_array(self.v, name="v")

        dt = _validate_positive_scalar(self.dt, name="dt")
        original_tf = _validate_positive_scalar(
            self.original_tf,
            name="original_tf",
        )

        resampled_tf = float(self.resampled_tf)

        if not np.isfinite(resampled_tf):
            raise ValueError(f"resampled_tf must be finite, got {self.resampled_tf}")

        if resampled_tf < 0.0:
            raise ValueError(f"resampled_tf must be >= 0, got {resampled_tf}")

        if time.ndim != 1:
            raise ValueError(f"time must be 1D, got shape {time.shape}")

        if time.size == 0:
            raise ValueError("time must be non-empty")

        if abs(time[0]) > 1e-12:
            raise ValueError(f"time must start at 0.0, got {time[0]}")

        if time.size > 1 and np.any(np.diff(time) <= 0.0):
            raise ValueError("time must be strictly increasing")

        if q.shape[0] != time.size:
            raise ValueError(
                f"q rows must match time size, got {q.shape[0]} and {time.size}"
            )

        if v.shape != q.shape:
            raise ValueError(f"v must have same shape as q, got {v.shape} and {q.shape}")

        u = None

        if self.u is not None:
            u = _as_2d_float_array(self.u, name="u")

            if u.shape[0] != time.size:
                raise ValueError(
                    f"u rows must match time size, got {u.shape[0]} and {time.size}"
                )

        if abs(resampled_tf - float(time[-1])) > 1e-9:
            raise ValueError(
                "resampled_tf must match time[-1]. "
                f"Got resampled_tf={resampled_tf}, time[-1]={time[-1]}"
            )

        object.__setattr__(self, "time", time)
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "v", v)
        object.__setattr__(self, "u", u)
        object.__setattr__(self, "dt", dt)
        object.__setattr__(self, "original_tf", original_tf)
        object.__setattr__(self, "resampled_tf", resampled_tf)


def build_fixed_dt_grid_from_tf(
    *,
    tf: float,
    dt: float,
    close_to_integer_tol: float = 1e-2,
) -> np.ndarray:
    """
    Build a fixed-dt time grid for an OCP solution.

    This function avoids significant extrapolation beyond the OCP duration.

    Example:
        tf = 0.400032
        dt = 0.02

    returns:
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
    tf_float = _validate_positive_scalar(tf, name="tf")
    dt_float = _validate_positive_scalar(dt, name="dt")
    tol = _validate_nonnegative_tolerance(
        close_to_integer_tol,
        name="close_to_integer_tol",
    )

    ratio = tf_float / dt_float
    nearest_intervals = int(round(ratio))

    if nearest_intervals <= 0:
        nearest_intervals = 1

    if abs(ratio - nearest_intervals) <= tol:
        num_intervals = nearest_intervals
    else:
        # Avoid extrapolating beyond the OCP trajectory duration.
        num_intervals = int(np.floor(ratio))
        num_intervals = max(num_intervals, 1)

    return np.arange(num_intervals + 1, dtype=float) * dt_float


def interpolate_vector_trajectory(
    *,
    values: np.ndarray,
    source_time: np.ndarray,
    target_time: np.ndarray,
    name: str = "values",
    endpoint_tolerance: float = 1e-8,
) -> np.ndarray:
    """
    Linearly interpolate a vector trajectory.

    values:
        Shape = (num_source_nodes, dim).

    source_time:
        Shape = (num_source_nodes,).

    target_time:
        Shape = (num_target_nodes,).

    endpoint_tolerance:
        Small tolerance for tiny floating-point mismatch at the final time.
    """
    values_arr = _as_2d_float_array(values, name=name)
    source_time_arr = _as_1d_float_array(source_time, name="source_time")
    target_time_arr = _as_1d_float_array(target_time, name="target_time")

    endpoint_tol = _validate_nonnegative_tolerance(
        endpoint_tolerance,
        name="endpoint_tolerance",
    )

    if values_arr.shape[0] != source_time_arr.size:
        raise ValueError(
            f"{name} rows must match source_time size. "
            f"Got {values_arr.shape[0]} and {source_time_arr.size}"
        )

    if source_time_arr.size == 0:
        raise ValueError("source_time must be non-empty")

    if target_time_arr.size == 0:
        raise ValueError("target_time must be non-empty")

    if abs(source_time_arr[0]) > 1e-12:
        raise ValueError(f"source_time must start at 0.0, got {source_time_arr[0]}")

    if np.any(np.diff(source_time_arr) <= 0.0):
        raise ValueError("source_time must be strictly increasing")

    if abs(target_time_arr[0]) > 1e-12:
        raise ValueError(f"target_time must start at 0.0, got {target_time_arr[0]}")

    if target_time_arr.size > 1 and np.any(np.diff(target_time_arr) <= 0.0):
        raise ValueError("target_time must be strictly increasing")

    source_end = float(source_time_arr[-1])
    target_end = float(target_time_arr[-1])

    if target_end > source_end + endpoint_tol:
        raise ValueError(
            "target_time exceeds source_time duration. "
            f"target end={target_end}, source end={source_end}"
        )

    # Tiny numerical mismatch only. Let np.interp safely clamp the final sample.
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
    Resample q and v trajectories to a fixed-dt grid.

    Returns
    -------
    time:
        Shape (num_resampled_nodes,).

    q_ref:
        Shape (num_resampled_nodes, nq).

    v_ref:
        Shape (num_resampled_nodes, nv).
    """
    q_arr = _as_2d_float_array(q_nodes, name="q_nodes")
    v_arr = _as_2d_float_array(v_nodes, name="v_nodes")

    tf_float = _validate_positive_scalar(tf, name="tf")
    dt_float = _validate_positive_scalar(dt, name="dt")

    if q_arr.shape != v_arr.shape:
        raise ValueError(
            f"q_nodes and v_nodes must have same shape. "
            f"Got {q_arr.shape} and {v_arr.shape}"
        )

    source_time = np.linspace(0.0, tf_float, q_arr.shape[0])
    target_time = build_fixed_dt_grid_from_tf(tf=tf_float, dt=dt_float)

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
    Resample an OCP solution to a fixed-dt grid.

    This is the function planner/executor should call before passing
    reference trajectory to RL or MuJoCo executor.
    """
    dt_float = _validate_positive_scalar(dt, name="dt")

    time, q_ref, v_ref = resample_qv_trajectory_to_dt(
        q_nodes=solution.q_nodes,
        v_nodes=solution.v_nodes,
        tf=solution.tf,
        dt=dt_float,
    )

    u_ref = None

    if include_u:
        U_nodes = _as_2d_float_array(solution.U_nodes, name="U_nodes")

        source_time = np.linspace(
            0.0,
            float(solution.tf),
            U_nodes.shape[0],
        )

        u_ref = interpolate_vector_trajectory(
            values=U_nodes,
            source_time=source_time,
            target_time=time,
            name="U_nodes",
        )

    return ResampledTrajectory(
        time=time,
        q=q_ref,
        v=v_ref,
        u=u_ref,
        dt=dt_float,
        original_tf=float(solution.tf),
        resampled_tf=float(time[-1]),
    )