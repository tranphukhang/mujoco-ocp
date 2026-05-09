from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pinocchio as pin

from manipulation_ocp.configs.ik import DualEEIKConfig
from manipulation_ocp.configs.initial_guess import InitialGuessConfig
from manipulation_ocp.pinocchio.dynamics import compute_rnea_trajectory
from manipulation_ocp.pinocchio.kinematics import (
    get_ee_position_jacobians,
    get_ee_positions,
)
from manipulation_ocp.pinocchio.limits import (
    clip_q_to_limits,
    clip_trajectory_u_to_limits,
    clip_trajectory_v_to_limits,
)
from manipulation_ocp.robots.g1_gripper import (
    LEFT_EE_SITE_NAME,
    RIGHT_EE_SITE_NAME,
)
from manipulation_ocp.utils.numerics import (
    as_vector,
    finite_difference_first_order,
    linear_interpolation,
)


@dataclass(frozen=True)
class DualEEIKResult:
    """
    Result of lightweight dual-end-effector IK.
    """

    q_goal: np.ndarray

    converged: bool
    accepted_for_initial_guess: bool
    used_fallback_to_seed: bool

    iterations: int
    init_error_norm: float
    best_error_norm: float
    final_error_norm: float
    improvement_abs: float
    improvement_ratio: float
    best_iter: int

    final_left_position: np.ndarray
    final_right_position: np.ndarray

    weight_left: float
    weight_right: float


@dataclass(frozen=True)
class InitialGuessResult:
    """
    Initial guess result for OCP.

    All trajectories use the Pinocchio/OCP model dimension:
        q, v, a, u ∈ R^17
    """

    q_goal: np.ndarray

    q_guess: np.ndarray
    v_guess: np.ndarray
    a_guess: np.ndarray
    u_guess: np.ndarray

    t: np.ndarray
    dt: float
    total_time: float

    ik_result: DualEEIKResult

    left_goal_position: np.ndarray
    right_goal_position: np.ndarray


def _validate_ik_config(config: DualEEIKConfig) -> None:
    if config.max_iters <= 0:
        raise ValueError("max_iters must be > 0")

    if config.tol <= 0.0:
        raise ValueError("tol must be > 0")

    if config.step_size <= 0.0:
        raise ValueError("step_size must be > 0")

    if config.damping < 0.0:
        raise ValueError("damping must be >= 0")

    if config.patience <= 0:
        raise ValueError("patience must be > 0")

    if config.min_progress_ratio < 0.0:
        raise ValueError("min_progress_ratio must be >= 0")

    if config.min_progress_abs < 0.0:
        raise ValueError("min_progress_abs must be >= 0")

    if config.weight_left <= 0.0:
        raise ValueError("weight_left must be > 0")

    if config.weight_right <= 0.0:
        raise ValueError("weight_right must be > 0")


def _validate_guess_config(config: InitialGuessConfig) -> None:
    if config.n_intervals <= 0:
        raise ValueError("n_intervals must be > 0")

    if config.dt <= 0.0:
        raise ValueError("dt must be > 0")

    if config.default_v_limit_if_invalid <= 0.0:
        raise ValueError("default_v_limit_if_invalid must be > 0")


def _get_dual_ee_positions(
    model: pin.Model,
    data: pin.Data,
    q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return left and right EE positions in the Pinocchio pelvis/base frame.
    """
    positions = get_ee_positions(model, data, q)

    return positions["left"], positions["right"]


def _get_dual_ee_position_jacobian(
    model: pin.Model,
    data: pin.Data,
    q: np.ndarray,
) -> np.ndarray:
    """
    Return stacked dual-EE position Jacobian.

    Shape:
        J_dual = (6, nv)

    Rows:
        0:3 -> left EE linear Jacobian
        3:6 -> right EE linear Jacobian
    """
    jacobians = get_ee_position_jacobians(model, data, q)

    return np.vstack(
        [
            jacobians["left"],
            jacobians["right"],
        ]
    )


def _weighted_dual_ee_error(
    left_current: np.ndarray,
    right_current: np.ndarray,
    left_target: np.ndarray,
    right_target: np.ndarray,
    *,
    sqrt_weight_left: float,
    sqrt_weight_right: float,
) -> np.ndarray:
    """
    Stack weighted dual-EE position error.

    Error convention:
        error = target - current
    """
    left_error = left_target - left_current
    right_error = right_target - right_current

    return np.concatenate(
        [
            sqrt_weight_left * left_error,
            sqrt_weight_right * right_error,
        ],
        axis=0,
    )


def _weighted_dual_ee_jacobian(
    left_jacobian: np.ndarray,
    right_jacobian: np.ndarray,
    *,
    sqrt_weight_left: float,
    sqrt_weight_right: float,
) -> np.ndarray:
    """
    Stack weighted dual-EE position Jacobian.

    Shape:
        left_jacobian  = (3, nv)
        right_jacobian = (3, nv)
        output         = (6, nv)
    """
    return np.vstack(
        [
            sqrt_weight_left * left_jacobian,
            sqrt_weight_right * right_jacobian,
        ]
    )


def solve_dual_ee_ik(
    model: pin.Model,
    data: pin.Data,
    q_seed: Sequence[float] | np.ndarray,
    left_target: Sequence[float] | np.ndarray,
    right_target: Sequence[float] | np.ndarray,
    *,
    config: DualEEIKConfig | None = None,
) -> DualEEIKResult:
    """
    Lightweight dual-end-effector position IK.

    This function only solves for one final configuration:

        q_seed -> q_goal

    It is intended for OCP initial guess generation.

    Frame convention
    ----------------
    left_target and right_target must be expressed in the Pinocchio
    pelvis/base frame.

    The IK error is:

        e = [
            sqrt(w_left)  * (p_left_target  - p_left(q))
            sqrt(w_right) * (p_right_target - p_right(q))
        ]

    The update is damped least-squares:

        A  = J Jᵀ + damping * I
        dq = Jᵀ A⁻¹ e
        q  = integrate(q, step_size * dq)
    """
    if config is None:
        config = DualEEIKConfig()

    _validate_ik_config(config)

    q_seed_vec = as_vector(q_seed, size=model.nq, name="q_seed")
    q = q_seed_vec.copy()

    left_target_vec = as_vector(left_target, size=3, name="left_target")
    right_target_vec = as_vector(right_target, size=3, name="right_target")

    if config.clip_to_limits:
        q = clip_q_to_limits(model, q)

    sqrt_weight_left = float(np.sqrt(config.weight_left))
    sqrt_weight_right = float(np.sqrt(config.weight_right))

    left_init, right_init = _get_dual_ee_positions(model, data, q)

    init_error = _weighted_dual_ee_error(
        left_init,
        right_init,
        left_target_vec,
        right_target_vec,
        sqrt_weight_left=sqrt_weight_left,
        sqrt_weight_right=sqrt_weight_right,
    )
    init_error_norm = float(np.linalg.norm(init_error))

    best_q = q.copy()
    best_error_norm = init_error_norm
    best_left_position = left_init.copy()
    best_right_position = right_init.copy()
    best_iter = 0

    if config.verbose:
        print(
            "[dual IK] init "
            f"|err|={init_error_norm:.6e} "
            f"left={left_init} "
            f"right={right_init}"
        )

    if init_error_norm < config.tol:
        return DualEEIKResult(
            q_goal=best_q,
            converged=True,
            accepted_for_initial_guess=True,
            used_fallback_to_seed=False,
            iterations=0,
            init_error_norm=init_error_norm,
            best_error_norm=best_error_norm,
            final_error_norm=best_error_norm,
            improvement_abs=0.0,
            improvement_ratio=0.0,
            best_iter=0,
            final_left_position=best_left_position,
            final_right_position=best_right_position,
            weight_left=config.weight_left,
            weight_right=config.weight_right,
        )

    no_improve_counter = 0
    converged = False
    iterations = 0

    for it in range(config.max_iters):
        iterations = it + 1

        left_current, right_current = _get_dual_ee_positions(model, data, q)

        error = _weighted_dual_ee_error(
            left_current,
            right_current,
            left_target_vec,
            right_target_vec,
            sqrt_weight_left=sqrt_weight_left,
            sqrt_weight_right=sqrt_weight_right,
        )
        error_norm = float(np.linalg.norm(error))

        if config.verbose:
            left_error_norm = float(
                np.linalg.norm(left_target_vec - left_current)
            )
            right_error_norm = float(
                np.linalg.norm(right_target_vec - right_current)
            )
            print(
                f"[dual IK] iter={it:03d} "
                f"|err|={error_norm:.6e} "
                f"|left|={left_error_norm:.6e} "
                f"|right|={right_error_norm:.6e}"
            )

        if error_norm < best_error_norm:
            best_error_norm = error_norm
            best_q = q.copy()
            best_left_position = left_current.copy()
            best_right_position = right_current.copy()
            best_iter = it
            no_improve_counter = 0
        else:
            no_improve_counter += 1

        if error_norm < config.tol:
            converged = True
            break

        if no_improve_counter >= config.patience:
            if config.verbose:
                print(
                    "[dual IK] Early stop: no improvement for "
                    f"{config.patience} iterations."
                )
            break

        dual_jacobian = _get_dual_ee_position_jacobian(model, data, q)

        weighted_jacobian = _weighted_dual_ee_jacobian(
            dual_jacobian[:3, :],
            dual_jacobian[3:, :],
            sqrt_weight_left=sqrt_weight_left,
            sqrt_weight_right=sqrt_weight_right,
        )

        lhs = (
            weighted_jacobian @ weighted_jacobian.T
            + config.damping * np.eye(6)
        )

        try:
            dq = weighted_jacobian.T @ np.linalg.solve(lhs, error)
        except np.linalg.LinAlgError:
            dq = weighted_jacobian.T @ np.linalg.pinv(lhs) @ error

        dq_norm = float(np.linalg.norm(dq))

        if dq_norm < 1e-10:
            if config.verbose:
                print("[dual IK] Early stop: dq is too small.")
            break

        q = pin.integrate(model, q, config.step_size * dq)

        if config.clip_to_limits:
            q = clip_q_to_limits(model, q)

    improvement_abs = init_error_norm - best_error_norm
    improvement_ratio = improvement_abs / max(init_error_norm, 1e-12)

    accepted_for_initial_guess = (
        converged
        or (improvement_ratio >= config.min_progress_ratio)
        or (improvement_abs >= config.min_progress_abs)
    )

    used_fallback_to_seed = False

    if (
        config.fallback_to_seed_if_poor_progress
        and not accepted_for_initial_guess
    ):
        q_out = q_seed_vec.copy()

        if config.clip_to_limits:
            q_out = clip_q_to_limits(model, q_out)

        left_out, right_out = _get_dual_ee_positions(model, data, q_out)

        final_error = _weighted_dual_ee_error(
            left_out,
            right_out,
            left_target_vec,
            right_target_vec,
            sqrt_weight_left=sqrt_weight_left,
            sqrt_weight_right=sqrt_weight_right,
        )
        final_error_norm = float(np.linalg.norm(final_error))
        used_fallback_to_seed = True
    else:
        q_out = best_q.copy()
        left_out = best_left_position.copy()
        right_out = best_right_position.copy()
        final_error_norm = best_error_norm

    return DualEEIKResult(
        q_goal=q_out,
        converged=converged,
        accepted_for_initial_guess=accepted_for_initial_guess,
        used_fallback_to_seed=used_fallback_to_seed,
        iterations=iterations,
        init_error_norm=init_error_norm,
        best_error_norm=best_error_norm,
        final_error_norm=final_error_norm,
        improvement_abs=improvement_abs,
        improvement_ratio=improvement_ratio,
        best_iter=best_iter,
        final_left_position=left_out,
        final_right_position=right_out,
        weight_left=config.weight_left,
        weight_right=config.weight_right,
    )


def build_linear_initial_guess(
    model: pin.Model,
    data: pin.Data,
    q_start: Sequence[float] | np.ndarray,
    q_goal: Sequence[float] | np.ndarray,
    *,
    config: InitialGuessConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build q, v, a, u initial guess from q_start and q_goal.

    Returns
    -------
    t:
        shape = (num_nodes,)

    q_guess:
        shape = (num_nodes, nq)

    v_guess:
        shape = (num_nodes, nv)

    a_guess:
        shape = (num_nodes, nv)

    u_guess:
        shape = (num_nodes, nv)
    """
    if config is None:
        config = InitialGuessConfig()

    _validate_guess_config(config)

    q_start_vec = as_vector(q_start, size=model.nq, name="q_start")
    q_goal_vec = as_vector(q_goal, size=model.nq, name="q_goal")

    q_start_vec = clip_q_to_limits(model, q_start_vec)
    q_goal_vec = clip_q_to_limits(model, q_goal_vec)

    t = np.linspace(0.0, config.total_time, config.num_nodes)

    q_guess = linear_interpolation(
        q_start_vec,
        q_goal_vec,
        config.num_nodes,
    )

    v_guess = finite_difference_first_order(
        q_guess,
        config.dt,
        axis=0,
    )

    if config.clip_v_to_limits:
        v_guess = clip_trajectory_v_to_limits(
            model,
            v_guess,
            default_if_invalid=config.default_v_limit_if_invalid,
        )

    a_guess = finite_difference_first_order(
        v_guess,
        config.dt,
        axis=0,
    )

    u_guess = compute_rnea_trajectory(
        model,
        data,
        q_guess,
        v_guess,
        a_guess,
    )

    if config.clip_u_to_limits:
        u_guess = clip_trajectory_u_to_limits(model, u_guess)

    return t, q_guess, v_guess, a_guess, u_guess


def build_initial_guess_from_dual_ee_targets(
    model: pin.Model,
    data: pin.Data,
    q_start: Sequence[float] | np.ndarray,
    left_target: Sequence[float] | np.ndarray,
    right_target: Sequence[float] | np.ndarray,
    *,
    ik_config: DualEEIKConfig | None = None,
    guess_config: InitialGuessConfig | None = None,
    q_seed: Sequence[float] | np.ndarray | None = None,
) -> InitialGuessResult:
    """
    Build OCP initial guess from dual-EE targets.

    Pipeline:
        dual-EE IK -> q_goal
        q_start -> q_goal linear interpolation -> q_guess
        finite difference -> v_guess, a_guess
        RNEA -> u_guess

    Frame convention:
        left_target and right_target must be expressed in the Pinocchio
        pelvis/base frame.
    """
    if ik_config is None:
        ik_config = DualEEIKConfig()

    if guess_config is None:
        guess_config = InitialGuessConfig()

    q_start_vec = as_vector(q_start, size=model.nq, name="q_start")
    q_start_vec = clip_q_to_limits(model, q_start_vec)

    if q_seed is None:
        q_seed_vec = q_start_vec.copy()
    else:
        q_seed_vec = as_vector(q_seed, size=model.nq, name="q_seed")
        q_seed_vec = clip_q_to_limits(model, q_seed_vec)

    left_target_vec = as_vector(left_target, size=3, name="left_target")
    right_target_vec = as_vector(right_target, size=3, name="right_target")

    ik_result = solve_dual_ee_ik(
        model,
        data,
        q_seed_vec,
        left_target_vec,
        right_target_vec,
        config=ik_config,
    )

    t, q_guess, v_guess, a_guess, u_guess = build_linear_initial_guess(
        model,
        data,
        q_start_vec,
        ik_result.q_goal,
        config=guess_config,
    )

    return InitialGuessResult(
        q_goal=ik_result.q_goal,
        q_guess=q_guess,
        v_guess=v_guess,
        a_guess=a_guess,
        u_guess=u_guess,
        t=t,
        dt=guess_config.dt,
        total_time=guess_config.total_time,
        ik_result=ik_result,
        left_goal_position=ik_result.final_left_position,
        right_goal_position=ik_result.final_right_position,
    )


def sample_dual_ee_targets_around_q(
    model: pin.Model,
    data: pin.Data,
    q_center: Sequence[float] | np.ndarray,
    *,
    left_xyz_span: float | Sequence[float] = (0.05, 0.05, 0.05),
    right_xyz_span: float | Sequence[float] = (0.05, 0.05, 0.05),
    seed: int | None = None,
    left_z_min: float | None = None,
    left_z_max: float | None = None,
    right_z_min: float | None = None,
    right_z_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """
    Sample random dual-EE targets around current EE positions.

    Targets are expressed in the Pinocchio pelvis/base frame.
    """
    q_center_vec = as_vector(q_center, size=model.nq, name="q_center")

    positions = get_ee_positions(model, data, q_center_vec)
    left_center = positions["left"]
    right_center = positions["right"]

    def _normalize_span(
        span: float | Sequence[float],
        name: str,
    ) -> np.ndarray:
        if np.isscalar(span):
            value = float(span)

            if value < 0.0:
                raise ValueError(f"{name} must be >= 0")

            return np.array([value, value, value], dtype=float)

        arr = as_vector(span, size=3, name=name)

        if np.any(arr < 0.0):
            raise ValueError(f"{name} entries must be >= 0")

        return arr

    left_span = _normalize_span(left_xyz_span, "left_xyz_span")
    right_span = _normalize_span(right_xyz_span, "right_xyz_span")

    rng = np.random.default_rng(seed)

    left_offset = np.array(
        [
            rng.uniform(-left_span[0], left_span[0]),
            rng.uniform(-left_span[1], left_span[1]),
            rng.uniform(-left_span[2], left_span[2]),
        ],
        dtype=float,
    )

    right_offset = np.array(
        [
            rng.uniform(-right_span[0], right_span[0]),
            rng.uniform(-right_span[1], right_span[1]),
            rng.uniform(-right_span[2], right_span[2]),
        ],
        dtype=float,
    )

    left_target = left_center + left_offset
    right_target = right_center + right_offset

    if left_z_min is not None:
        left_target[2] = max(float(left_z_min), left_target[2])

    if left_z_max is not None:
        left_target[2] = min(float(left_z_max), left_target[2])

    if right_z_min is not None:
        right_target[2] = max(float(right_z_min), right_target[2])

    if right_z_max is not None:
        right_target[2] = min(float(right_z_max), right_target[2])

    info: dict[str, object] = {
        "seed": seed,
        "left": {
            "frame": LEFT_EE_SITE_NAME,
            "center": left_center,
            "offset": left_offset,
            "target": left_target,
        },
        "right": {
            "frame": RIGHT_EE_SITE_NAME,
            "center": right_center,
            "offset": right_offset,
            "target": right_target,
        },
    }

    return left_target, right_target, info