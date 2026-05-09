from __future__ import annotations

import casadi as ca

from manipulation_ocp.casadi.pinocchio_kinematics import (
    CasadiPinocchioKinematics,
)


def quadratic_cost(
    error: ca.MX,
    W: ca.DM,
) -> ca.MX:
    """
    Weighted quadratic cost.

        cost = error.T @ W @ error

    Parameters
    ----------
    error:
        Error vector.

    W:
        Positive semi-definite weight matrix.

    Returns
    -------
    cost:
        Scalar CasADi expression.
    """
    return ca.mtimes([error.T, W, error])


def split_state(
    x: ca.MX,
    *,
    nq: int,
    nv: int,
) -> tuple[ca.MX, ca.MX]:
    """
    Split state:

        x = [q; v]

    where:
        q ∈ R^nq
        v ∈ R^nv
    """
    q = x[0:nq]
    v = x[nq : nq + nv]

    return q, v


def build_dual_ee_stage_cost(
    *,
    x: ca.MX,
    u: ca.MX,
    kinematics: CasadiPinocchioKinematics,
    p_ref: ca.MX,
    W_ee: ca.DM,
    W_v: ca.DM,
    W_u: ca.DM,
    nq: int,
    nv: int,
) -> ca.MX:
    """
    Build dual-end-effector stage cost.

    This function is generic: it can be evaluated at a node or at an
    implicit-midpoint collocation point.

    In the current OCP formulation, it is evaluated at the interval midpoint:

        x = x_mid
        u = u_mid

    Stage cost:

        L(x, u)
          = ||p(q) - p_ref||^2_W_ee
          + ||v||^2_W_v
          + ||u||^2_W_u

    where:

        x = [q; v]

    Important:
        End-effector position p(q) depends only on q, not on full state
        x = [q; v].

    Therefore this function intentionally uses:

        kinematics.dual_ee_position_fn(q)

    instead of a full-state FK function.

    This keeps the CasADi graph smaller and makes the dependency structure
    clearer.
    """
    q, v = split_state(x, nq=nq, nv=nv)

    p_pair = kinematics.dual_ee_position_fn(q)
    p_error = p_pair - p_ref

    ee_cost = quadratic_cost(p_error, W_ee)
    velocity_cost = quadratic_cost(v, W_v)
    control_cost = quadratic_cost(u, W_u)

    return ee_cost + velocity_cost + control_cost


def build_dual_ee_terminal_cost(
    *,
    x: ca.MX,
    kinematics: CasadiPinocchioKinematics,
    p_ref: ca.MX,
    W_ee_f: ca.DM,
    W_v_f: ca.DM,
    nq: int,
    nv: int,
) -> ca.MX:
    """
    Build dual-end-effector terminal cost.

    Terminal cost is evaluated at the final node:

        x = x_N = [q_N; v_N]

    Terminal cost:

        Phi(x_N)
          = ||p(q_N) - p_ref||^2_W_ee_f
          + ||v_N||^2_W_v_f

    Important:
        End-effector position p(q_N) depends only on q_N, not on v_N.
    """
    q, v = split_state(x, nq=nq, nv=nv)

    p_pair = kinematics.dual_ee_position_fn(q)
    p_error = p_pair - p_ref

    ee_terminal_cost = quadratic_cost(p_error, W_ee_f)
    terminal_velocity_cost = quadratic_cost(v, W_v_f)

    return ee_terminal_cost + terminal_velocity_cost