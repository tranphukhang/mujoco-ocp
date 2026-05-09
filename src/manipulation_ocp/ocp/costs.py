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
    Weighted quadratic cost:

        error^T W error
    """
    return ca.mtimes([error.T, W, error])


def split_state(
    x: ca.MX,
    *,
    nq: int,
    nv: int,
) -> tuple[ca.MX, ca.MX]:
    """
    Split x = [q; v].
    """
    q = x[0:nq]
    v = x[nq : nq + nv]
    return q, v


def build_dual_ee_stage_cost(
    *,
    x: ca.MX,
    u: ca.MX,
    kinematics: CasadiPinocchioKinematics,
    p_ref: ca.DM,
    W_ee: ca.DM,
    W_v: ca.DM,
    W_u: ca.DM,
    nq: int,
    nv: int,
) -> ca.MX:
    """
    Stage cost:

        L(x, u)
          = ||p(q) - p_ref||^2_W_ee
          + ||v||^2_W_v
          + ||u||^2_W_u

    where:
        p(q) = [p_left(q); p_right(q)] ∈ R6
    """
    _, v = split_state(x, nq=nq, nv=nv)

    p_pair = kinematics.dual_ee_position_x_fn(x)
    p_error = p_pair - p_ref

    ee_cost = quadratic_cost(p_error, W_ee)
    v_cost = quadratic_cost(v, W_v)
    u_cost = quadratic_cost(u, W_u)

    return ee_cost + v_cost + u_cost


def build_dual_ee_terminal_cost(
    *,
    x: ca.MX,
    kinematics: CasadiPinocchioKinematics,
    p_ref: ca.DM,
    W_ee_f: ca.DM,
    W_v_f: ca.DM,
    nq: int,
    nv: int,
) -> ca.MX:
    """
    Terminal cost:

        Phi(x_N)
          = ||p(q_N) - p_ref||^2_W_ee_f
          + ||v_N||^2_W_v_f
    """
    _, v = split_state(x, nq=nq, nv=nv)

    p_pair = kinematics.dual_ee_position_x_fn(x)
    p_error = p_pair - p_ref

    ee_terminal_cost = quadratic_cost(p_error, W_ee_f)
    v_terminal_cost = quadratic_cost(v, W_v_f)

    return ee_terminal_cost + v_terminal_cost