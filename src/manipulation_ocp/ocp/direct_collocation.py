from __future__ import annotations

import casadi as ca


def compute_midpoint_control(
    uk: ca.MX,
    uk_next: ca.MX,
) -> ca.MX:
    """
    Midpoint control for Hermite-Simpson.

    u_c = 0.5 * (u_k + u_{k+1})
    """
    return 0.5 * (uk + uk_next)


def compute_cubic_midpoint_state(
    *,
    xk: ca.MX,
    xk_next: ca.MX,
    fk: ca.MX,
    fk_next: ca.MX,
    h: ca.MX,
) -> ca.MX:
    """
    Cubic Hermite midpoint state.

    x_c = 0.5 * (x_k + x_{k+1})
        + h/8 * (f_k - f_{k+1})
    """
    return 0.5 * (xk + xk_next) + (h / 8.0) * (fk - fk_next)


def compute_hermite_simpson_defect(
    *,
    xk: ca.MX,
    xk_next: ca.MX,
    fk: ca.MX,
    fc: ca.MX,
    fk_next: ca.MX,
    h: ca.MX,
) -> ca.MX:
    """
    Hermite-Simpson defect constraint.

    defect_k =
        x_{k+1} - x_k
        - h/6 * (f_k + 4 f_c + f_{k+1})

    Dynamics constraint:
        defect_k = 0
    """
    return xk_next - xk - (h / 6.0) * (fk + 4.0 * fc + fk_next)


def compute_hermite_simpson_interval(
    *,
    f_dyn: ca.Function,
    xk: ca.MX,
    xk_next: ca.MX,
    uk: ca.MX,
    uk_next: ca.MX,
    h: ca.MX,
) -> tuple[ca.MX, ca.MX, ca.MX, ca.MX, ca.MX, ca.MX]:
    """
    Compute all Hermite-Simpson quantities for one interval.

    Returns
    -------
    defect:
        Dynamics defect constraint, shape (nx, 1)

    xc:
        Cubic midpoint state.

    uc:
        Midpoint control.

    fk, fc, fk_next:
        Explicit dynamics at node k, midpoint, and node k+1.
    """
    fk = f_dyn(xk, uk)
    fk_next = f_dyn(xk_next, uk_next)

    uc = compute_midpoint_control(uk, uk_next)

    xc = compute_cubic_midpoint_state(
        xk=xk,
        xk_next=xk_next,
        fk=fk,
        fk_next=fk_next,
        h=h,
    )

    fc = f_dyn(xc, uc)

    defect = compute_hermite_simpson_defect(
        xk=xk,
        xk_next=xk_next,
        fk=fk,
        fc=fc,
        fk_next=fk_next,
        h=h,
    )

    return defect, xc, uc, fk, fc, fk_next