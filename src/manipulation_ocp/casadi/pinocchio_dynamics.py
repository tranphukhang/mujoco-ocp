from __future__ import annotations

from dataclasses import dataclass

import casadi as ca
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin

from manipulation_ocp.utils.casadi import (
    evaluate_vector_function,
    sx_vector,
)
from manipulation_ocp.utils.numerics import as_vector


@dataclass(frozen=True)
class CasadiPinocchioDynamics:
    """
    CasADi symbolic dynamics built from a Pinocchio model.

    This class provides both:

    1. Implicit dynamics residual, preferred for direct collocation:

        residual = F(x, xdot, u)

        x     = [q; v]
        xdot  = [qdot; vdot]
        u     = tau

        F = [
            qdot - v
            RNEA(q, v, vdot) - tau
        ]

    2. Explicit dynamics, useful for debugging:

        xdot = f(x, u)

        xdot = [
            v
            ABA(q, v, tau)
        ]
    """

    model: pin.Model
    cmodel: cpin.Model
    cdata: object

    nq: int
    nv: int
    nx: int
    nu: int

    rnea_fn: ca.Function
    aba_fn: ca.Function
    implicit_dynamics_fn: ca.Function
    explicit_dynamics_fn: ca.Function

    def evaluate_rnea(
        self,
        q: np.ndarray,
        v: np.ndarray,
        a: np.ndarray,
    ) -> np.ndarray:
        """Evaluate symbolic RNEA numerically."""
        q_vec = as_vector(q, size=self.nq, name="q")
        v_vec = as_vector(v, size=self.nv, name="v")
        a_vec = as_vector(a, size=self.nv, name="a")

        return evaluate_vector_function(
            self.rnea_fn,
            q_vec,
            v_vec,
            a_vec,
            expected_size=self.nv,
            name="tau",
        )

    def evaluate_aba(
        self,
        q: np.ndarray,
        v: np.ndarray,
        u: np.ndarray,
    ) -> np.ndarray:
        """Evaluate symbolic ABA numerically."""
        q_vec = as_vector(q, size=self.nq, name="q")
        v_vec = as_vector(v, size=self.nv, name="v")
        u_vec = as_vector(u, size=self.nu, name="u")

        return evaluate_vector_function(
            self.aba_fn,
            q_vec,
            v_vec,
            u_vec,
            expected_size=self.nv,
            name="ddq",
        )

    def evaluate_explicit_dynamics(
        self,
        x: np.ndarray,
        u: np.ndarray,
    ) -> np.ndarray:
        """Evaluate xdot = f(x, u)."""
        x_vec = as_vector(x, size=self.nx, name="x")
        u_vec = as_vector(u, size=self.nu, name="u")

        return evaluate_vector_function(
            self.explicit_dynamics_fn,
            x_vec,
            u_vec,
            expected_size=self.nx,
            name="xdot",
        )

    def evaluate_implicit_dynamics(
        self,
        x: np.ndarray,
        xdot: np.ndarray,
        u: np.ndarray,
    ) -> np.ndarray:
        """Evaluate residual F(x, xdot, u)."""
        x_vec = as_vector(x, size=self.nx, name="x")
        xdot_vec = as_vector(xdot, size=self.nx, name="xdot")
        u_vec = as_vector(u, size=self.nu, name="u")

        return evaluate_vector_function(
            self.implicit_dynamics_fn,
            x_vec,
            xdot_vec,
            u_vec,
            expected_size=self.nx,
            name="implicit_residual",
        )


def _validate_fixed_base_model(model: pin.Model) -> None:
    """
    Validate assumptions for the current fixed-base manipulator model.

    Current OCP setup assumes:
        nq == nv
        nu == nv
    """
    if model.nq != model.nv:
        raise ValueError(
            "This dynamics builder assumes fixed-base model with nq == nv. "
            f"Got nq={model.nq}, nv={model.nv}."
        )


def build_casadi_pinocchio_dynamics(
    model: pin.Model,
    *,
    name_prefix: str = "pinocchio",
) -> CasadiPinocchioDynamics:
    """
    Build CasADi symbolic dynamics functions from a Pinocchio model.

    Parameters
    ----------
    model:
        Numeric Pinocchio model.

    name_prefix:
        Prefix used for CasADi function names.

    Returns
    -------
    dyn:
        CasadiPinocchioDynamics object containing symbolic functions.

    Main function for direct collocation:
        dyn.implicit_dynamics_fn(x, xdot, u) -> residual

    Optional debug function:
        dyn.explicit_dynamics_fn(x, u) -> xdot
    """
    _validate_fixed_base_model(model)

    cmodel = cpin.Model(model)
    cdata = cmodel.createData()

    nq = model.nq
    nv = model.nv
    nx = nq + nv
    nu = nv

    # -------------------------------------------------------------------------
    # Basic symbolic variables
    # -------------------------------------------------------------------------
    q = sx_vector("q", nq)
    v = sx_vector("v", nv)
    a = sx_vector("a", nv)
    u = sx_vector("u", nu)

    x = sx_vector("x", nx)
    xdot = sx_vector("xdot", nx)

    q_x = x[:nq]
    v_x = x[nq : nq + nv]

    qdot_xdot = xdot[:nq]
    vdot_xdot = xdot[nq : nq + nv]

    # -------------------------------------------------------------------------
    # RNEA: inverse dynamics
    # tau = RNEA(q, v, a)
    # -------------------------------------------------------------------------
    tau_rnea = cpin.rnea(
        cmodel,
        cdata,
        q,
        v,
        a,
    )

    rnea_fn = ca.Function(
        f"{name_prefix}_rnea",
        [q, v, a],
        [tau_rnea],
        ["q", "v", "a"],
        ["tau"],
    )

    # -------------------------------------------------------------------------
    # ABA: forward dynamics
    # ddq = ABA(q, v, tau)
    # -------------------------------------------------------------------------
    ddq_aba = cpin.aba(
        cmodel,
        cdata,
        q,
        v,
        u,
    )

    aba_fn = ca.Function(
        f"{name_prefix}_aba",
        [q, v, u],
        [ddq_aba],
        ["q", "v", "u"],
        ["ddq"],
    )

    # -------------------------------------------------------------------------
    # Explicit continuous dynamics
    # xdot = [v; ABA(q, v, u)]
    # -------------------------------------------------------------------------
    ddq_x = cpin.aba(
        cmodel,
        cdata,
        q_x,
        v_x,
        u,
    )

    explicit_xdot = ca.vertcat(
        v_x,
        ddq_x,
    )

    explicit_dynamics_fn = ca.Function(
        f"{name_prefix}_explicit_dynamics",
        [x, u],
        [explicit_xdot],
        ["x", "u"],
        ["xdot"],
    )

    # -------------------------------------------------------------------------
    # Implicit continuous dynamics residual
    #
    # residual = [
    #   qdot - v
    #   RNEA(q, v, vdot) - u
    # ]
    #
    # Direct collocation should use this residual:
    #   implicit_dynamics_fn(x_colloc, xdot_colloc, u_colloc) == 0
    # -------------------------------------------------------------------------
    tau_from_acc = cpin.rnea(
        cmodel,
        cdata,
        q_x,
        v_x,
        vdot_xdot,
    )

    implicit_residual = ca.vertcat(
        qdot_xdot - v_x,
        tau_from_acc - u,
    )

    implicit_dynamics_fn = ca.Function(
        f"{name_prefix}_implicit_dynamics",
        [x, xdot, u],
        [implicit_residual],
        ["x", "xdot", "u"],
        ["residual"],
    )

    return CasadiPinocchioDynamics(
        model=model,
        cmodel=cmodel,
        cdata=cdata,
        nq=nq,
        nv=nv,
        nx=nx,
        nu=nu,
        rnea_fn=rnea_fn,
        aba_fn=aba_fn,
        implicit_dynamics_fn=implicit_dynamics_fn,
        explicit_dynamics_fn=explicit_dynamics_fn,
    )