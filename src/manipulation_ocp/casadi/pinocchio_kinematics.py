from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import casadi as ca
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin

from manipulation_ocp.robots.g1_gripper import (
    LEFT_EE_SITE_NAME,
    RIGHT_EE_SITE_NAME,
)
from manipulation_ocp.utils.casadi import (
    evaluate_vector_function,
    sx_vector,
)
from manipulation_ocp.utils.numerics import as_vector


@dataclass(frozen=True)
class CasadiPinocchioKinematics:
    """
    CasADi symbolic kinematics functions built from a Pinocchio model.

    Main usage for OCP cost:
        dual_ee_position_fn(q)   -> [p_left; p_right]
        dual_ee_position_x_fn(x) -> [p_left; p_right]

    where:
        q = 17 G1 upper-body joint positions
        x = [q; v] ∈ R34

    Frame convention:
        EE positions are expressed in the Pinocchio pelvis/base frame.
    """

    model: pin.Model
    cmodel: cpin.Model

    nq: int
    nv: int
    nx: int

    left_frame_name: str
    right_frame_name: str

    left_frame_id: int
    right_frame_id: int

    left_ee_position_fn: ca.Function
    right_ee_position_fn: ca.Function
    dual_ee_position_fn: ca.Function
    dual_ee_position_x_fn: ca.Function

    def evaluate_left_position(
        self,
        q: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Evaluate left EE position numerically."""
        q_vec = as_vector(q, size=self.nq, name="q")

        return evaluate_vector_function(
            self.left_ee_position_fn,
            q_vec,
            expected_size=3,
            name="left_ee_position",
        )

    def evaluate_right_position(
        self,
        q: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Evaluate right EE position numerically."""
        q_vec = as_vector(q, size=self.nq, name="q")

        return evaluate_vector_function(
            self.right_ee_position_fn,
            q_vec,
            expected_size=3,
            name="right_ee_position",
        )

    def evaluate_dual_position(
        self,
        q: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """
        Evaluate stacked dual-EE position numerically.

        Returns
        -------
        p_pair:
            [p_left; p_right], shape (6,)
        """
        q_vec = as_vector(q, size=self.nq, name="q")

        return evaluate_vector_function(
            self.dual_ee_position_fn,
            q_vec,
            expected_size=6,
            name="dual_ee_position",
        )

    def evaluate_dual_position_from_state(
        self,
        x: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """
        Evaluate stacked dual-EE position from state x = [q; v].
        """
        x_vec = as_vector(x, size=self.nx, name="x")

        return evaluate_vector_function(
            self.dual_ee_position_x_fn,
            x_vec,
            expected_size=6,
            name="dual_ee_position_x",
        )


def _validate_fixed_base_model(model: pin.Model) -> None:
    if model.nq != model.nv:
        raise ValueError(
            "This kinematics builder assumes fixed-base model with nq == nv. "
            f"Got nq={model.nq}, nv={model.nv}."
        )


def _get_frame_id(model: pin.Model, frame_name: str) -> int:
    if not model.existFrame(frame_name):
        available_frames = [frame.name for frame in model.frames]
        raise ValueError(
            f"Frame not found: {frame_name}\n"
            f"Available frames: {available_frames}"
        )

    return model.getFrameId(frame_name)


def _frame_position_expr(
    cmodel: cpin.Model,
    q: ca.SX,
    frame_id: int,
) -> ca.SX:
    """
    Build symbolic frame position expression.

    A fresh cdata is created for each expression to avoid symbolic graph
    interference when building multiple functions.
    """
    cdata = cmodel.createData()

    cpin.forwardKinematics(cmodel, cdata, q)
    cpin.updateFramePlacements(cmodel, cdata)

    placement = cdata.oMf[frame_id]

    return placement.translation


def build_casadi_pinocchio_kinematics(
    model: pin.Model,
    *,
    left_frame_name: str = LEFT_EE_SITE_NAME,
    right_frame_name: str = RIGHT_EE_SITE_NAME,
    name_prefix: str = "pinocchio",
) -> CasadiPinocchioKinematics:
    """
    Build CasADi symbolic dual-EE kinematics functions.

    Parameters
    ----------
    model:
        Numeric Pinocchio model.

    left_frame_name, right_frame_name:
        EE frame names in the Pinocchio model.

    name_prefix:
        Prefix for CasADi function names.

    Returns
    -------
    kin:
        CasadiPinocchioKinematics object.
    """
    _validate_fixed_base_model(model)

    left_frame_id = _get_frame_id(model, left_frame_name)
    right_frame_id = _get_frame_id(model, right_frame_name)

    cmodel = cpin.Model(model)

    nq = model.nq
    nv = model.nv
    nx = nq + nv

    q = sx_vector("q", nq)
    x = sx_vector("x", nx)

    q_x = x[:nq]

    # -------------------------------------------------------------------------
    # q-based EE position functions
    # -------------------------------------------------------------------------
    p_left = _frame_position_expr(
        cmodel,
        q,
        left_frame_id,
    )

    p_right = _frame_position_expr(
        cmodel,
        q,
        right_frame_id,
    )

    p_pair = ca.vertcat(
        p_left,
        p_right,
    )

    left_ee_position_fn = ca.Function(
        f"{name_prefix}_left_ee_position",
        [q],
        [p_left],
        ["q"],
        ["p_left"],
    )

    right_ee_position_fn = ca.Function(
        f"{name_prefix}_right_ee_position",
        [q],
        [p_right],
        ["q"],
        ["p_right"],
    )

    dual_ee_position_fn = ca.Function(
        f"{name_prefix}_dual_ee_position",
        [q],
        [p_pair],
        ["q"],
        ["p_pair"],
    )

    # -------------------------------------------------------------------------
    # x-based EE position function
    # Useful inside OCP because decision variable is x = [q; v].
    # -------------------------------------------------------------------------
    p_left_x = _frame_position_expr(
        cmodel,
        q_x,
        left_frame_id,
    )

    p_right_x = _frame_position_expr(
        cmodel,
        q_x,
        right_frame_id,
    )

    p_pair_x = ca.vertcat(
        p_left_x,
        p_right_x,
    )

    dual_ee_position_x_fn = ca.Function(
        f"{name_prefix}_dual_ee_position_x",
        [x],
        [p_pair_x],
        ["x"],
        ["p_pair"],
    )

    return CasadiPinocchioKinematics(
        model=model,
        cmodel=cmodel,
        nq=nq,
        nv=nv,
        nx=nx,
        left_frame_name=left_frame_name,
        right_frame_name=right_frame_name,
        left_frame_id=left_frame_id,
        right_frame_id=right_frame_id,
        left_ee_position_fn=left_ee_position_fn,
        right_ee_position_fn=right_ee_position_fn,
        dual_ee_position_fn=dual_ee_position_fn,
        dual_ee_position_x_fn=dual_ee_position_x_fn,
    )