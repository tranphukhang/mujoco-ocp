from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import mujoco
import numpy as np


Limit = tuple[float, float]


@dataclass(frozen=True)
class JointLimit:
    """Limits and addresses for one MuJoCo joint."""

    name: str
    joint_id: int
    qpos_id: int
    dof_id: int
    position: Limit
    velocity: Limit
    actuator_force: Limit


@dataclass(frozen=True)
class ActuatorLimit:
    """Limits for one MuJoCo actuator."""

    name: str
    actuator_id: int
    ctrl: Limit
    force: Limit


@dataclass(frozen=True)
class OcpBounds:
    """
    Bounds for the reduced OCP model.

    q bounds:
        joint position limits for G1 + normalized gripper command limits [0, 1]

    v bounds:
        velocity limits. By default this is [-inf, inf] until robot-specific
        velocity limits are defined.

    u bounds:
        torque limits for G1 joints + effort limits for grippers.
    """

    coordinate_names: tuple[str, ...]
    control_names: tuple[str, ...]

    q_lower: np.ndarray
    q_upper: np.ndarray

    v_lower: np.ndarray
    v_upper: np.ndarray

    u_lower: np.ndarray
    u_upper: np.ndarray


def get_joint_id(model: mujoco.MjModel, joint_name: str) -> int:
    """Return MuJoCo joint id from joint name."""
    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name,
    )

    if joint_id == -1:
        raise ValueError(f"Joint not found: {joint_name}")

    return joint_id


def get_actuator_id(model: mujoco.MjModel, actuator_name: str) -> int:
    """Return MuJoCo actuator id from actuator name."""
    actuator_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        actuator_name,
    )

    if actuator_id == -1:
        raise ValueError(f"Actuator not found: {actuator_name}")

    return actuator_id


def get_joint_qpos_addr(model: mujoco.MjModel, joint_name: str) -> int:
    """Return qpos address of a scalar joint."""
    joint_id = get_joint_id(model, joint_name)
    return int(model.jnt_qposadr[joint_id])


def get_joint_dof_addr(model: mujoco.MjModel, joint_name: str) -> int:
    """Return qvel/dof address of a scalar joint."""
    joint_id = get_joint_id(model, joint_name)
    return int(model.jnt_dofadr[joint_id])


def get_joint_position_limit(
    model: mujoco.MjModel,
    joint_name: str,
    *,
    default: Limit = (-np.inf, np.inf),
) -> Limit:
    """
    Return joint position limit.

    If the joint is not limited, return default.
    """
    joint_id = get_joint_id(model, joint_name)

    if hasattr(model, "jnt_limited") and not bool(model.jnt_limited[joint_id]):
        return default

    lower, upper = model.jnt_range[joint_id]
    return float(lower), float(upper)


def get_joint_velocity_limit(
    model: mujoco.MjModel,
    joint_name: str,
    *,
    default: Limit = (-np.inf, np.inf),
) -> Limit:
    """
    Return joint velocity limit.

    MuJoCo joint range is position range, not velocity range.
    For now, velocity limits are robot/OCP-specific and should be set later.
    """
    _ = get_joint_id(model, joint_name)
    return default


def get_joint_actuator_force_limit(
    model: mujoco.MjModel,
    joint_name: str,
    *,
    default: Limit = (-np.inf, np.inf),
) -> Limit:
    """
    Return joint actuator force limit if available.

    For hinge joints, this is interpreted as a torque limit.
    It corresponds to joint-level actuatorfrcrange in MuJoCo XML.
    """
    joint_id = get_joint_id(model, joint_name)

    if not hasattr(model, "jnt_actfrcrange"):
        return default

    if hasattr(model, "jnt_actfrclimited") and not bool(
        model.jnt_actfrclimited[joint_id]
    ):
        return default

    lower, upper = model.jnt_actfrcrange[joint_id]
    return float(lower), float(upper)


def get_joint_limit(
    model: mujoco.MjModel,
    joint_name: str,
    *,
    velocity_default: Limit = (-np.inf, np.inf),
    actuator_force_default: Limit = (-np.inf, np.inf),
) -> JointLimit:
    """Return all available limit information for one joint."""
    joint_id = get_joint_id(model, joint_name)

    return JointLimit(
        name=joint_name,
        joint_id=joint_id,
        qpos_id=int(model.jnt_qposadr[joint_id]),
        dof_id=int(model.jnt_dofadr[joint_id]),
        position=get_joint_position_limit(model, joint_name),
        velocity=get_joint_velocity_limit(
            model,
            joint_name,
            default=velocity_default,
        ),
        actuator_force=get_joint_actuator_force_limit(
            model,
            joint_name,
            default=actuator_force_default,
        ),
    )


def get_joint_limits(
    model: mujoco.MjModel,
    joint_names: Sequence[str],
    *,
    velocity_default: Limit = (-np.inf, np.inf),
    actuator_force_default: Limit = (-np.inf, np.inf),
) -> dict[str, JointLimit]:
    """Return limits for multiple joints."""
    return {
        joint_name: get_joint_limit(
            model,
            joint_name,
            velocity_default=velocity_default,
            actuator_force_default=actuator_force_default,
        )
        for joint_name in joint_names
    }


def get_actuator_ctrl_limit(
    model: mujoco.MjModel,
    actuator_name: str,
    *,
    default: Limit = (-np.inf, np.inf),
) -> Limit:
    """
    Return actuator control range.

    For G1 position actuators, this is the MuJoCo replay/viewer ctrl range.
    It is not the OCP torque limit.

    For Robotiq, this is the command range [0, 255], not a joint position.
    """
    actuator_id = get_actuator_id(model, actuator_name)

    if hasattr(model, "actuator_ctrllimited") and not bool(
        model.actuator_ctrllimited[actuator_id]
    ):
        return default

    lower, upper = model.actuator_ctrlrange[actuator_id]
    return float(lower), float(upper)


def get_actuator_force_limit(
    model: mujoco.MjModel,
    actuator_name: str,
    *,
    default: Limit = (-np.inf, np.inf),
) -> Limit:
    """
    Return actuator force range if available.

    For Robotiq gripper actuator, this is the effort/force bound used by OCP.
    """
    actuator_id = get_actuator_id(model, actuator_name)

    if hasattr(model, "actuator_forcelimited") and not bool(
        model.actuator_forcelimited[actuator_id]
    ):
        return default

    lower, upper = model.actuator_forcerange[actuator_id]
    return float(lower), float(upper)


def get_actuator_limit(
    model: mujoco.MjModel,
    actuator_name: str,
) -> ActuatorLimit:
    """Return ctrl and force limits for one actuator."""
    actuator_id = get_actuator_id(model, actuator_name)

    return ActuatorLimit(
        name=actuator_name,
        actuator_id=actuator_id,
        ctrl=get_actuator_ctrl_limit(model, actuator_name),
        force=get_actuator_force_limit(model, actuator_name),
    )


def get_actuator_limits(
    model: mujoco.MjModel,
    actuator_names: Sequence[str],
) -> dict[str, ActuatorLimit]:
    """Return limits for multiple actuators."""
    return {
        actuator_name: get_actuator_limit(model, actuator_name)
        for actuator_name in actuator_names
    }


def limits_to_arrays(
    limits: Sequence[Limit],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a sequence of (lower, upper) limits to lower/upper arrays."""
    lower = np.array([item[0] for item in limits], dtype=float)
    upper = np.array([item[1] for item in limits], dtype=float)
    return lower, upper


def build_ocp_bounds(
    model: mujoco.MjModel,
    *,
    g1_joint_names: Sequence[str],
    gripper_coordinate_names: Sequence[str],
    control_names: Sequence[str],
    gripper_coordinate_limits: Mapping[str, Limit],
    gripper_effort_actuator_names: Sequence[str],
    velocity_limits: Mapping[str, Limit] | None = None,
    default_velocity_limit: Limit = (-np.inf, np.inf),
) -> OcpBounds:
    """
    Build reduced OCP bounds.

    The reduced coordinate order is:

        q_ocp = [
            G1 joint positions,
            normalized gripper coordinates,
        ]

    The reduced control order is:

        u_ocp = [
            G1 joint torques,
            gripper efforts,
        ]

    G1 torque limits:
        read from joint actuatorfrcrange.

    Gripper effort limits:
        read from actuator forcerange.

    Gripper coordinate limits:
        provided explicitly, usually [0, 1].
    """
    g1_joint_names = tuple(g1_joint_names)
    gripper_coordinate_names = tuple(gripper_coordinate_names)
    control_names = tuple(control_names)
    gripper_effort_actuator_names = tuple(gripper_effort_actuator_names)

    coordinate_names = g1_joint_names + gripper_coordinate_names

    if velocity_limits is None:
        velocity_limits = {}

    # ------------------------------------------------------------------
    # q bounds
    # ------------------------------------------------------------------

    g1_position_limits = [
        get_joint_position_limit(model, joint_name)
        for joint_name in g1_joint_names
    ]

    gripper_position_limits = []
    for name in gripper_coordinate_names:
        if name not in gripper_coordinate_limits:
            raise ValueError(f"Missing gripper coordinate limit: {name}")
        gripper_position_limits.append(gripper_coordinate_limits[name])

    q_lower, q_upper = limits_to_arrays(
        g1_position_limits + gripper_position_limits
    )

    # ------------------------------------------------------------------
    # v bounds
    # ------------------------------------------------------------------

    velocity_limit_list = [
        velocity_limits.get(name, default_velocity_limit)
        for name in coordinate_names
    ]

    v_lower, v_upper = limits_to_arrays(velocity_limit_list)

    # ------------------------------------------------------------------
    # u bounds
    # ------------------------------------------------------------------

    g1_torque_limits = [
        get_joint_actuator_force_limit(model, joint_name)
        for joint_name in g1_joint_names
    ]

    gripper_effort_limits = [
        get_actuator_force_limit(model, actuator_name)
        for actuator_name in gripper_effort_actuator_names
    ]

    u_lower, u_upper = limits_to_arrays(
        g1_torque_limits + gripper_effort_limits
    )

    expected_nu = len(control_names)
    actual_nu = len(u_lower)

    if expected_nu != actual_nu:
        raise ValueError(
            f"Control dimension mismatch: "
            f"len(control_names)={expected_nu}, "
            f"len(u_bounds)={actual_nu}"
        )

    return OcpBounds(
        coordinate_names=coordinate_names,
        control_names=control_names,
        q_lower=q_lower,
        q_upper=q_upper,
        v_lower=v_lower,
        v_upper=v_upper,
        u_lower=u_lower,
        u_upper=u_upper,
    )