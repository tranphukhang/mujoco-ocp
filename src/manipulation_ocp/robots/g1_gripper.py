from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import mujoco

from manipulation_ocp.mujoco.info import (
    get_actuator_names,
    get_joint_names,
    get_site_names,
)
from manipulation_ocp.utils.paths import (
    G1_FIXED_BASE_XML,
    G1_GRIPPER_XML,
)


# =============================================================================
# Robot assets
# =============================================================================
# Pinocchio backend:
#   - uses upper-body fixed-base G1 model
#   - 17 DoF: waist + two arms
#   - used for FK, Jacobian, IK, dynamics, OCP
#
# MuJoCo replay backend:
#   - uses G1 + Robotiq grippers model
#   - used for viewer/replay and checking gripper open/close behavior

PINOCCHIO_MODEL_XML: Path = G1_FIXED_BASE_XML
MUJOCO_REPLAY_XML: Path = G1_GRIPPER_XML

# Backward-compatible alias.
# Prefer PINOCCHIO_MODEL_XML or MUJOCO_REPLAY_XML in new code.
XML_PATH: Path = MUJOCO_REPLAY_XML


# =============================================================================
# G1 upper-body joints
# =============================================================================

WAIST_JOINT_NAMES: tuple[str, ...] = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
)

LEFT_ARM_JOINT_NAMES: tuple[str, ...] = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)

RIGHT_ARM_JOINT_NAMES: tuple[str, ...] = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

ACTIVE_G1_JOINT_NAMES: tuple[str, ...] = (
    WAIST_JOINT_NAMES
    + LEFT_ARM_JOINT_NAMES
    + RIGHT_ARM_JOINT_NAMES
)


# =============================================================================
# Pinocchio model names
# =============================================================================
# The Pinocchio model is built from g1_fixed_base.xml after removing leg joints.
# It contains exactly these 17 joints.
#
# Note:
#   Pinocchio also has the "universe" joint internally, but it is not part of
#   the actuated generalized coordinates.

PINOCCHIO_JOINT_NAMES: tuple[str, ...] = ACTIVE_G1_JOINT_NAMES

LEFT_EE_SITE_NAME = "left_2f85_grip_site"
RIGHT_EE_SITE_NAME = "right_2f85_grip_site"

EE_SITE_NAMES: Mapping[str, str] = {
    "left": LEFT_EE_SITE_NAME,
    "right": RIGHT_EE_SITE_NAME,
}

PINOCCHIO_EE_FRAME_NAMES: tuple[str, ...] = (
    LEFT_EE_SITE_NAME,
    RIGHT_EE_SITE_NAME,
)


# =============================================================================
# Robotiq gripper internal joints in MuJoCo replay model
# =============================================================================
# These joints exist only in the MuJoCo replay model qpos/qvel.
# They are not part of the Pinocchio model.
#
# In the OCP, each gripper is represented by one normalized command coordinate:
#   left_gripper_opening  in [0, 1]
#   right_gripper_opening in [0, 1]

LEFT_GRIPPER_JOINT_NAMES: tuple[str, ...] = (
    "left_2f85_left_driver_joint",
    "left_2f85_left_spring_link_joint",
    "left_2f85_left_follower",
    "left_2f85_right_driver_joint",
    "left_2f85_right_spring_link_joint",
    "left_2f85_right_follower_joint",
)

RIGHT_GRIPPER_JOINT_NAMES: tuple[str, ...] = (
    "right_2f85_left_driver_joint",
    "right_2f85_left_spring_link_joint",
    "right_2f85_left_follower",
    "right_2f85_right_driver_joint",
    "right_2f85_right_spring_link_joint",
    "right_2f85_right_follower_joint",
)

GRIPPER_JOINT_NAMES: tuple[str, ...] = (
    LEFT_GRIPPER_JOINT_NAMES
    + RIGHT_GRIPPER_JOINT_NAMES
)

MUJOCO_REPLAY_JOINT_NAMES: tuple[str, ...] = (
    ACTIVE_G1_JOINT_NAMES
    + GRIPPER_JOINT_NAMES
)

# Backward-compatible alias.
MUJOCO_FULL_JOINT_NAMES: tuple[str, ...] = MUJOCO_REPLAY_JOINT_NAMES


# =============================================================================
# MuJoCo actuator names
# =============================================================================
# Important:
# - These are MuJoCo XML actuator names.
# - For G1, XML actuators are position actuators.
# - For Robotiq, XML actuator ctrl is a command range [0, 255].
# - These are used for MuJoCo replay/viewer, not directly as OCP torque inputs.

MUJOCO_POSITION_ACTUATOR_NAMES: tuple[str, ...] = ACTIVE_G1_JOINT_NAMES

MUJOCO_GRIPPER_ACTUATOR_NAMES: tuple[str, ...] = (
    "left_2f85_fingers_actuator",
    "right_2f85_fingers_actuator",
)

MUJOCO_ACTUATOR_NAMES: tuple[str, ...] = (
    MUJOCO_POSITION_ACTUATOR_NAMES
    + MUJOCO_GRIPPER_ACTUATOR_NAMES
)


# =============================================================================
# Gripper command normalization
# =============================================================================
# MuJoCo Robotiq actuator:
#   ctrlrange = [0, 255]
#
# OCP reduced gripper coordinate:
#   left_gripper_opening, right_gripper_opening in [0, 1]
#
# Convention:
#   0.0 = open / minimum command
#   1.0 = close / maximum command

GRIPPER_COMMAND_MIN = 0.0
GRIPPER_COMMAND_MAX = 1.0

MUJOCO_GRIPPER_CTRL_MIN = 0.0
MUJOCO_GRIPPER_CTRL_MAX = 255.0

GRIPPER_COMMAND_TO_MUJOCO_CTRL_SCALE = (
    MUJOCO_GRIPPER_CTRL_MAX - MUJOCO_GRIPPER_CTRL_MIN
)

GRIPPER_COMMAND_TO_MUJOCO_CTRL_OFFSET = MUJOCO_GRIPPER_CTRL_MIN

OCP_GRIPPER_COORDINATE_LIMITS: Mapping[str, tuple[float, float]] = {
    "left_gripper_opening": (GRIPPER_COMMAND_MIN, GRIPPER_COMMAND_MAX),
    "right_gripper_opening": (GRIPPER_COMMAND_MIN, GRIPPER_COMMAND_MAX),
}

MUJOCO_GRIPPER_CTRL_LIMITS: Mapping[str, tuple[float, float]] = {
    "left_2f85_fingers_actuator": (
        MUJOCO_GRIPPER_CTRL_MIN,
        MUJOCO_GRIPPER_CTRL_MAX,
    ),
    "right_2f85_fingers_actuator": (
        MUJOCO_GRIPPER_CTRL_MIN,
        MUJOCO_GRIPPER_CTRL_MAX,
    ),
}


def gripper_command_to_mujoco_ctrl(command: float) -> float:
    """
    Convert normalized OCP gripper command in [0, 1] to MuJoCo ctrl in [0, 255].
    """
    if command < GRIPPER_COMMAND_MIN or command > GRIPPER_COMMAND_MAX:
        raise ValueError(
            f"Gripper command must be in "
            f"[{GRIPPER_COMMAND_MIN}, {GRIPPER_COMMAND_MAX}], got {command}"
        )

    return (
        GRIPPER_COMMAND_TO_MUJOCO_CTRL_OFFSET
        + GRIPPER_COMMAND_TO_MUJOCO_CTRL_SCALE * command
    )


def mujoco_ctrl_to_gripper_command(ctrl: float) -> float:
    """
    Convert MuJoCo gripper ctrl in [0, 255] to normalized OCP command in [0, 1].
    """
    if ctrl < MUJOCO_GRIPPER_CTRL_MIN or ctrl > MUJOCO_GRIPPER_CTRL_MAX:
        raise ValueError(
            f"MuJoCo gripper ctrl must be in "
            f"[{MUJOCO_GRIPPER_CTRL_MIN}, {MUJOCO_GRIPPER_CTRL_MAX}], got {ctrl}"
        )

    return (
        (ctrl - GRIPPER_COMMAND_TO_MUJOCO_CTRL_OFFSET)
        / GRIPPER_COMMAND_TO_MUJOCO_CTRL_SCALE
    )


# =============================================================================
# OCP reduced coordinates and controls
# =============================================================================
# OCP reduced model:
#
#   q_ocp = [
#       17 G1 upper-body joint positions,
#       left_gripper_opening,
#       right_gripper_opening,
#   ]
#
#   v_ocp = [
#       17 G1 upper-body joint velocities,
#       left_gripper_opening_velocity,
#       right_gripper_opening_velocity,
#   ]
#
#   u_ocp = [
#       17 G1 joint torques,
#       left_gripper_effort,
#       right_gripper_effort,
#   ]
#
# Pinocchio handles only the first 17 G1 coordinates.
# The last 2 gripper coordinates are scalar OCP variables and are mapped to
# MuJoCo gripper commands during replay/checking.

OCP_G1_JOINT_NAMES: tuple[str, ...] = PINOCCHIO_JOINT_NAMES

OCP_GRIPPER_COORDINATE_NAMES: tuple[str, ...] = (
    "left_gripper_opening",
    "right_gripper_opening",
)

OCP_COORDINATE_NAMES: tuple[str, ...] = (
    OCP_G1_JOINT_NAMES
    + OCP_GRIPPER_COORDINATE_NAMES
)

OCP_DEFAULT_VELOCITY_LIMIT: tuple[float, float] = (-10.0, 10.0)

OCP_VELOCITY_LIMITS: Mapping[str, tuple[float, float]] = {
    name: OCP_DEFAULT_VELOCITY_LIMIT
    for name in OCP_COORDINATE_NAMES
}

OCP_G1_TORQUE_NAMES: tuple[str, ...] = tuple(
    f"{joint_name}_torque"
    for joint_name in OCP_G1_JOINT_NAMES
)

OCP_GRIPPER_EFFORT_NAMES: tuple[str, ...] = (
    "left_gripper_effort",
    "right_gripper_effort",
)

OCP_CONTROL_NAMES: tuple[str, ...] = (
    OCP_G1_TORQUE_NAMES
    + OCP_GRIPPER_EFFORT_NAMES
)

OUTPUT_COORDINATE_NAMES: tuple[str, ...] = OCP_COORDINATE_NAMES

OUTPUT_VELOCITY_NAMES: tuple[str, ...] = tuple(
    f"{name}_velocity"
    for name in OUTPUT_COORDINATE_NAMES
)


# =============================================================================
# Dimensions
# =============================================================================

N_PINOCCHIO_Q = len(PINOCCHIO_JOINT_NAMES)
N_PINOCCHIO_V = len(PINOCCHIO_JOINT_NAMES)

N_G1_ACTIVE_JOINTS = len(ACTIVE_G1_JOINT_NAMES)
N_GRIPPER_COORDINATES = len(OCP_GRIPPER_COORDINATE_NAMES)

NQ_OCP = len(OCP_COORDINATE_NAMES)
NV_OCP = len(OCP_COORDINATE_NAMES)
NU_OCP = len(OCP_CONTROL_NAMES)

N_MUJOCO_ACTUATORS = len(MUJOCO_ACTUATOR_NAMES)


# =============================================================================
# Config dataclass
# =============================================================================

@dataclass(frozen=True)
class G1GripperConfig:
    # Assets
    pinocchio_model_xml: Path
    mujoco_replay_xml: Path

    # Backward-compatible alias.
    # Prefer pinocchio_model_xml or mujoco_replay_xml in new code.
    xml_path: Path

    # G1 upper-body groups
    waist_joint_names: tuple[str, ...]
    left_arm_joint_names: tuple[str, ...]
    right_arm_joint_names: tuple[str, ...]
    active_g1_joint_names: tuple[str, ...]

    # Pinocchio model names
    pinocchio_joint_names: tuple[str, ...]
    pinocchio_ee_frame_names: tuple[str, ...]

    # MuJoCo replay model names
    left_gripper_joint_names: tuple[str, ...]
    right_gripper_joint_names: tuple[str, ...]
    gripper_joint_names: tuple[str, ...]
    mujoco_replay_joint_names: tuple[str, ...]

    # MuJoCo actuator names for replay/viewer
    mujoco_position_actuator_names: tuple[str, ...]
    mujoco_gripper_actuator_names: tuple[str, ...]
    mujoco_actuator_names: tuple[str, ...]

    # Gripper command mapping
    ocp_gripper_coordinate_limits: Mapping[str, tuple[float, float]]
    mujoco_gripper_ctrl_limits: Mapping[str, tuple[float, float]]
    gripper_command_to_mujoco_ctrl_scale: float
    gripper_command_to_mujoco_ctrl_offset: float

    # OCP reduced model names
    ocp_g1_joint_names: tuple[str, ...]
    ocp_gripper_coordinate_names: tuple[str, ...]
    ocp_coordinate_names: tuple[str, ...]
    ocp_g1_torque_names: tuple[str, ...]
    ocp_gripper_effort_names: tuple[str, ...]
    ocp_control_names: tuple[str, ...]
    ocp_velocity_limits: Mapping[str, tuple[float, float]]

    # Output trajectory names
    output_coordinate_names: tuple[str, ...]
    output_velocity_names: tuple[str, ...]

    # End-effector frame/site names
    ee_site_names: Mapping[str, str]

    # Dimensions
    n_pinocchio_q: int
    n_pinocchio_v: int
    nq_ocp: int
    nv_ocp: int
    nu_ocp: int
    n_mujoco_actuators: int


G1_GRIPPER_CONFIG = G1GripperConfig(
    pinocchio_model_xml=PINOCCHIO_MODEL_XML,
    mujoco_replay_xml=MUJOCO_REPLAY_XML,
    xml_path=XML_PATH,

    waist_joint_names=WAIST_JOINT_NAMES,
    left_arm_joint_names=LEFT_ARM_JOINT_NAMES,
    right_arm_joint_names=RIGHT_ARM_JOINT_NAMES,
    active_g1_joint_names=ACTIVE_G1_JOINT_NAMES,

    pinocchio_joint_names=PINOCCHIO_JOINT_NAMES,
    pinocchio_ee_frame_names=PINOCCHIO_EE_FRAME_NAMES,

    left_gripper_joint_names=LEFT_GRIPPER_JOINT_NAMES,
    right_gripper_joint_names=RIGHT_GRIPPER_JOINT_NAMES,
    gripper_joint_names=GRIPPER_JOINT_NAMES,
    mujoco_replay_joint_names=MUJOCO_REPLAY_JOINT_NAMES,

    mujoco_position_actuator_names=MUJOCO_POSITION_ACTUATOR_NAMES,
    mujoco_gripper_actuator_names=MUJOCO_GRIPPER_ACTUATOR_NAMES,
    mujoco_actuator_names=MUJOCO_ACTUATOR_NAMES,

    ocp_gripper_coordinate_limits=OCP_GRIPPER_COORDINATE_LIMITS,
    mujoco_gripper_ctrl_limits=MUJOCO_GRIPPER_CTRL_LIMITS,
    gripper_command_to_mujoco_ctrl_scale=GRIPPER_COMMAND_TO_MUJOCO_CTRL_SCALE,
    gripper_command_to_mujoco_ctrl_offset=GRIPPER_COMMAND_TO_MUJOCO_CTRL_OFFSET,

    ocp_g1_joint_names=OCP_G1_JOINT_NAMES,
    ocp_gripper_coordinate_names=OCP_GRIPPER_COORDINATE_NAMES,
    ocp_coordinate_names=OCP_COORDINATE_NAMES,
    ocp_g1_torque_names=OCP_G1_TORQUE_NAMES,
    ocp_gripper_effort_names=OCP_GRIPPER_EFFORT_NAMES,
    ocp_control_names=OCP_CONTROL_NAMES,
    ocp_velocity_limits=OCP_VELOCITY_LIMITS,

    output_coordinate_names=OUTPUT_COORDINATE_NAMES,
    output_velocity_names=OUTPUT_VELOCITY_NAMES,

    ee_site_names=EE_SITE_NAMES,

    n_pinocchio_q=N_PINOCCHIO_Q,
    n_pinocchio_v=N_PINOCCHIO_V,
    nq_ocp=NQ_OCP,
    nv_ocp=NV_OCP,
    nu_ocp=NU_OCP,
    n_mujoco_actuators=N_MUJOCO_ACTUATORS,
)


# =============================================================================
# MuJoCo replay model validation
# =============================================================================

def validate_mujoco_replay_model(model: mujoco.MjModel) -> None:
    """
    Validate that the MuJoCo replay model contains all names required by
    the G1 gripper configuration.

    This validates the G1_with_gripper.xml model:
    - 17 active G1 upper-body joints
    - 12 Robotiq internal joints
    - 19 MuJoCo actuators
    - 2 gripper end-effector sites

    It does not validate OCP reduced coordinate names because names such as
    'left_gripper_opening' and 'right_gripper_opening' are conceptual OCP
    coordinates, not MuJoCo joint names.
    """
    joint_names = set(get_joint_names(model))
    actuator_names = set(get_actuator_names(model))
    site_names = set(get_site_names(model))

    missing_joints = [
        name
        for name in MUJOCO_REPLAY_JOINT_NAMES
        if name not in joint_names
    ]

    missing_actuators = [
        name
        for name in MUJOCO_ACTUATOR_NAMES
        if name not in actuator_names
    ]

    missing_sites = [
        name
        for name in EE_SITE_NAMES.values()
        if name not in site_names
    ]

    errors: list[str] = []

    if missing_joints:
        errors.append(f"Missing joints: {missing_joints}")

    if missing_actuators:
        errors.append(f"Missing actuators: {missing_actuators}")

    if missing_sites:
        errors.append(f"Missing sites: {missing_sites}")

    if errors:
        raise ValueError("Invalid MuJoCo replay model:\n" + "\n".join(errors))


# Backward-compatible alias.
def validate_g1_gripper_model(model: mujoco.MjModel) -> None:
    """
    Validate MuJoCo replay model.

    Deprecated name kept for compatibility. Prefer validate_mujoco_replay_model.
    """
    validate_mujoco_replay_model(model)


def get_g1_gripper_config() -> G1GripperConfig:
    """Return the default G1 gripper robot configuration."""
    return G1_GRIPPER_CONFIG