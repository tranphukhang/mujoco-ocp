from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import mujoco
import pinocchio as pin

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
# Pinocchio/OCP backend:
#   - g1_fixed_base.xml
#   - 17 DoF: waist + left arm + right arm
#   - used for FK, Jacobian, IK, dynamics, OCP
#
# MuJoCo replay backend:
#   - G1_with_gripper.xml
#   - 29 qpos: 17 G1 joints + 12 Robotiq internal joints
#   - 19 actuators: 17 G1 position actuators + 2 gripper actuators
#   - used only for viewer/replay/checking gripper behavior

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
# The Pinocchio model contains exactly these 17 generalized coordinates.
# Pinocchio also has the internal "universe" joint, but it is not part of q.

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
# Robotiq gripper names in MuJoCo replay model
# =============================================================================
# These joints/actuators exist only in the MuJoCo replay model.
# They are NOT part of the OCP decision variables.

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
# - These are used for MuJoCo replay/viewer, not as OCP torque inputs.

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
# Gripper command normalization for replay
# =============================================================================
# Gripper is not part of OCP.
# These helpers are kept only for MuJoCo replay/task logic.
#
# MuJoCo Robotiq actuator:
#   ctrlrange = [0, 255]
#
# Normalized command convention:
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

GRIPPER_COMMAND_LIMITS: Mapping[str, tuple[float, float]] = {
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
    Convert normalized gripper command in [0, 1] to MuJoCo ctrl in [0, 255].

    This is for replay/task logic only. Gripper command is not an OCP variable.
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
    Convert MuJoCo gripper ctrl in [0, 255] to normalized command in [0, 1].

    This is for replay/task logic only. Gripper command is not an OCP variable.
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
# OCP coordinates and controls
# =============================================================================
# Gripper is NOT part of the optimization problem.
#
# OCP state:
#   q_ocp = 17 G1 upper-body joint positions
#   v_ocp = 17 G1 upper-body joint velocities
#
# OCP control:
#   u_ocp = 17 G1 joint torques
#
# MuJoCo replay may still close/open the gripper using separate task logic.

OCP_G1_JOINT_NAMES: tuple[str, ...] = PINOCCHIO_JOINT_NAMES

OCP_COORDINATE_NAMES: tuple[str, ...] = OCP_G1_JOINT_NAMES

OCP_DEFAULT_VELOCITY_LIMIT: tuple[float, float] = (-10.0, 10.0)

OCP_VELOCITY_LIMITS: Mapping[str, tuple[float, float]] = {
    name: OCP_DEFAULT_VELOCITY_LIMIT
    for name in OCP_COORDINATE_NAMES
}

OCP_CONTROL_NAMES: tuple[str, ...] = tuple(
    f"{joint_name}_torque"
    for joint_name in OCP_G1_JOINT_NAMES
)

OUTPUT_COORDINATE_NAMES: tuple[str, ...] = OCP_COORDINATE_NAMES

OUTPUT_VELOCITY_NAMES: tuple[str, ...] = tuple(
    f"{name}_velocity"
    for name in OUTPUT_COORDINATE_NAMES
)

# Backward-compatible empty aliases.
# These are intentionally empty because gripper is no longer optimized.
OCP_GRIPPER_COORDINATE_NAMES: tuple[str, ...] = ()
OCP_GRIPPER_EFFORT_NAMES: tuple[str, ...] = ()
OCP_GRIPPER_COORDINATE_LIMITS: Mapping[str, tuple[float, float]] = {}

OCP_G1_TORQUE_NAMES: tuple[str, ...] = OCP_CONTROL_NAMES


# =============================================================================
# Dimensions
# =============================================================================

N_PINOCCHIO_Q = len(PINOCCHIO_JOINT_NAMES)
N_PINOCCHIO_V = len(PINOCCHIO_JOINT_NAMES)

N_G1_ACTIVE_JOINTS = len(ACTIVE_G1_JOINT_NAMES)

# Gripper exists in MuJoCo replay but is not part of OCP.
N_GRIPPER_COORDINATES = 0

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

    # Gripper replay command mapping
    gripper_command_limits: Mapping[str, tuple[float, float]]
    mujoco_gripper_ctrl_limits: Mapping[str, tuple[float, float]]
    gripper_command_to_mujoco_ctrl_scale: float
    gripper_command_to_mujoco_ctrl_offset: float

    # OCP model names
    ocp_g1_joint_names: tuple[str, ...]
    ocp_coordinate_names: tuple[str, ...]
    ocp_control_names: tuple[str, ...]
    ocp_velocity_limits: Mapping[str, tuple[float, float]]

    # Backward-compatible empty fields
    ocp_gripper_coordinate_names: tuple[str, ...]
    ocp_gripper_effort_names: tuple[str, ...]
    ocp_gripper_coordinate_limits: Mapping[str, tuple[float, float]]

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

    gripper_command_limits=GRIPPER_COMMAND_LIMITS,
    mujoco_gripper_ctrl_limits=MUJOCO_GRIPPER_CTRL_LIMITS,
    gripper_command_to_mujoco_ctrl_scale=GRIPPER_COMMAND_TO_MUJOCO_CTRL_SCALE,
    gripper_command_to_mujoco_ctrl_offset=GRIPPER_COMMAND_TO_MUJOCO_CTRL_OFFSET,

    ocp_g1_joint_names=OCP_G1_JOINT_NAMES,
    ocp_coordinate_names=OCP_COORDINATE_NAMES,
    ocp_control_names=OCP_CONTROL_NAMES,
    ocp_velocity_limits=OCP_VELOCITY_LIMITS,

    ocp_gripper_coordinate_names=OCP_GRIPPER_COORDINATE_NAMES,
    ocp_gripper_effort_names=OCP_GRIPPER_EFFORT_NAMES,
    ocp_gripper_coordinate_limits=OCP_GRIPPER_COORDINATE_LIMITS,

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
# Validation
# =============================================================================

def validate_pinocchio_model(model: pin.Model) -> None:
    """
    Validate the Pinocchio upper-body model.

    Expected:
    - 17 generalized coordinates
    - 17 velocities
    - waist + left arm + right arm joints
    - left/right virtual gripper EE frames
    """
    errors: list[str] = []

    if model.nq != NQ_OCP:
        errors.append(f"Expected Pinocchio nq={NQ_OCP}, got {model.nq}")

    if model.nv != NV_OCP:
        errors.append(f"Expected Pinocchio nv={NV_OCP}, got {model.nv}")

    joint_names = set(str(name) for name in model.names)
    frame_names = set(frame.name for frame in model.frames)

    missing_joints = [
        name
        for name in PINOCCHIO_JOINT_NAMES
        if name not in joint_names
    ]

    missing_frames = [
        name
        for name in PINOCCHIO_EE_FRAME_NAMES
        if name not in frame_names
    ]

    if missing_joints:
        errors.append(f"Missing Pinocchio joints: {missing_joints}")

    if missing_frames:
        errors.append(f"Missing Pinocchio EE frames: {missing_frames}")

    if errors:
        raise ValueError("Invalid Pinocchio G1 model:\n" + "\n".join(errors))


def validate_mujoco_replay_model(model: mujoco.MjModel) -> None:
    """
    Validate that the MuJoCo replay model contains all names required by
    the G1 gripper replay configuration.

    This validates G1_with_gripper.xml:
    - 17 active G1 upper-body joints
    - 12 Robotiq internal joints
    - 19 MuJoCo actuators
    - 2 gripper end-effector sites

    It does not validate OCP gripper variables because gripper is not part
    of the OCP decision variables.
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
        errors.append(f"Missing MuJoCo joints: {missing_joints}")

    if missing_actuators:
        errors.append(f"Missing MuJoCo actuators: {missing_actuators}")

    if missing_sites:
        errors.append(f"Missing MuJoCo sites: {missing_sites}")

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