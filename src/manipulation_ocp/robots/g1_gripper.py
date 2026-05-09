from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco

from manipulation_ocp.mujoco.info import (
    get_actuator_names,
    get_joint_names,
    get_site_names,
)
from manipulation_ocp.utils.paths import G1_GRIPPER_XML


# =============================================================================
# Robot asset
# =============================================================================

XML_PATH: Path = G1_GRIPPER_XML


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
# Robotiq gripper internal joints in MuJoCo full model
# =============================================================================
# These joints exist in MuJoCo qpos/qvel, but they are not treated as
# independent OCP coordinates. Each gripper is represented in OCP by one
# reduced opening coordinate.

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

MUJOCO_FULL_JOINT_NAMES: tuple[str, ...] = (
    ACTIVE_G1_JOINT_NAMES
    + GRIPPER_JOINT_NAMES
)


# =============================================================================
# MuJoCo actuator names
# =============================================================================
# Important:
# - These are MuJoCo XML actuator names.
# - For G1, XML actuators are position actuators.
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
# OCP reduced coordinates and controls
# =============================================================================
# OCP uses a reduced manipulation model:
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
# MuJoCo full model still has 29 qpos/qvel because each Robotiq gripper has
# internal linkage joints.

OCP_G1_JOINT_NAMES: tuple[str, ...] = ACTIVE_G1_JOINT_NAMES

OCP_GRIPPER_COORDINATE_NAMES: tuple[str, ...] = (
    "left_gripper_opening",
    "right_gripper_opening",
)

OCP_COORDINATE_NAMES: tuple[str, ...] = (
    OCP_G1_JOINT_NAMES
    + OCP_GRIPPER_COORDINATE_NAMES
)

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
# End-effector sites
# =============================================================================

LEFT_EE_SITE_NAME = "left_2f85_grip_site"
RIGHT_EE_SITE_NAME = "right_2f85_grip_site"

EE_SITE_NAMES: dict[str, str] = {
    "left": LEFT_EE_SITE_NAME,
    "right": RIGHT_EE_SITE_NAME,
}


# =============================================================================
# Dimensions
# =============================================================================

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
    # Asset
    xml_path: Path

    # MuJoCo full model names
    waist_joint_names: tuple[str, ...]
    left_arm_joint_names: tuple[str, ...]
    right_arm_joint_names: tuple[str, ...]
    active_g1_joint_names: tuple[str, ...]
    left_gripper_joint_names: tuple[str, ...]
    right_gripper_joint_names: tuple[str, ...]
    gripper_joint_names: tuple[str, ...]
    mujoco_full_joint_names: tuple[str, ...]

    # MuJoCo actuator names for replay/viewer
    mujoco_position_actuator_names: tuple[str, ...]
    mujoco_gripper_actuator_names: tuple[str, ...]
    mujoco_actuator_names: tuple[str, ...]

    # OCP reduced model names
    ocp_g1_joint_names: tuple[str, ...]
    ocp_gripper_coordinate_names: tuple[str, ...]
    ocp_coordinate_names: tuple[str, ...]
    ocp_g1_torque_names: tuple[str, ...]
    ocp_gripper_effort_names: tuple[str, ...]
    ocp_control_names: tuple[str, ...]

    # Output trajectory names
    output_coordinate_names: tuple[str, ...]
    output_velocity_names: tuple[str, ...]

    # End-effector sites
    ee_site_names: dict[str, str]

    # Dimensions
    nq_ocp: int
    nv_ocp: int
    nu_ocp: int
    n_mujoco_actuators: int


G1_GRIPPER_CONFIG = G1GripperConfig(
    xml_path=XML_PATH,

    waist_joint_names=WAIST_JOINT_NAMES,
    left_arm_joint_names=LEFT_ARM_JOINT_NAMES,
    right_arm_joint_names=RIGHT_ARM_JOINT_NAMES,
    active_g1_joint_names=ACTIVE_G1_JOINT_NAMES,
    left_gripper_joint_names=LEFT_GRIPPER_JOINT_NAMES,
    right_gripper_joint_names=RIGHT_GRIPPER_JOINT_NAMES,
    gripper_joint_names=GRIPPER_JOINT_NAMES,
    mujoco_full_joint_names=MUJOCO_FULL_JOINT_NAMES,

    mujoco_position_actuator_names=MUJOCO_POSITION_ACTUATOR_NAMES,
    mujoco_gripper_actuator_names=MUJOCO_GRIPPER_ACTUATOR_NAMES,
    mujoco_actuator_names=MUJOCO_ACTUATOR_NAMES,

    ocp_g1_joint_names=OCP_G1_JOINT_NAMES,
    ocp_gripper_coordinate_names=OCP_GRIPPER_COORDINATE_NAMES,
    ocp_coordinate_names=OCP_COORDINATE_NAMES,
    ocp_g1_torque_names=OCP_G1_TORQUE_NAMES,
    ocp_gripper_effort_names=OCP_GRIPPER_EFFORT_NAMES,
    ocp_control_names=OCP_CONTROL_NAMES,

    output_coordinate_names=OUTPUT_COORDINATE_NAMES,
    output_velocity_names=OUTPUT_VELOCITY_NAMES,

    ee_site_names=EE_SITE_NAMES,

    nq_ocp=NQ_OCP,
    nv_ocp=NV_OCP,
    nu_ocp=NU_OCP,
    n_mujoco_actuators=N_MUJOCO_ACTUATORS,
)


# =============================================================================
# Validation
# =============================================================================

def validate_g1_gripper_model(model: mujoco.MjModel) -> None:
    """
    Validate that the loaded MuJoCo model contains all names required by
    the G1 gripper configuration.

    This validates the MuJoCo full model:
    - 17 active G1 joints
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
        for name in MUJOCO_FULL_JOINT_NAMES
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
        raise ValueError("Invalid G1 gripper model:\n" + "\n".join(errors))


def get_g1_gripper_config() -> G1GripperConfig:
    """Return the default G1 gripper robot configuration."""
    return G1_GRIPPER_CONFIG