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


# ---------------------------------------------------------------------
# Robot asset
# ---------------------------------------------------------------------

XML_PATH: Path = G1_GRIPPER_XML


# ---------------------------------------------------------------------
# G1 upper-body joints
# ---------------------------------------------------------------------

WAIST_JOINT_NAMES: list[str] = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
]

LEFT_ARM_JOINT_NAMES: list[str] = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]

RIGHT_ARM_JOINT_NAMES: list[str] = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

ACTIVE_G1_JOINT_NAMES: list[str] = (
    WAIST_JOINT_NAMES
    + LEFT_ARM_JOINT_NAMES
    + RIGHT_ARM_JOINT_NAMES
)


# ---------------------------------------------------------------------
# Robotiq gripper joints
# ---------------------------------------------------------------------
# These joints are part of qpos/qvel, but they are not independently
# controlled by the OCP. Each gripper is controlled by one actuator.

LEFT_GRIPPER_JOINT_NAMES: list[str] = [
    "left_2f85_left_driver_joint",
    "left_2f85_left_spring_link_joint",
    "left_2f85_left_follower",
    "left_2f85_right_driver_joint",
    "left_2f85_right_spring_link_joint",
    "left_2f85_right_follower_joint",
]

RIGHT_GRIPPER_JOINT_NAMES: list[str] = [
    "right_2f85_left_driver_joint",
    "right_2f85_left_spring_link_joint",
    "right_2f85_left_follower",
    "right_2f85_right_driver_joint",
    "right_2f85_right_spring_link_joint",
    "right_2f85_right_follower_joint",
]

GRIPPER_JOINT_NAMES: list[str] = (
    LEFT_GRIPPER_JOINT_NAMES
    + RIGHT_GRIPPER_JOINT_NAMES
)


# ---------------------------------------------------------------------
# Actuators
# ---------------------------------------------------------------------

G1_ACTUATOR_NAMES: list[str] = ACTIVE_G1_JOINT_NAMES.copy()

GRIPPER_ACTUATOR_NAMES: list[str] = [
    "left_2f85_fingers_actuator",
    "right_2f85_fingers_actuator",
]

ACTIVE_ACTUATOR_NAMES: list[str] = (
    G1_ACTUATOR_NAMES
    + GRIPPER_ACTUATOR_NAMES
)


# ---------------------------------------------------------------------
# End-effector sites
# ---------------------------------------------------------------------

LEFT_EE_SITE_NAME = "left_2f85_grip_site"
RIGHT_EE_SITE_NAME = "right_2f85_grip_site"

EE_SITE_NAMES: dict[str, str] = {
    "left": LEFT_EE_SITE_NAME,
    "right": RIGHT_EE_SITE_NAME,
}


@dataclass(frozen=True)
class G1GripperConfig:
    xml_path: Path
    waist_joint_names: list[str]
    left_arm_joint_names: list[str]
    right_arm_joint_names: list[str]
    active_g1_joint_names: list[str]
    left_gripper_joint_names: list[str]
    right_gripper_joint_names: list[str]
    gripper_joint_names: list[str]
    g1_actuator_names: list[str]
    gripper_actuator_names: list[str]
    active_actuator_names: list[str]
    ee_site_names: dict[str, str]


G1_GRIPPER_CONFIG = G1GripperConfig(
    xml_path=XML_PATH,
    waist_joint_names=WAIST_JOINT_NAMES,
    left_arm_joint_names=LEFT_ARM_JOINT_NAMES,
    right_arm_joint_names=RIGHT_ARM_JOINT_NAMES,
    active_g1_joint_names=ACTIVE_G1_JOINT_NAMES,
    left_gripper_joint_names=LEFT_GRIPPER_JOINT_NAMES,
    right_gripper_joint_names=RIGHT_GRIPPER_JOINT_NAMES,
    gripper_joint_names=GRIPPER_JOINT_NAMES,
    g1_actuator_names=G1_ACTUATOR_NAMES,
    gripper_actuator_names=GRIPPER_ACTUATOR_NAMES,
    active_actuator_names=ACTIVE_ACTUATOR_NAMES,
    ee_site_names=EE_SITE_NAMES,
)


def validate_g1_gripper_model(model: mujoco.MjModel) -> None:
    """
    Validate that the loaded MuJoCo model contains all names required by
    the G1 gripper configuration.
    """
    joint_names = set(get_joint_names(model))
    actuator_names = set(get_actuator_names(model))
    site_names = set(get_site_names(model))

    missing_joints = [
        name
        for name in ACTIVE_G1_JOINT_NAMES + GRIPPER_JOINT_NAMES
        if name not in joint_names
    ]

    missing_actuators = [
        name
        for name in ACTIVE_ACTUATOR_NAMES
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