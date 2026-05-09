from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pinocchio as pin

from manipulation_ocp.robots.g1_gripper import (
    LEFT_EE_SITE_NAME,
    RIGHT_EE_SITE_NAME,
)


@dataclass(frozen=True)
class FramePose:
    """World pose of a Pinocchio frame."""

    position: np.ndarray  # shape (3,)
    rotation: np.ndarray  # shape (3, 3)


@dataclass(frozen=True)
class FrameJacobian:
    """World-aligned frame Jacobian."""

    linear: np.ndarray   # shape (3, nv)
    angular: np.ndarray  # shape (3, nv)
    full: np.ndarray     # shape (6, nv)


def as_pinocchio_q(
    model: pin.Model,
    q: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """
    Convert input q to a Pinocchio configuration vector.

    Accepted inputs:
    - q with shape (model.nq,)
    - q_ocp with shape larger than model.nq, where the first model.nq
      entries are the G1 upper-body coordinates and the remaining entries
      are non-Pinocchio coordinates such as gripper commands.

    For the current setup:
    - Pinocchio model nq = 17
    - OCP q has 19 coordinates = 17 G1 joints + 2 gripper commands
    """
    q_array = np.asarray(q, dtype=float).reshape(-1)

    if q_array.size < model.nq:
        raise ValueError(
            f"q must have at least model.nq={model.nq} entries, "
            f"got {q_array.size}"
        )

    return q_array[: model.nq].copy()


def get_frame_id(model: pin.Model, frame_name: str) -> int:
    """
    Get Pinocchio frame id from frame name.

    Raises
    ------
    ValueError
        If the frame does not exist.
    """
    if not model.existFrame(frame_name):
        raise ValueError(f"Frame not found: {frame_name}")

    return model.getFrameId(frame_name)


def forward_kinematics(
    model: pin.Model,
    data: pin.Data,
    q: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """
    Run Pinocchio forward kinematics and update frame placements.

    Parameters
    ----------
    model:
        Pinocchio model.

    data:
        Pinocchio data.

    q:
        Configuration vector. Can be either Pinocchio q with shape (17,)
        or OCP q with shape (19,). If q has more than model.nq entries,
        only the first model.nq entries are used.

    Returns
    -------
    q_pin:
        Configuration actually used by Pinocchio, shape (model.nq,).
    """
    q_pin = as_pinocchio_q(model, q)

    pin.forwardKinematics(model, data, q_pin)
    pin.updateFramePlacements(model, data)

    return q_pin


def get_frame_pose(
    model: pin.Model,
    data: pin.Data,
    q: Sequence[float] | np.ndarray,
    frame_name: str,
    *,
    update: bool = True,
) -> FramePose:
    """
    Get world pose of a frame.

    Parameters
    ----------
    frame_name:
        Name of a Pinocchio frame, for example:
        - left_2f85_grip_site
        - right_2f85_grip_site
    """
    if update:
        forward_kinematics(model, data, q)

    frame_id = get_frame_id(model, frame_name)
    placement = data.oMf[frame_id]

    return FramePose(
        position=placement.translation.copy(),
        rotation=placement.rotation.copy(),
    )


def get_frame_position(
    model: pin.Model,
    data: pin.Data,
    q: Sequence[float] | np.ndarray,
    frame_name: str,
    *,
    update: bool = True,
) -> np.ndarray:
    """Get world position of a frame."""
    return get_frame_pose(
        model,
        data,
        q,
        frame_name,
        update=update,
    ).position


def get_frame_rotation(
    model: pin.Model,
    data: pin.Data,
    q: Sequence[float] | np.ndarray,
    frame_name: str,
    *,
    update: bool = True,
) -> np.ndarray:
    """Get world rotation matrix of a frame."""
    return get_frame_pose(
        model,
        data,
        q,
        frame_name,
        update=update,
    ).rotation


def get_frame_jacobian(
    model: pin.Model,
    data: pin.Data,
    q: Sequence[float] | np.ndarray,
    frame_name: str,
    *,
    reference_frame: pin.ReferenceFrame = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
    update: bool = True,
) -> FrameJacobian:
    """
    Get frame Jacobian.

    The default reference frame is LOCAL_WORLD_ALIGNED, which is convenient
    for Cartesian position IK because the linear part is expressed in the
    world-aligned frame.

    Returns
    -------
    FrameJacobian:
        linear:  shape (3, nv)
        angular: shape (3, nv)
        full:    shape (6, nv)
    """
    q_pin = as_pinocchio_q(model, q)
    frame_id = get_frame_id(model, frame_name)

    if update:
        pin.forwardKinematics(model, data, q_pin)
        pin.updateFramePlacements(model, data)

    full_jacobian = pin.computeFrameJacobian(
        model,
        data,
        q_pin,
        frame_id,
        reference_frame,
    )

    return FrameJacobian(
        linear=full_jacobian[:3, :].copy(),
        angular=full_jacobian[3:, :].copy(),
        full=full_jacobian.copy(),
    )


def get_ee_poses(
    model: pin.Model,
    data: pin.Data,
    q: Sequence[float] | np.ndarray,
    *,
    update: bool = True,
) -> dict[str, FramePose]:
    """
    Get world poses of left and right gripper virtual end-effector frames.

    Returns
    -------
    poses:
        {
            "left": FramePose(...),
            "right": FramePose(...),
        }
    """
    if update:
        forward_kinematics(model, data, q)

    return {
        "left": get_frame_pose(
            model,
            data,
            q,
            LEFT_EE_SITE_NAME,
            update=False,
        ),
        "right": get_frame_pose(
            model,
            data,
            q,
            RIGHT_EE_SITE_NAME,
            update=False,
        ),
    }


def get_ee_positions(
    model: pin.Model,
    data: pin.Data,
    q: Sequence[float] | np.ndarray,
    *,
    update: bool = True,
) -> dict[str, np.ndarray]:
    """
    Get world positions of left and right gripper virtual end-effector frames.
    """
    poses = get_ee_poses(model, data, q, update=update)

    return {
        side: pose.position
        for side, pose in poses.items()
    }


def get_ee_jacobians(
    model: pin.Model,
    data: pin.Data,
    q: Sequence[float] | np.ndarray,
    *,
    reference_frame: pin.ReferenceFrame = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
    update: bool = True,
) -> dict[str, FrameJacobian]:
    """
    Get frame Jacobians of left and right gripper virtual end-effector frames.
    """
    if update:
        forward_kinematics(model, data, q)

    return {
        "left": get_frame_jacobian(
            model,
            data,
            q,
            LEFT_EE_SITE_NAME,
            reference_frame=reference_frame,
            update=False,
        ),
        "right": get_frame_jacobian(
            model,
            data,
            q,
            RIGHT_EE_SITE_NAME,
            reference_frame=reference_frame,
            update=False,
        ),
    }


def get_ee_position_jacobians(
    model: pin.Model,
    data: pin.Data,
    q: Sequence[float] | np.ndarray,
    *,
    reference_frame: pin.ReferenceFrame = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
    update: bool = True,
) -> dict[str, np.ndarray]:
    """
    Get linear position Jacobians of left and right gripper frames.
    """
    jacobians = get_ee_jacobians(
        model,
        data,
        q,
        reference_frame=reference_frame,
        update=update,
    )

    return {
        side: jacobian.linear
        for side, jacobian in jacobians.items()
    }


def get_multiple_frame_positions(
    model: pin.Model,
    data: pin.Data,
    q: Sequence[float] | np.ndarray,
    frame_names: Sequence[str],
    *,
    update: bool = True,
) -> dict[str, np.ndarray]:
    """
    Get world positions of multiple frames.
    """
    if update:
        forward_kinematics(model, data, q)

    return {
        frame_name: get_frame_position(
            model,
            data,
            q,
            frame_name,
            update=False,
        )
        for frame_name in frame_names
    }


def stack_position_errors(
    current_positions: Mapping[str, np.ndarray],
    target_positions: Mapping[str, np.ndarray],
) -> np.ndarray:
    """
    Stack Cartesian position errors.

    Error convention:
        error = target - current

    This is useful for IK.
    """
    errors: list[np.ndarray] = []

    for name, target in target_positions.items():
        if name not in current_positions:
            raise ValueError(f"Missing current position for frame/key: {name}")

        current = np.asarray(current_positions[name], dtype=float).reshape(3)
        target = np.asarray(target, dtype=float).reshape(3)

        errors.append(target - current)

    return np.concatenate(errors, axis=0)


def stack_position_jacobians(
    jacobians: Mapping[str, np.ndarray],
    keys: Sequence[str],
) -> np.ndarray:
    """
    Stack position Jacobians vertically.

    Example:
        left J:  shape (3, nv)
        right J: shape (3, nv)

        stacked J: shape (6, nv)
    """
    rows: list[np.ndarray] = []

    for key in keys:
        if key not in jacobians:
            raise ValueError(f"Missing Jacobian for key: {key}")

        rows.append(np.asarray(jacobians[key], dtype=float))

    return np.vstack(rows)