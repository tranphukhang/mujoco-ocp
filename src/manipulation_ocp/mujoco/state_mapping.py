from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import mujoco
import numpy as np

from manipulation_ocp.mujoco.limits import (
    get_joint_dof_addr,
    get_joint_qpos_addr,
)
from manipulation_ocp.robots.g1_gripper import (
    GRIPPER_JOINT_NAMES,
    OCP_COORDINATE_NAMES,
    OCP_G1_JOINT_NAMES,
    OCP_GRIPPER_COORDINATE_NAMES,
)


@dataclass(frozen=True)
class OcpMujocoStateMapping:
    """
    Mapping between reduced OCP state and full MuJoCo state.

    OCP reduced state:
        q_ocp, v_ocp: shape (19,)

        [
            17 G1 upper-body joints,
            left_gripper_opening,
            right_gripper_opening,
        ]

    MuJoCo full state:
        qpos, qvel: shape (29,)

        [
            17 active G1 joints,
            12 Robotiq internal joints,
        ]

    Important:
        Gripper opening coordinates are part of the reduced OCP state, but they
        are not mapped directly to MuJoCo internal finger joints here.

        For direct state replay, the gripper internal joints are kept from a
        reference MuJoCo state, usually the home keyframe.
    """

    ocp_coordinate_names: tuple[str, ...]
    ocp_g1_joint_names: tuple[str, ...]
    ocp_gripper_coordinate_names: tuple[str, ...]

    mujoco_gripper_joint_names: tuple[str, ...]

    g1_qpos_indices: tuple[int, ...]
    g1_qvel_indices: tuple[int, ...]

    gripper_qpos_indices: tuple[int, ...]
    gripper_qvel_indices: tuple[int, ...]


def build_state_mapping(model: mujoco.MjModel) -> OcpMujocoStateMapping:
    """
    Build mapping indices between reduced OCP state and full MuJoCo state.
    """
    g1_qpos_indices = tuple(
        get_joint_qpos_addr(model, joint_name)
        for joint_name in OCP_G1_JOINT_NAMES
    )

    g1_qvel_indices = tuple(
        get_joint_dof_addr(model, joint_name)
        for joint_name in OCP_G1_JOINT_NAMES
    )

    gripper_qpos_indices = tuple(
        get_joint_qpos_addr(model, joint_name)
        for joint_name in GRIPPER_JOINT_NAMES
    )

    gripper_qvel_indices = tuple(
        get_joint_dof_addr(model, joint_name)
        for joint_name in GRIPPER_JOINT_NAMES
    )

    return OcpMujocoStateMapping(
        ocp_coordinate_names=tuple(OCP_COORDINATE_NAMES),
        ocp_g1_joint_names=tuple(OCP_G1_JOINT_NAMES),
        ocp_gripper_coordinate_names=tuple(OCP_GRIPPER_COORDINATE_NAMES),
        mujoco_gripper_joint_names=tuple(GRIPPER_JOINT_NAMES),
        g1_qpos_indices=g1_qpos_indices,
        g1_qvel_indices=g1_qvel_indices,
        gripper_qpos_indices=gripper_qpos_indices,
        gripper_qvel_indices=gripper_qvel_indices,
    )


def _as_1d_array(
    values: Sequence[float] | np.ndarray,
    *,
    expected_size: int,
    name: str,
) -> np.ndarray:
    """Convert values to a 1D float array and validate its size."""
    array = np.asarray(values, dtype=float).reshape(-1)

    if array.size != expected_size:
        raise ValueError(
            f"{name} must have size {expected_size}, got {array.size}"
        )

    return array


def ocp_state_to_mujoco_state(
    model: mujoco.MjModel,
    q_ocp: Sequence[float] | np.ndarray,
    v_ocp: Sequence[float] | np.ndarray | None = None,
    *,
    qpos_reference: Sequence[float] | np.ndarray | None = None,
    qvel_reference: Sequence[float] | np.ndarray | None = None,
    mapping: OcpMujocoStateMapping | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert reduced OCP state to full MuJoCo qpos/qvel.

    Parameters
    ----------
    model:
        MuJoCo model.

    q_ocp:
        Reduced OCP coordinate vector, shape (19,).

    v_ocp:
        Reduced OCP velocity vector, shape (19,). If None, zeros are used.

    qpos_reference:
        Reference full MuJoCo qpos, shape (model.nq,). If None, zeros are used.

        This is important because gripper internal joints are not directly
        reconstructed from the reduced gripper opening coordinates.

    qvel_reference:
        Reference full MuJoCo qvel, shape (model.nv,). If None, zeros are used.

    mapping:
        Optional precomputed mapping.

    Returns
    -------
    qpos:
        Full MuJoCo qpos, shape (model.nq,).

    qvel:
        Full MuJoCo qvel, shape (model.nv,).
    """
    if mapping is None:
        mapping = build_state_mapping(model)

    q_ocp = _as_1d_array(
        q_ocp,
        expected_size=len(mapping.ocp_coordinate_names),
        name="q_ocp",
    )

    if v_ocp is None:
        v_ocp = np.zeros_like(q_ocp)
    else:
        v_ocp = _as_1d_array(
            v_ocp,
            expected_size=len(mapping.ocp_coordinate_names),
            name="v_ocp",
        )

    if qpos_reference is None:
        qpos = np.zeros(model.nq, dtype=float)
    else:
        qpos = _as_1d_array(
            qpos_reference,
            expected_size=model.nq,
            name="qpos_reference",
        ).copy()

    if qvel_reference is None:
        qvel = np.zeros(model.nv, dtype=float)
    else:
        qvel = _as_1d_array(
            qvel_reference,
            expected_size=model.nv,
            name="qvel_reference",
        ).copy()

    n_g1 = len(mapping.ocp_g1_joint_names)

    # Map 17 G1 upper-body coordinates directly.
    qpos[list(mapping.g1_qpos_indices)] = q_ocp[:n_g1]
    qvel[list(mapping.g1_qvel_indices)] = v_ocp[:n_g1]

    # The last 2 OCP coordinates are normalized gripper openings.
    # They are intentionally not expanded to the 12 internal Robotiq joints here.
    # The gripper internal qpos/qvel are kept from the reference state.

    return qpos, qvel


def apply_ocp_state_to_data(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q_ocp: Sequence[float] | np.ndarray,
    v_ocp: Sequence[float] | np.ndarray | None = None,
    *,
    qpos_reference: Sequence[float] | np.ndarray | None = None,
    qvel_reference: Sequence[float] | np.ndarray | None = None,
    mapping: OcpMujocoStateMapping | None = None,
    forward: bool = True,
) -> None:
    """
    Apply reduced OCP state directly to MuJoCo data.qpos/data.qvel.

    This is intended for direct state replay, not actuator-based replay.
    """
    qpos, qvel = ocp_state_to_mujoco_state(
        model,
        q_ocp,
        v_ocp,
        qpos_reference=qpos_reference,
        qvel_reference=qvel_reference,
        mapping=mapping,
    )

    data.qpos[:] = qpos
    data.qvel[:] = qvel

    if forward:
        mujoco.mj_forward(model, data)


def mujoco_state_to_ocp_state(
    model: mujoco.MjModel,
    qpos: Sequence[float] | np.ndarray,
    qvel: Sequence[float] | np.ndarray | None = None,
    *,
    gripper_opening: tuple[float, float] = (0.0, 0.0),
    mapping: OcpMujocoStateMapping | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract reduced OCP state from full MuJoCo qpos/qvel.

    Since reduced gripper opening coordinates cannot be uniquely inferred from
    the 12 Robotiq internal joints, they are provided explicitly through
    gripper_opening.

    Parameters
    ----------
    gripper_opening:
        Tuple:
            (left_gripper_opening, right_gripper_opening)

        Each value should be normalized in [0, 1].
    """
    if mapping is None:
        mapping = build_state_mapping(model)

    qpos = _as_1d_array(
        qpos,
        expected_size=model.nq,
        name="qpos",
    )

    if qvel is None:
        qvel = np.zeros(model.nv, dtype=float)
    else:
        qvel = _as_1d_array(
            qvel,
            expected_size=model.nv,
            name="qvel",
        )

    if len(gripper_opening) != 2:
        raise ValueError(
            f"gripper_opening must have size 2, got {len(gripper_opening)}"
        )

    q_ocp = np.zeros(len(mapping.ocp_coordinate_names), dtype=float)
    v_ocp = np.zeros(len(mapping.ocp_coordinate_names), dtype=float)

    n_g1 = len(mapping.ocp_g1_joint_names)

    q_ocp[:n_g1] = qpos[list(mapping.g1_qpos_indices)]
    v_ocp[:n_g1] = qvel[list(mapping.g1_qvel_indices)]

    q_ocp[n_g1:] = np.asarray(gripper_opening, dtype=float)
    v_ocp[n_g1:] = 0.0

    return q_ocp, v_ocp


def extract_ocp_state_from_data(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    gripper_opening: tuple[float, float] = (0.0, 0.0),
    mapping: OcpMujocoStateMapping | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract reduced OCP q/v from current MuJoCo data.
    """
    return mujoco_state_to_ocp_state(
        model,
        data.qpos,
        data.qvel,
        gripper_opening=gripper_opening,
        mapping=mapping,
    )