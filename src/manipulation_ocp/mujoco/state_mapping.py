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
)


@dataclass(frozen=True)
class OcpMujocoStateMapping:
    """
    Mapping between OCP/Pinocchio state and full MuJoCo replay state.

    OCP state:
        q_ocp, v_ocp: shape (17,)

        [
            3 waist joints,
            7 left arm joints,
            7 right arm joints,
        ]

    MuJoCo replay state:
        qpos, qvel: shape (29,)

        [
            17 active G1 upper-body joints,
            12 Robotiq internal gripper joints,
        ]

    Important:
        Gripper is NOT part of the OCP decision variables.

        Therefore, this mapping only writes the 17 G1 upper-body joints into
        the MuJoCo replay state. The 12 internal gripper joints are kept from
        a reference MuJoCo state, usually a keyframe such as "home" or "stand".
    """

    ocp_coordinate_names: tuple[str, ...]
    ocp_g1_joint_names: tuple[str, ...]

    mujoco_gripper_joint_names: tuple[str, ...]

    g1_qpos_indices: tuple[int, ...]
    g1_qvel_indices: tuple[int, ...]

    gripper_qpos_indices: tuple[int, ...]
    gripper_qvel_indices: tuple[int, ...]


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


def build_state_mapping(model: mujoco.MjModel) -> OcpMujocoStateMapping:
    """
    Build mapping indices between 17-DoF OCP state and full MuJoCo replay state.
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
        mujoco_gripper_joint_names=tuple(GRIPPER_JOINT_NAMES),
        g1_qpos_indices=g1_qpos_indices,
        g1_qvel_indices=g1_qvel_indices,
        gripper_qpos_indices=gripper_qpos_indices,
        gripper_qvel_indices=gripper_qvel_indices,
    )


def get_keyframe_id(model: mujoco.MjModel, key_name: str) -> int:
    """
    Return MuJoCo keyframe id by name.
    """
    key_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        key_name,
    )

    if key_id < 0:
        available = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i)
            for i in range(model.nkey)
        ]
        raise ValueError(
            f"Keyframe not found: {key_name}. "
            f"Available keyframes: {available}"
        )

    return key_id


def get_keyframe_state(
    model: mujoco.MjModel,
    key_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get qpos/qvel reference from a MuJoCo keyframe.

    Returns
    -------
    qpos:
        shape = (model.nq,)

    qvel:
        shape = (model.nv,)
    """
    key_id = get_keyframe_id(model, key_name)

    qpos = np.asarray(model.key_qpos[key_id], dtype=float).reshape(model.nq)

    # MuJoCo keyframes may contain qvel. If unavailable in a given binding/model,
    # fall back to zero velocity.
    if hasattr(model, "key_qvel"):
        qvel = np.asarray(model.key_qvel[key_id], dtype=float).reshape(model.nv)
    else:
        qvel = np.zeros(model.nv, dtype=float)

    return qpos.copy(), qvel.copy()


def get_default_replay_reference_state(
    model: mujoco.MjModel,
    *,
    preferred_key_names: tuple[str, ...] = ("home", "stand"),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get a reasonable MuJoCo replay reference state.

    Priority:
        1. keyframe "home" if available
        2. keyframe "stand" if available
        3. first keyframe if model.nkey > 0
        4. zeros

    This reference is mainly used to preserve internal gripper joints during
    state replay.
    """
    for key_name in preferred_key_names:
        key_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_KEY,
            key_name,
        )
        if key_id >= 0:
            return get_keyframe_state(model, key_name)

    if model.nkey > 0:
        qpos = np.asarray(model.key_qpos[0], dtype=float).reshape(model.nq)

        if hasattr(model, "key_qvel"):
            qvel = np.asarray(model.key_qvel[0], dtype=float).reshape(model.nv)
        else:
            qvel = np.zeros(model.nv, dtype=float)

        return qpos.copy(), qvel.copy()

    return np.zeros(model.nq, dtype=float), np.zeros(model.nv, dtype=float)


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
    Convert 17-DoF OCP state to full MuJoCo replay qpos/qvel.

    Parameters
    ----------
    q_ocp:
        OCP joint positions, shape (17,).

    v_ocp:
        OCP joint velocities, shape (17,). If None, zeros are used.

    qpos_reference:
        Full MuJoCo qpos reference, shape (model.nq,). If None, a default
        keyframe reference is used.

        This is important because the 12 internal gripper joints are not
        reconstructed from OCP.

    qvel_reference:
        Full MuJoCo qvel reference, shape (model.nv,). If None, a default
        keyframe reference is used.

    Returns
    -------
    qpos:
        Full MuJoCo qpos, shape (model.nq,).

    qvel:
        Full MuJoCo qvel, shape (model.nv,).
    """
    if mapping is None:
        mapping = build_state_mapping(model)

    q_ocp_vec = _as_1d_array(
        q_ocp,
        expected_size=len(mapping.ocp_coordinate_names),
        name="q_ocp",
    )

    if v_ocp is None:
        v_ocp_vec = np.zeros_like(q_ocp_vec)
    else:
        v_ocp_vec = _as_1d_array(
            v_ocp,
            expected_size=len(mapping.ocp_coordinate_names),
            name="v_ocp",
        )

    if qpos_reference is None or qvel_reference is None:
        default_qpos, default_qvel = get_default_replay_reference_state(model)
    else:
        default_qpos = None
        default_qvel = None

    if qpos_reference is None:
        qpos = default_qpos.copy()
    else:
        qpos = _as_1d_array(
            qpos_reference,
            expected_size=model.nq,
            name="qpos_reference",
        ).copy()

    if qvel_reference is None:
        qvel = default_qvel.copy()
    else:
        qvel = _as_1d_array(
            qvel_reference,
            expected_size=model.nv,
            name="qvel_reference",
        ).copy()

    # Map only the 17 G1 upper-body coordinates.
    qpos[list(mapping.g1_qpos_indices)] = q_ocp_vec
    qvel[list(mapping.g1_qvel_indices)] = v_ocp_vec

    # Gripper internal joints are intentionally left unchanged from reference.

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
    Apply 17-DoF OCP state directly to MuJoCo data.qpos/data.qvel.

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


def ocp_trajectory_to_mujoco_trajectory(
    model: mujoco.MjModel,
    q_ocp_traj: Sequence[Sequence[float]] | np.ndarray,
    v_ocp_traj: Sequence[Sequence[float]] | np.ndarray | None = None,
    *,
    qpos_reference: Sequence[float] | np.ndarray | None = None,
    qvel_reference: Sequence[float] | np.ndarray | None = None,
    mapping: OcpMujocoStateMapping | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert OCP q/v trajectories to full MuJoCo qpos/qvel trajectories.

    Expected input shapes:
        q_ocp_traj = (num_nodes, 17)
        v_ocp_traj = (num_nodes, 17), optional

    Returns:
        qpos_traj = (num_nodes, model.nq)
        qvel_traj = (num_nodes, model.nv)
    """
    if mapping is None:
        mapping = build_state_mapping(model)

    q_arr = np.asarray(q_ocp_traj, dtype=float)

    if q_arr.ndim != 2:
        raise ValueError(f"q_ocp_traj must be 2D, got shape {q_arr.shape}")

    num_nodes, q_dim = q_arr.shape
    expected_dim = len(mapping.ocp_coordinate_names)

    if q_dim != expected_dim:
        raise ValueError(
            f"q_ocp_traj must have shape (num_nodes, {expected_dim}), "
            f"got {q_arr.shape}"
        )

    if v_ocp_traj is None:
        v_arr = np.zeros_like(q_arr)
    else:
        v_arr = np.asarray(v_ocp_traj, dtype=float)

        if v_arr.shape != q_arr.shape:
            raise ValueError(
                f"v_ocp_traj must have shape {q_arr.shape}, got {v_arr.shape}"
            )

    if qpos_reference is None or qvel_reference is None:
        default_qpos, default_qvel = get_default_replay_reference_state(model)
    else:
        default_qpos = None
        default_qvel = None

    if qpos_reference is None:
        qpos_ref = default_qpos
    else:
        qpos_ref = _as_1d_array(
            qpos_reference,
            expected_size=model.nq,
            name="qpos_reference",
        )

    if qvel_reference is None:
        qvel_ref = default_qvel
    else:
        qvel_ref = _as_1d_array(
            qvel_reference,
            expected_size=model.nv,
            name="qvel_reference",
        )

    qpos_traj = np.zeros((num_nodes, model.nq), dtype=float)
    qvel_traj = np.zeros((num_nodes, model.nv), dtype=float)

    for k in range(num_nodes):
        qpos_k, qvel_k = ocp_state_to_mujoco_state(
            model,
            q_arr[k],
            v_arr[k],
            qpos_reference=qpos_ref,
            qvel_reference=qvel_ref,
            mapping=mapping,
        )

        qpos_traj[k] = qpos_k
        qvel_traj[k] = qvel_k

    return qpos_traj, qvel_traj


def mujoco_state_to_ocp_state(
    model: mujoco.MjModel,
    qpos: Sequence[float] | np.ndarray,
    qvel: Sequence[float] | np.ndarray | None = None,
    *,
    mapping: OcpMujocoStateMapping | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract 17-DoF OCP q/v from full MuJoCo replay qpos/qvel.

    Gripper internal joints are ignored.
    """
    if mapping is None:
        mapping = build_state_mapping(model)

    qpos_vec = _as_1d_array(
        qpos,
        expected_size=model.nq,
        name="qpos",
    )

    if qvel is None:
        qvel_vec = np.zeros(model.nv, dtype=float)
    else:
        qvel_vec = _as_1d_array(
            qvel,
            expected_size=model.nv,
            name="qvel",
        )

    q_ocp = np.zeros(len(mapping.ocp_coordinate_names), dtype=float)
    v_ocp = np.zeros(len(mapping.ocp_coordinate_names), dtype=float)

    q_ocp[:] = qpos_vec[list(mapping.g1_qpos_indices)]
    v_ocp[:] = qvel_vec[list(mapping.g1_qvel_indices)]

    return q_ocp, v_ocp


def extract_ocp_state_from_data(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    mapping: OcpMujocoStateMapping | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract 17-DoF OCP q/v from current MuJoCo data.
    """
    return mujoco_state_to_ocp_state(
        model,
        data.qpos,
        data.qvel,
        mapping=mapping,
    )