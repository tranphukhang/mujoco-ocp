from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from manipulation_ocp.kinematics.fk import forward_kinematics, get_site_id
from manipulation_ocp.robots.g1_gripper import EE_SITE_NAMES


@dataclass(frozen=True)
class SiteJacobian:
    """World-frame site Jacobian."""

    jacp: np.ndarray  # linear position Jacobian, shape (3, nv)
    jacr: np.ndarray  # rotational Jacobian, shape (3, nv)


def get_site_jacobian(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_name: str,
    *,
    update: bool = True,
) -> SiteJacobian:
    """
    Get linear and angular Jacobian of a MuJoCo site.

    The returned Jacobians satisfy approximately:

        p_dot = jacp @ qvel
        omega = jacr @ qvel

    Parameters
    ----------
    model:
        MuJoCo model.
    data:
        MuJoCo data.
    site_name:
        Name of the site.
    update:
        If True, call mj_forward before computing Jacobian.

    Returns
    -------
    SiteJacobian
        jacp and jacr, each with shape (3, nv).
    """
    if update:
        forward_kinematics(model, data)

    site_id = get_site_id(model, site_name)

    jacp = np.zeros((3, model.nv), dtype=float)
    jacr = np.zeros((3, model.nv), dtype=float)

    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

    return SiteJacobian(jacp=jacp, jacr=jacr)


def get_site_position_jacobian(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_name: str,
    *,
    update: bool = True,
) -> np.ndarray:
    """
    Get position Jacobian of a site.

    Returns
    -------
    jacp:
        Linear Jacobian, shape (3, nv).
    """
    return get_site_jacobian(
        model,
        data,
        site_name,
        update=update,
    ).jacp


def get_site_rotation_jacobian(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_name: str,
    *,
    update: bool = True,
) -> np.ndarray:
    """
    Get rotational Jacobian of a site.

    Returns
    -------
    jacr:
        Angular Jacobian, shape (3, nv).
    """
    return get_site_jacobian(
        model,
        data,
        site_name,
        update=update,
    ).jacr


def get_ee_position_jacobians(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    update: bool = True,
) -> dict[str, np.ndarray]:
    """
    Get position Jacobians of G1 gripper end-effector sites.

    Returns
    -------
    jacobians:
        Dictionary with keys:
        - "left"
        - "right"

        Each value has shape (3, nv).
    """
    if update:
        forward_kinematics(model, data)

    return {
        side: get_site_position_jacobian(
            model,
            data,
            site_name,
            update=False,
        )
        for side, site_name in EE_SITE_NAMES.items()
    }


def get_ee_jacobians(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    update: bool = True,
) -> dict[str, SiteJacobian]:
    """
    Get full Jacobians of G1 gripper end-effector sites.

    Returns
    -------
    jacobians:
        Dictionary with keys:
        - "left"
        - "right"

        Each value contains jacp and jacr.
    """
    if update:
        forward_kinematics(model, data)

    return {
        side: get_site_jacobian(
            model,
            data,
            site_name,
            update=False,
        )
        for side, site_name in EE_SITE_NAMES.items()
    }


def get_joint_dof_indices(
    model: mujoco.MjModel,
    joint_names: list[str] | tuple[str, ...],
) -> list[int]:
    """
    Get qvel/dof indices corresponding to a list of joint names.

    For this G1 gripper model, all joints are hinge joints, so each joint
    contributes one dof. This function is still written generally enough to
    use MuJoCo's jnt_dofadr.
    """
    dof_indices: list[int] = []

    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        if joint_id == -1:
            raise ValueError(f"Joint not found: {joint_name}")

        dof_start = int(model.jnt_dofadr[joint_id])
        joint_type = int(model.jnt_type[joint_id])

        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            dof_count = 6
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            dof_count = 3
        else:
            dof_count = 1

        dof_indices.extend(range(dof_start, dof_start + dof_count))

    return dof_indices


def select_jacobian_columns(
    jacobian: np.ndarray,
    dof_indices: list[int] | tuple[int, ...],
) -> np.ndarray:
    """
    Select Jacobian columns corresponding to a subset of qvel dofs.

    Parameters
    ----------
    jacobian:
        Jacobian with shape (m, nv).
    dof_indices:
        qvel/dof indices to keep.

    Returns
    -------
    reduced_jacobian:
        Jacobian with shape (m, len(dof_indices)).
    """
    return jacobian[:, list(dof_indices)].copy()