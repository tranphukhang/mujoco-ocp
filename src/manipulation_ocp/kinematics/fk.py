from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from manipulation_ocp.robots.g1_gripper import EE_SITE_NAMES


@dataclass(frozen=True)
class SitePose:
    """World pose of a MuJoCo site."""

    position: np.ndarray
    rotation: np.ndarray


def get_site_id(model: mujoco.MjModel, site_name: str) -> int:
    """
    Get MuJoCo site id from site name.

    Parameters
    ----------
    model:
        MuJoCo model.
    site_name:
        Name of the site.

    Returns
    -------
    site_id:
        Integer site id.

    Raises
    ------
    ValueError
        If the site does not exist.
    """
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)

    if site_id == -1:
        raise ValueError(f"Site not found: {site_name}")

    return site_id


def forward_kinematics(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """
    Run MuJoCo forward kinematics for the current qpos.

    This updates site positions, body positions, sensors, etc.
    """
    mujoco.mj_forward(model, data)


def get_site_position(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_name: str,
    *,
    update: bool = True,
) -> np.ndarray:
    """
    Get world position of a site.

    Parameters
    ----------
    model:
        MuJoCo model.
    data:
        MuJoCo data.
    site_name:
        Name of the site.
    update:
        If True, call mj_forward before reading the site position.

    Returns
    -------
    position:
        Site position in world frame, shape (3,).
    """
    if update:
        forward_kinematics(model, data)

    site_id = get_site_id(model, site_name)
    return data.site_xpos[site_id].copy()


def get_site_rotation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_name: str,
    *,
    update: bool = True,
) -> np.ndarray:
    """
    Get world rotation matrix of a site.

    Parameters
    ----------
    model:
        MuJoCo model.
    data:
        MuJoCo data.
    site_name:
        Name of the site.
    update:
        If True, call mj_forward before reading the site rotation.

    Returns
    -------
    rotation:
        Site rotation matrix in world frame, shape (3, 3).
    """
    if update:
        forward_kinematics(model, data)

    site_id = get_site_id(model, site_name)
    return data.site_xmat[site_id].reshape(3, 3).copy()


def get_site_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_name: str,
    *,
    update: bool = True,
) -> SitePose:
    """
    Get world pose of a site.

    Returns both position and rotation matrix.
    """
    if update:
        forward_kinematics(model, data)

    site_id = get_site_id(model, site_name)

    return SitePose(
        position=data.site_xpos[site_id].copy(),
        rotation=data.site_xmat[site_id].reshape(3, 3).copy(),
    )


def get_site_positions(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_names: list[str] | tuple[str, ...],
    *,
    update: bool = True,
) -> dict[str, np.ndarray]:
    """
    Get world positions of multiple sites.
    """
    if update:
        forward_kinematics(model, data)

    return {
        site_name: get_site_position(
            model,
            data,
            site_name,
            update=False,
        )
        for site_name in site_names
    }


def get_ee_positions(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    update: bool = True,
) -> dict[str, np.ndarray]:
    """
    Get world positions of G1 gripper end-effector sites.

    Returns
    -------
    ee_positions:
        Dictionary with keys:
        - "left"
        - "right"
    """
    if update:
        forward_kinematics(model, data)

    return {
        side: get_site_position(
            model,
            data,
            site_name,
            update=False,
        )
        for side, site_name in EE_SITE_NAMES.items()
    }


def get_ee_poses(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    update: bool = True,
) -> dict[str, SitePose]:
    """
    Get world poses of G1 gripper end-effector sites.

    Returns
    -------
    ee_poses:
        Dictionary with keys:
        - "left"
        - "right"
    """
    if update:
        forward_kinematics(model, data)

    return {
        side: get_site_pose(
            model,
            data,
            site_name,
            update=False,
        )
        for side, site_name in EE_SITE_NAMES.items()
    }