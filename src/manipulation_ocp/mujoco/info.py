from __future__ import annotations

from dataclasses import dataclass

import mujoco


@dataclass(frozen=True)
class MujocoModelInfo:
    nq: int
    nv: int
    nu: int
    nbody: int
    njnt: int
    nsite: int
    nkey: int
    joint_names: list[str]
    actuator_names: list[str]
    site_names: list[str]
    body_names: list[str]


def _name_from_id(
    model: mujoco.MjModel,
    obj_type: mujoco.mjtObj,
    obj_id: int,
) -> str:
    """Return MuJoCo object name by id."""
    name = mujoco.mj_id2name(model, obj_type, obj_id)
    return name if name is not None else ""


def get_joint_names(model: mujoco.MjModel) -> list[str]:
    """Return all joint names in MuJoCo joint order."""
    return [
        _name_from_id(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ]


def get_actuator_names(model: mujoco.MjModel) -> list[str]:
    """Return all actuator names in MuJoCo actuator order."""
    return [
        _name_from_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        for i in range(model.nu)
    ]


def get_site_names(model: mujoco.MjModel) -> list[str]:
    """Return all site names in MuJoCo site order."""
    return [
        _name_from_id(model, mujoco.mjtObj.mjOBJ_SITE, i)
        for i in range(model.nsite)
    ]


def get_body_names(model: mujoco.MjModel) -> list[str]:
    """Return all body names in MuJoCo body order."""
    return [
        _name_from_id(model, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in range(model.nbody)
    ]


def get_model_info(model: mujoco.MjModel) -> MujocoModelInfo:
    """Collect basic MuJoCo model information."""
    return MujocoModelInfo(
        nq=model.nq,
        nv=model.nv,
        nu=model.nu,
        nbody=model.nbody,
        njnt=model.njnt,
        nsite=model.nsite,
        nkey=model.nkey,
        joint_names=get_joint_names(model),
        actuator_names=get_actuator_names(model),
        site_names=get_site_names(model),
        body_names=get_body_names(model),
    )


def print_model_summary(model: mujoco.MjModel) -> None:
    """Print a compact model summary."""
    info = get_model_info(model)

    print("[MODEL]")
    print(f"nq    = {info.nq}")
    print(f"nv    = {info.nv}")
    print(f"nu    = {info.nu}")
    print(f"nbody = {info.nbody}")
    print(f"njnt  = {info.njnt}")
    print(f"nsite = {info.nsite}")
    print(f"nkey  = {info.nkey}")

    print("\n[JOINTS]")
    for i, name in enumerate(info.joint_names):
        print(f"{i:02d}: {name}")

    print("\n[ACTUATORS]")
    for i, name in enumerate(info.actuator_names):
        print(f"{i:02d}: {name}")

    print("\n[SITES]")
    for i, name in enumerate(info.site_names):
        print(f"{i:02d}: {name}")