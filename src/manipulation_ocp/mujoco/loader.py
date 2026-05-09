from __future__ import annotations

from pathlib import Path
from typing import Union

import mujoco


PathLike = Union[str, Path]


def load_model(xml_path: PathLike) -> mujoco.MjModel:
    """
    Load a MuJoCo model from an XML path.

    Parameters
    ----------
    xml_path:
        Path to a MuJoCo XML file.

    Returns
    -------
    model:
        Loaded MuJoCo model.
    """
    xml_path = Path(xml_path).resolve()

    if not xml_path.is_file():
        raise FileNotFoundError(f"MuJoCo XML file not found: {xml_path}")

    return mujoco.MjModel.from_xml_path(str(xml_path))


def make_data(model: mujoco.MjModel) -> mujoco.MjData:
    """
    Create MuJoCo data for a loaded model.
    """
    return mujoco.MjData(model)


def load_model_and_data(xml_path: PathLike) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """
    Load a MuJoCo model and create its corresponding data object.
    """
    model = load_model(xml_path)
    data = make_data(model)

    return model, data


def reset_data(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """
    Reset MuJoCo data to default state and run forward kinematics.
    """
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)


def reset_to_keyframe(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    key_name: str,
) -> None:
    """
    Reset MuJoCo data to a named keyframe.

    Raises
    ------
    ValueError
        If the keyframe name does not exist.
    """
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)

    if key_id == -1:
        raise ValueError(f"Keyframe not found: {key_name}")

    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)