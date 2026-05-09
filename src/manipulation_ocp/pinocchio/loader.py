from __future__ import annotations

from pathlib import Path
from typing import Union

import pinocchio as pin


PathLike = Union[str, Path]


def load_model(xml_path: PathLike) -> pin.Model:
    """
    Load a Pinocchio model from a MuJoCo MJCF XML file.

    This loader assumes the MJCF file is already compatible with Pinocchio.
    """
    xml_path = Path(xml_path).resolve()

    if not xml_path.is_file():
        raise FileNotFoundError(f"MJCF XML file not found: {xml_path}")

    try:
        model = pin.buildModelFromMJCF(str(xml_path))
    except Exception as exc:
        raise RuntimeError(
            "Failed to load MJCF model with Pinocchio.\n"
            f"XML path: {xml_path}\n"
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    return model


def make_data(model: pin.Model) -> pin.Data:
    """Create Pinocchio data for a loaded model."""
    return model.createData()


def load_model_and_data(xml_path: PathLike) -> tuple[pin.Model, pin.Data]:
    """Load Pinocchio model and create its data object."""
    model = load_model(xml_path)
    data = make_data(model)

    return model, data