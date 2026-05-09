from __future__ import annotations

from dataclasses import dataclass

import pinocchio as pin


@dataclass(frozen=True)
class PinocchioModelInfo:
    nq: int
    nv: int
    njoints: int
    nframes: int
    joint_names: list[str]
    frame_names: list[str]


def get_joint_names(model: pin.Model) -> list[str]:
    """
    Return Pinocchio joint names.

    Note:
        Pinocchio includes the root/universe joint as the first entry.
    """
    return [str(name) for name in model.names]


def get_frame_names(model: pin.Model) -> list[str]:
    """Return Pinocchio frame names."""
    return [frame.name for frame in model.frames]


def get_model_info(model: pin.Model) -> PinocchioModelInfo:
    """Collect basic Pinocchio model information."""
    return PinocchioModelInfo(
        nq=model.nq,
        nv=model.nv,
        njoints=model.njoints,
        nframes=len(model.frames),
        joint_names=get_joint_names(model),
        frame_names=get_frame_names(model),
    )


def find_frame_names_containing(
    model: pin.Model,
    text: str,
) -> list[str]:
    """Return frame names that contain a given text."""
    return [
        frame.name
        for frame in model.frames
        if text in frame.name
    ]


def find_joint_names_containing(
    model: pin.Model,
    text: str,
) -> list[str]:
    """Return joint names that contain a given text."""
    return [
        str(name)
        for name in model.names
        if text in str(name)
    ]


def print_model_summary(model: pin.Model) -> None:
    """Print a compact Pinocchio model summary."""
    info = get_model_info(model)

    print("[PINOCCHIO MODEL]")
    print(f"nq      = {info.nq}")
    print(f"nv      = {info.nv}")
    print(f"njoints = {info.njoints}")
    print(f"nframes = {info.nframes}")

    print("\n[JOINTS]")
    for i, name in enumerate(info.joint_names):
        print(f"{i:02d}: {name}")

    print("\n[FRAMES CONTAINING 'grip']")
    grip_frames = find_frame_names_containing(model, "grip")
    for name in grip_frames:
        print(name)

    print("\n[FRAMES CONTAINING '2f85']")
    gripper_frames = find_frame_names_containing(model, "2f85")
    for name in gripper_frames:
        print(name)