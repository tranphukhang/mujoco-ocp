from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence, TypeAlias

import numpy as np


ArmSide: TypeAlias = Literal["left", "right"]
HoldMode: TypeAlias = Literal["current", "home"]


def _validate_side(side: str) -> ArmSide:
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    return side  # type: ignore[return-value]


def _as_vec3(value: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)

    if arr.size != 3:
        raise ValueError(f"{name} must have size 3, got shape {arr.shape}")

    return arr


@dataclass(frozen=True)
class HoldAction:
    """
    Keep one arm idle during a parallel task step.

    mode:
        "current":
            Keep current EE target.

        "home":
            Use the arm home EE position as target.
            Useful at the first step when right arm should stay home.
    """

    side: ArmSide
    mode: HoldMode = "current"

    def __post_init__(self) -> None:
        object.__setattr__(self, "side", _validate_side(self.side))

        if self.mode not in ("current", "home"):
            raise ValueError(
                f"HoldAction.mode must be 'current' or 'home', got {self.mode!r}"
            )


@dataclass(frozen=True)
class ReachAction:
    """
    Move one end-effector to a target position.

    target:
        EE target position in the Pinocchio pelvis/base frame.
    """

    side: ArmSide
    target: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "side", _validate_side(self.side))
        object.__setattr__(
            self,
            "target",
            _as_vec3(self.target, name="ReachAction.target"),
        )


@dataclass(frozen=True)
class LiftAction:
    """
    Lift one end-effector after grasping.

    Exactly one of dz or z_threshold should be provided.

    dz:
        Relative lifting distance along +z in pelvis/base frame.

    z_threshold:
        Absolute desired z height in pelvis/base frame.
    """

    side: ArmSide
    dz: float | None = None
    z_threshold: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "side", _validate_side(self.side))

        has_dz = self.dz is not None
        has_z_threshold = self.z_threshold is not None

        if has_dz == has_z_threshold:
            raise ValueError(
                "LiftAction requires exactly one of dz or z_threshold."
            )

        if self.dz is not None and self.dz <= 0.0:
            raise ValueError(f"LiftAction.dz must be > 0, got {self.dz}")

        if self.z_threshold is not None and not np.isfinite(self.z_threshold):
            raise ValueError(
                f"LiftAction.z_threshold must be finite, got {self.z_threshold}"
            )


@dataclass(frozen=True)
class GripperAction:
    """
    Smoothly open or close one gripper.

    command convention:
        0.0 = open
        1.0 = close

    duration:
        Duration of gripper command trajectory in seconds.
    """

    side: ArmSide
    command: float
    duration: float = 0.4

    def __post_init__(self) -> None:
        object.__setattr__(self, "side", _validate_side(self.side))

        if self.command < 0.0 or self.command > 1.0:
            raise ValueError(
                f"GripperAction.command must be in [0, 1], got {self.command}"
            )

        if self.duration <= 0.0:
            raise ValueError(
                f"GripperAction.duration must be > 0, got {self.duration}"
            )


@dataclass(frozen=True)
class ReturnArmAction:
    """
    Return one arm to its initial joint configuration.

    This action does not use OCP. It should be executed with a simple
    joint-space trajectory generator.

    side:
        "left"  -> return q[3:10]
        "right" -> return q[10:17]
    """

    side: ArmSide
    duration: float = 0.6

    def __post_init__(self) -> None:
        object.__setattr__(self, "side", _validate_side(self.side))

        if self.duration <= 0.0:
            raise ValueError(
                f"ReturnArmAction.duration must be > 0, got {self.duration}"
            )


ArmAction: TypeAlias = (
    HoldAction
    | ReachAction
    | LiftAction
    | GripperAction
    | ReturnArmAction
)


@dataclass(frozen=True)
class ParallelArmStep:
    """
    One task-level step containing one action for each arm.

    This is used to represent staggered left/right arm behavior.

    Example:
        step 0:
            left  = ReachAction(...)
            right = HoldAction(mode="home")

        step 1:
            left  = LiftAction(...)
            right = ReachAction(...)
    """

    name: str
    left: ArmAction
    right: ArmAction

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ParallelArmStep.name must be non-empty")

        if self.left.side != "left":
            raise ValueError(
                f"ParallelArmStep.left must have side='left', got {self.left.side!r}"
            )

        if self.right.side != "right":
            raise ValueError(
                f"ParallelArmStep.right must have side='right', got {self.right.side!r}"
            )


@dataclass(frozen=True)
class StageResult:
    """
    Result for one ParallelArmStep.

    This is intentionally generic so planner and MuJoCo executor can attach
    OCP solution, resampled OCP reference, gripper trajectories, or joint-space
    return trajectories.
    """

    step_name: str
    left_action: ArmAction
    right_action: ArmAction

    x_start: np.ndarray
    x_end: np.ndarray

    # Raw OCP solution.
    ocp_solution: Any | None = None

    # OCP trajectory resampled to planner dt, usually dt = 0.02.
    # These are the references that should be passed to RL / MuJoCo executor.
    ocp_time_ref: np.ndarray | None = None
    ocp_q_ref: np.ndarray | None = None
    ocp_v_ref: np.ndarray | None = None
    ocp_u_ref: np.ndarray | None = None

    # Smooth normalized gripper command trajectories.
    left_gripper_command_traj: np.ndarray | None = None
    right_gripper_command_traj: np.ndarray | None = None

    # Joint-space return trajectory for ReturnArmAction.
    q_return_traj: np.ndarray | None = None
    v_return_traj: np.ndarray | None = None


@dataclass(frozen=True)
class TaskPlanResult:
    """
    Result for a complete task-level plan.
    """

    stage_results: list[StageResult]
    final_state: np.ndarray

    @property
    def num_stages(self) -> int:
        return len(self.stage_results)