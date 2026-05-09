from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pinocchio as pin

from manipulation_ocp.configs.ik import DualEEIKConfig
from manipulation_ocp.configs.initial_guess import InitialGuessConfig
from manipulation_ocp.ocp.fast_dual_ee_problem import (
    FastDualEEOCP,
    FastDualEEOCPSolution,
    make_initial_guess_data,
)
from manipulation_ocp.pinocchio.ik import build_initial_guess_from_dual_ee_targets
from manipulation_ocp.pinocchio.kinematics import get_ee_positions
from manipulation_ocp.task.stages import (
    ArmAction,
    GripperAction,
    HoldAction,
    LiftAction,
    ParallelArmStep,
    ReachAction,
    ReturnArmAction,
    StageResult,
    TaskPlanResult,
)
from manipulation_ocp.utils.numerics import as_vector
from manipulation_ocp.utils.trajectory import (
    ResampledTrajectory,
    build_arm_return_trajectory,
    build_gripper_command_trajectory,
    resample_ocp_solution_to_dt,
)


@dataclass(frozen=True)
class TaskPlannerConfig:
    """
    Task-level planner config.

    dt:
        Fixed sampling time for planner output.
        This should match RL dt. Current target: 0.02 s.

    default_gripper_open_command:
        Normalized gripper open command.

    default_gripper_close_command:
        Normalized gripper close command.
    """

    dt: float = 0.02

    default_gripper_open_command: float = 0.0
    default_gripper_close_command: float = 1.0

    def validate(self) -> None:
        if self.dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {self.dt}")

        if not (0.0 <= self.default_gripper_open_command <= 1.0):
            raise ValueError(
                "default_gripper_open_command must be in [0, 1], "
                f"got {self.default_gripper_open_command}"
            )

        if not (0.0 <= self.default_gripper_close_command <= 1.0):
            raise ValueError(
                "default_gripper_close_command must be in [0, 1], "
                f"got {self.default_gripper_close_command}"
            )


class PickPlaceTaskPlanner:
    """
    Task-level planner for staggered dual-arm manipulation.

    This planner does NOT open MuJoCo viewer.
    It converts task stages into:
        - OCP solutions
        - resampled OCP q/v/u references at fixed dt
        - smooth gripper command trajectories
        - return-arm joint trajectories

    OCP build-once design:
        ocp.build_problem() should be called before planner.run(...).

    Then this planner only calls:
        ocp.solve_stage(...)

    for each motion stage.
    """

    def __init__(
        self,
        *,
        ocp: FastDualEEOCP,
        pin_model: pin.Model,
        pin_data: pin.Data,
        q_initial: Sequence[float] | np.ndarray,
        config: TaskPlannerConfig | None = None,
        ik_config: DualEEIKConfig | None = None,
        initial_guess_config: InitialGuessConfig | None = None,
        left_gripper_command: float = 0.0,
        right_gripper_command: float = 0.0,
    ) -> None:
        if config is None:
            config = TaskPlannerConfig()

        config.validate()

        self.ocp = ocp
        self.pin_model = pin_model
        self.pin_data = pin_data
        self.config = config

        self.q_initial = as_vector(q_initial, size=ocp.nq, name="q_initial")

        if ik_config is None:
            ik_config = DualEEIKConfig()

        if initial_guess_config is None:
            initial_guess_config = InitialGuessConfig(
                n_intervals=ocp.config.n_intervals,
                dt=config.dt,
            )

        self.ik_config = ik_config
        self.initial_guess_config = initial_guess_config

        self.left_gripper_command = float(left_gripper_command)
        self.right_gripper_command = float(right_gripper_command)

        self._validate_gripper_command(
            self.left_gripper_command,
            name="left_gripper_command",
        )
        self._validate_gripper_command(
            self.right_gripper_command,
            name="right_gripper_command",
        )

        home_ee = get_ee_positions(
            self.pin_model,
            self.pin_data,
            self.q_initial,
        )

        self.left_home_ee = np.asarray(home_ee["left"], dtype=float).reshape(3)
        self.right_home_ee = np.asarray(home_ee["right"], dtype=float).reshape(3)

    @staticmethod
    def _validate_gripper_command(command: float, *, name: str) -> None:
        if command < 0.0 or command > 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {command}")

    @staticmethod
    def _is_motion_action(action: ArmAction) -> bool:
        """
        Actions that require an OCP solve.
        """
        return isinstance(action, (ReachAction, LiftAction))

    def _get_current_ee_positions(
        self,
        x_current: np.ndarray,
    ) -> dict[str, np.ndarray]:
        q_current = x_current[: self.ocp.nq]

        ee = get_ee_positions(
            self.pin_model,
            self.pin_data,
            q_current,
        )

        return {
            "left": np.asarray(ee["left"], dtype=float).reshape(3),
            "right": np.asarray(ee["right"], dtype=float).reshape(3),
        }

    def _get_home_ee_position(
        self,
        side: str,
    ) -> np.ndarray:
        if side == "left":
            return self.left_home_ee.copy()

        if side == "right":
            return self.right_home_ee.copy()

        raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    def _target_from_action(
        self,
        *,
        action: ArmAction,
        current_ee: dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Convert one arm action into an EE target for the dual-EE OCP.

        For non-motion actions, the target is used as a hold target when the
        other arm needs to move.
        """
        side = action.side

        if isinstance(action, ReachAction):
            return action.target.copy()

        if isinstance(action, LiftAction):
            target = current_ee[side].copy()

            if action.dz is not None:
                target[2] += float(action.dz)
                return target

            if action.z_threshold is not None:
                target[2] = float(action.z_threshold)
                return target

            raise RuntimeError("Invalid LiftAction: missing dz/z_threshold")

        if isinstance(action, HoldAction):
            if action.mode == "home":
                return self._get_home_ee_position(side)

            if action.mode == "current":
                return current_ee[side].copy()

            raise RuntimeError(f"Unsupported HoldAction mode: {action.mode!r}")

        if isinstance(action, (GripperAction, ReturnArmAction)):
            return current_ee[side].copy()

        raise TypeError(f"Unsupported action type: {type(action).__name__}")

    def _build_initial_guess_for_targets(
        self,
        *,
        x0_value: np.ndarray,
        left_target: np.ndarray,
        right_target: np.ndarray,
    ):
        """
        Build IK/RNEA warm start for one OCP stage.
        """
        q0 = x0_value[: self.ocp.nq]

        guess = build_initial_guess_from_dual_ee_targets(
            self.pin_model,
            self.pin_data,
            q0,
            left_target,
            right_target,
            ik_config=self.ik_config,
            guess_config=self.initial_guess_config,
        )

        return make_initial_guess_data(
            guess.q_guess,
            guess.v_guess,
            guess.u_guess,
            tf=self.ocp.config.tf_init,
        )

    def _solve_motion_if_needed(
        self,
        *,
        step: ParallelArmStep,
        x_current: np.ndarray,
    ) -> tuple[
        np.ndarray,
        FastDualEEOCPSolution | None,
        ResampledTrajectory | None,
    ]:
        """
        Solve OCP if either arm has ReachAction or LiftAction.

        If motion exists:
            - build dual EE targets
            - build IK/RNEA initial guess
            - solve OCP
            - resample q/v/u to fixed planner dt

        If no motion action exists:
            returns x_current, None, None.
        """
        needs_motion = (
            self._is_motion_action(step.left)
            or self._is_motion_action(step.right)
        )

        if not needs_motion:
            return x_current, None, None

        current_ee = self._get_current_ee_positions(x_current)

        left_target = self._target_from_action(
            action=step.left,
            current_ee=current_ee,
        )

        right_target = self._target_from_action(
            action=step.right,
            current_ee=current_ee,
        )

        initial_guess = self._build_initial_guess_for_targets(
            x0_value=x_current,
            left_target=left_target,
            right_target=right_target,
        )

        solution = self.ocp.solve_stage(
            x0_value=x_current,
            left_ee_target=left_target,
            right_ee_target=right_target,
            initial_guess=initial_guess,
        )

        reference = resample_ocp_solution_to_dt(
            solution=solution,
            dt=self.config.dt,
            include_u=True,
        )

        # The fixed-dt reference is the official output of the planner.
        # Therefore the next stage starts from the final resampled state.
        x_next = np.concatenate(
            [
                reference.q[-1],
                reference.v[-1],
            ]
        )

        return x_next, solution, reference

    def _build_gripper_traj_if_needed(
        self,
        *,
        action: ArmAction,
    ) -> np.ndarray | None:
        """
        Build smooth normalized gripper command trajectory if needed.
        """
        if not isinstance(action, GripperAction):
            return None

        if action.side == "left":
            command_start = self.left_gripper_command
        elif action.side == "right":
            command_start = self.right_gripper_command
        else:
            raise ValueError(f"Invalid side: {action.side!r}")

        command_traj = build_gripper_command_trajectory(
            command_start=command_start,
            command_goal=action.command,
            duration=action.duration,
            dt=self.config.dt,
        )

        if action.side == "left":
            self.left_gripper_command = float(action.command)
        else:
            self.right_gripper_command = float(action.command)

        return command_traj

    def _apply_return_actions_if_needed(
        self,
        *,
        step: ParallelArmStep,
        x_current: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """
        Apply ReturnArmAction after any OCP motion in the step.

        If both arms request return in the same step, they are returned
        sequentially:
            left first, then right.

        Returns
        -------
        x_next:
            Final state after return action.

        q_return_traj:
            Return-arm joint trajectory, shape (num_nodes, 17), or None.

        v_return_traj:
            Return-arm velocity trajectory, shape (num_nodes, 17), or None.
        """
        return_actions: list[ReturnArmAction] = []

        if isinstance(step.left, ReturnArmAction):
            return_actions.append(step.left)

        if isinstance(step.right, ReturnArmAction):
            return_actions.append(step.right)

        if not return_actions:
            return x_current, None, None

        q_segments: list[np.ndarray] = []
        v_segments: list[np.ndarray] = []

        x_next = x_current.copy()

        for action in return_actions:
            q_current = x_next[: self.ocp.nq]

            q_traj, v_traj = build_arm_return_trajectory(
                q_current=q_current,
                q_initial=self.q_initial,
                side=action.side,
                duration=action.duration,
                dt=self.config.dt,
            )

            if q_segments:
                # Avoid duplicate boundary node.
                q_segments.append(q_traj[1:])
                v_segments.append(v_traj[1:])
            else:
                q_segments.append(q_traj)
                v_segments.append(v_traj)

            x_next = np.concatenate([q_traj[-1], v_traj[-1]])

        q_return_traj = np.vstack(q_segments)
        v_return_traj = np.vstack(v_segments)

        return x_next, q_return_traj, v_return_traj

    def run_step(
        self,
        *,
        step: ParallelArmStep,
        x_current: Sequence[float] | np.ndarray,
    ) -> StageResult:
        """
        Run one parallel arm step.

        Planner execution order:
            1. Build smooth gripper command trajectories if any.
            2. Solve OCP motion if Reach/Lift exists.
            3. Resample OCP q/v/u to fixed dt.
            4. Apply return-arm trajectory if any.

        MuJoCo executor can later decide how to visualize these outputs.
        """
        x_start = as_vector(x_current, size=self.ocp.nx, name="x_current")
        x_work = x_start.copy()

        left_gripper_traj = self._build_gripper_traj_if_needed(
            action=step.left,
        )

        right_gripper_traj = self._build_gripper_traj_if_needed(
            action=step.right,
        )

        x_after_motion, ocp_solution, ocp_reference = (
            self._solve_motion_if_needed(
                step=step,
                x_current=x_work,
            )
        )

        x_after_return, q_return_traj, v_return_traj = (
            self._apply_return_actions_if_needed(
                step=step,
                x_current=x_after_motion,
            )
        )

        return StageResult(
            step_name=step.name,
            left_action=step.left,
            right_action=step.right,
            x_start=x_start,
            x_end=x_after_return,
            ocp_solution=ocp_solution,
            ocp_time_ref=None if ocp_reference is None else ocp_reference.time,
            ocp_q_ref=None if ocp_reference is None else ocp_reference.q,
            ocp_v_ref=None if ocp_reference is None else ocp_reference.v,
            ocp_u_ref=None if ocp_reference is None else ocp_reference.u,
            left_gripper_command_traj=left_gripper_traj,
            right_gripper_command_traj=right_gripper_traj,
            q_return_traj=q_return_traj,
            v_return_traj=v_return_traj,
        )

    def run(
        self,
        *,
        x0_start: Sequence[float] | np.ndarray,
        steps: Sequence[ParallelArmStep],
    ) -> TaskPlanResult:
        """
        Run a full task plan.

        The caller defines staggered left/right behavior through the sequence
        of ParallelArmStep objects.
        """
        x_current = as_vector(x0_start, size=self.ocp.nx, name="x0_start")

        results: list[StageResult] = []

        for step in steps:
            result = self.run_step(
                step=step,
                x_current=x_current,
            )

            results.append(result)
            x_current = result.x_end.copy()

        return TaskPlanResult(
            stage_results=results,
            final_state=x_current,
        )