from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import time

import mujoco
import numpy as np

try:
    import mujoco.viewer as mujoco_viewer
except Exception:  # pragma: no cover
    mujoco_viewer = None

from manipulation_ocp.mujoco.loader import load_model_and_data, reset_to_keyframe
from manipulation_ocp.mujoco.state_mapping import (
    OcpMujocoStateMapping,
    apply_ocp_state_to_data,
    build_state_mapping,
    get_default_replay_reference_state,
)
from manipulation_ocp.robots.g1_gripper import (
    MUJOCO_GRIPPER_ACTUATOR_NAMES,
    MUJOCO_POSITION_ACTUATOR_NAMES,
    MUJOCO_REPLAY_XML,
    gripper_command_to_mujoco_ctrl,
    validate_mujoco_replay_model,
)
from manipulation_ocp.task.stages import StageResult, TaskPlanResult


@dataclass(frozen=True)
class MujocoTaskExecutorConfig:
    """
    MuJoCo task executor config.

    This executor is for visualization/replay, not for OCP solving.

    show_viewer:
        If True, open MuJoCo passive viewer.

    realtime:
        If True, sleep between samples using dt.

    dt:
        Fallback sample time if a stage does not provide time vector.

    reset_keyframe:
        Optional MuJoCo keyframe name. If None, the executor uses the default
        replay reference state from state_mapping.py.

    pause_between_stages:
        If True, hold the viewer briefly between stages.

    stage_pause_time:
        Pause duration after each stage.

    hold_final_time:
        Viewer hold duration after the full plan is done.

    set_g1_position_ctrl:
        If True, also writes G1 position actuator ctrl to q_ocp.
        The visual replay still primarily uses direct state replay.

    use_model_timestep_substeps:
        If True, each planner sample is advanced using multiple MuJoCo steps
        based on model.opt.timestep.

    substeps_per_sample:
        Optional manual override. If None, computed from dt/model timestep.

    print_stage:
        If True, print stage names during execution.
    """

    show_viewer: bool = True
    realtime: bool = True

    dt: float = 0.02

    reset_keyframe: str | None = None

    pause_between_stages: bool = True
    stage_pause_time: float = 0.25
    hold_final_time: float = 0.5

    set_g1_position_ctrl: bool = True

    use_model_timestep_substeps: bool = True
    substeps_per_sample: int | None = None

    print_stage: bool = True

    def validate(self) -> None:
        if not np.isfinite(self.dt):
            raise ValueError(f"dt must be finite, got {self.dt}")

        if self.dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {self.dt}")

        for name, value in {
            "stage_pause_time": self.stage_pause_time,
            "hold_final_time": self.hold_final_time,
        }.items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")

            if value < 0.0:
                raise ValueError(f"{name} must be >= 0, got {value}")

        if self.substeps_per_sample is not None and self.substeps_per_sample <= 0:
            raise ValueError(
                "substeps_per_sample must be > 0 when provided, "
                f"got {self.substeps_per_sample}"
            )


class MujocoTaskExecutor:
    """
    Replay a TaskPlanResult in MuJoCo.

    This executor supports:
        - OCP q/v reference replay
        - return-arm q/v replay
        - smooth gripper command replay
        - optional MuJoCo passive viewer

    Important:
        OCP does not optimize gripper joints.
        Gripper is replayed through MuJoCo gripper actuator controls.

    Replay strategy:
        G1 upper-body joints are applied by direct state replay using
        OCP q/v references.

        Gripper command is applied through normalized command trajectories:
            0.0 = open
            1.0 = close

        which are converted to MuJoCo ctrl in [0, 255].
    """

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        config: MujocoTaskExecutorConfig | None = None,
        mapping: OcpMujocoStateMapping | None = None,
        left_gripper_command: float = 0.0,
        right_gripper_command: float = 0.0,
        validate_model: bool = True,
    ) -> None:
        if config is None:
            config = MujocoTaskExecutorConfig()

        config.validate()

        if validate_model:
            validate_mujoco_replay_model(model)

        self.model = model
        self.data = data
        self.config = config

        self.mapping = mapping if mapping is not None else build_state_mapping(model)

        self.position_actuator_ids = self._get_actuator_ids(
            MUJOCO_POSITION_ACTUATOR_NAMES,
        )

        self.left_gripper_actuator_id = self._get_actuator_id(
            MUJOCO_GRIPPER_ACTUATOR_NAMES[0],
        )
        self.right_gripper_actuator_id = self._get_actuator_id(
            MUJOCO_GRIPPER_ACTUATOR_NAMES[1],
        )

        self.left_gripper_command = self._validate_gripper_command(
            left_gripper_command,
            name="left_gripper_command",
        )
        self.right_gripper_command = self._validate_gripper_command(
            right_gripper_command,
            name="right_gripper_command",
        )

        self.reset()

    @classmethod
    def from_xml_path(
        cls,
        xml_path: str | Path = MUJOCO_REPLAY_XML,
        *,
        config: MujocoTaskExecutorConfig | None = None,
        left_gripper_command: float = 0.0,
        right_gripper_command: float = 0.0,
        validate_model: bool = True,
    ) -> MujocoTaskExecutor:
        """
        Load MuJoCo replay model and create executor.
        """
        model, data = load_model_and_data(xml_path)

        return cls(
            model=model,
            data=data,
            config=config,
            left_gripper_command=left_gripper_command,
            right_gripper_command=right_gripper_command,
            validate_model=validate_model,
        )

    @staticmethod
    def _validate_gripper_command(
        command: float,
        *,
        name: str,
    ) -> float:
        command_float = float(command)

        if not np.isfinite(command_float):
            raise ValueError(f"{name} must be finite, got {command}")

        if command_float < 0.0 or command_float > 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {command_float}")

        return command_float

    def _get_actuator_id(self, name: str) -> int:
        actuator_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            name,
        )

        if actuator_id < 0:
            raise ValueError(f"MuJoCo actuator not found: {name}")

        return int(actuator_id)

    def _get_actuator_ids(
        self,
        names: Sequence[str],
    ) -> tuple[int, ...]:
        return tuple(self._get_actuator_id(name) for name in names)

    def reset(self) -> None:
        """
        Reset MuJoCo data to a stable replay state.
        """
        if self.config.reset_keyframe is not None:
            reset_to_keyframe(
                self.model,
                self.data,
                self.config.reset_keyframe,
            )
        else:
            qpos_ref, qvel_ref = get_default_replay_reference_state(self.model)
            self.data.qpos[:] = qpos_ref
            self.data.qvel[:] = qvel_ref
            mujoco.mj_forward(self.model, self.data)

        self._set_gripper_commands(
            left_command=self.left_gripper_command,
            right_command=self.right_gripper_command,
        )

        mujoco.mj_forward(self.model, self.data)

    def _set_g1_position_ctrl_from_q(
        self,
        q_ocp: np.ndarray,
    ) -> None:
        """
        Set G1 position actuator controls from OCP q.

        This is not the primary replay mechanism. Direct state replay is.
        This just helps MuJoCo actuators remain consistent when stepping.
        """
        if not self.config.set_g1_position_ctrl:
            return

        q = np.asarray(q_ocp, dtype=float).reshape(-1)

        if q.size != len(self.position_actuator_ids):
            raise ValueError(
                "q_ocp size must match number of G1 position actuators. "
                f"Got q size={q.size}, actuators={len(self.position_actuator_ids)}"
            )

        for actuator_id, value in zip(self.position_actuator_ids, q):
            self.data.ctrl[actuator_id] = float(value)

    def _set_gripper_commands(
        self,
        *,
        left_command: float | None = None,
        right_command: float | None = None,
    ) -> None:
        """
        Set MuJoCo gripper actuator controls from normalized commands.
        """
        if left_command is not None:
            left = self._validate_gripper_command(
                left_command,
                name="left_command",
            )
            self.data.ctrl[self.left_gripper_actuator_id] = (
                gripper_command_to_mujoco_ctrl(left)
            )

        if right_command is not None:
            right = self._validate_gripper_command(
                right_command,
                name="right_command",
            )
            self.data.ctrl[self.right_gripper_actuator_id] = (
                gripper_command_to_mujoco_ctrl(right)
            )

    def _apply_ocp_state(
        self,
        *,
        q_ocp: np.ndarray,
        v_ocp: np.ndarray | None,
    ) -> None:
        """
        Apply one OCP q/v sample to MuJoCo data.

        Current MuJoCo qpos/qvel are used as reference so gripper internal
        joints are preserved while G1 upper-body joints are overwritten.
        """
        q = np.asarray(q_ocp, dtype=float).reshape(-1)

        if v_ocp is None:
            v = np.zeros_like(q)
        else:
            v = np.asarray(v_ocp, dtype=float).reshape(-1)

        apply_ocp_state_to_data(
            self.model,
            self.data,
            q,
            v,
            qpos_reference=self.data.qpos.copy(),
            qvel_reference=self.data.qvel.copy(),
            mapping=self.mapping,
            forward=True,
        )

        self._set_g1_position_ctrl_from_q(q)

    def _num_substeps_for_sample(
        self,
        sample_dt: float,
    ) -> int:
        if self.config.substeps_per_sample is not None:
            return int(self.config.substeps_per_sample)

        if not self.config.use_model_timestep_substeps:
            return 1

        model_dt = float(self.model.opt.timestep)

        if model_dt <= 0.0 or not np.isfinite(model_dt):
            return 1

        return max(1, int(round(sample_dt / model_dt)))

    def _sync_viewer(
        self,
        viewer: Any | None,
    ) -> bool:
        """
        Sync viewer if available.

        Returns False if viewer was closed.
        """
        if viewer is None:
            return True

        if hasattr(viewer, "is_running") and not viewer.is_running():
            return False

        viewer.sync()
        return True

    def _sleep_realtime(
        self,
        duration: float,
    ) -> None:
        if self.config.realtime and duration > 0.0:
            time.sleep(duration)

    def _advance_sample(
        self,
        *,
        viewer: Any | None,
        sample_dt: float,
        step_simulation: bool,
    ) -> bool:
        """
        Advance one visual sample.

        If step_simulation=True, MuJoCo is stepped so gripper actuator dynamics
        can move. Otherwise only mj_forward + viewer sync is used.
        """
        sample_dt = float(sample_dt)

        if sample_dt <= 0.0 or not np.isfinite(sample_dt):
            sample_dt = self.config.dt

        if step_simulation:
            substeps = self._num_substeps_for_sample(sample_dt)
            for _ in range(substeps):
                mujoco.mj_step(self.model, self.data)
        else:
            mujoco.mj_forward(self.model, self.data)

        ok = self._sync_viewer(viewer)
        self._sleep_realtime(sample_dt)

        return ok

    @staticmethod
    def _sample_optional_traj(
        traj: np.ndarray | None,
        index: int,
        *,
        current_value: float,
    ) -> float:
        if traj is None:
            return float(current_value)

        arr = np.asarray(traj, dtype=float).reshape(-1)

        if arr.size == 0:
            return float(current_value)

        sample_id = min(index, arr.size - 1)

        return float(arr[sample_id])

    @staticmethod
    def _stage_sample_dt(
        *,
        time_ref: np.ndarray | None,
        index: int,
        fallback_dt: float,
    ) -> float:
        if time_ref is None:
            return fallback_dt

        time_arr = np.asarray(time_ref, dtype=float).reshape(-1)

        if time_arr.size < 2:
            return fallback_dt

        if index < time_arr.size - 1:
            dt = float(time_arr[index + 1] - time_arr[index])
        else:
            dt = float(time_arr[-1] - time_arr[-2])

        if dt <= 0.0 or not np.isfinite(dt):
            return fallback_dt

        return dt

    def _replay_qv_trajectory(
        self,
        *,
        q_traj: np.ndarray,
        v_traj: np.ndarray | None,
        time_ref: np.ndarray | None,
        left_gripper_traj: np.ndarray | None,
        right_gripper_traj: np.ndarray | None,
        viewer: Any | None,
    ) -> bool:
        """
        Replay a q/v trajectory, optionally with gripper command trajectories.
        """
        q_arr = np.asarray(q_traj, dtype=float)

        if q_arr.ndim != 2:
            raise ValueError(f"q_traj must be 2D, got shape {q_arr.shape}")

        if v_traj is None:
            v_arr = np.zeros_like(q_arr)
        else:
            v_arr = np.asarray(v_traj, dtype=float)

            if v_arr.shape != q_arr.shape:
                raise ValueError(
                    f"v_traj must have shape {q_arr.shape}, got {v_arr.shape}"
                )

        has_gripper_motion = (
            left_gripper_traj is not None
            or right_gripper_traj is not None
        )

        for k in range(q_arr.shape[0]):
            q_k = q_arr[k]
            v_k = v_arr[k]

            left_command = self._sample_optional_traj(
                left_gripper_traj,
                k,
                current_value=self.left_gripper_command,
            )
            right_command = self._sample_optional_traj(
                right_gripper_traj,
                k,
                current_value=self.right_gripper_command,
            )

            self._apply_ocp_state(q_ocp=q_k, v_ocp=v_k)
            self._set_gripper_commands(
                left_command=left_command,
                right_command=right_command,
            )

            sample_dt = self._stage_sample_dt(
                time_ref=time_ref,
                index=k,
                fallback_dt=self.config.dt,
            )

            ok = self._advance_sample(
                viewer=viewer,
                sample_dt=sample_dt,
                step_simulation=has_gripper_motion,
            )

            if not ok:
                return False

        if left_gripper_traj is not None:
            self.left_gripper_command = float(
                np.asarray(left_gripper_traj, dtype=float).reshape(-1)[-1]
            )

        if right_gripper_traj is not None:
            self.right_gripper_command = float(
                np.asarray(right_gripper_traj, dtype=float).reshape(-1)[-1]
            )

        return True

    def _replay_gripper_only(
        self,
        *,
        left_gripper_traj: np.ndarray | None,
        right_gripper_traj: np.ndarray | None,
        viewer: Any | None,
    ) -> bool:
        """
        Replay gripper command trajectories without changing OCP q/v.
        """
        left_len = 0 if left_gripper_traj is None else len(left_gripper_traj)
        right_len = 0 if right_gripper_traj is None else len(right_gripper_traj)

        num_samples = max(left_len, right_len)

        if num_samples == 0:
            return True

        for k in range(num_samples):
            left_command = self._sample_optional_traj(
                left_gripper_traj,
                k,
                current_value=self.left_gripper_command,
            )
            right_command = self._sample_optional_traj(
                right_gripper_traj,
                k,
                current_value=self.right_gripper_command,
            )

            self._set_gripper_commands(
                left_command=left_command,
                right_command=right_command,
            )

            ok = self._advance_sample(
                viewer=viewer,
                sample_dt=self.config.dt,
                step_simulation=True,
            )

            if not ok:
                return False

        if left_gripper_traj is not None:
            self.left_gripper_command = float(
                np.asarray(left_gripper_traj, dtype=float).reshape(-1)[-1]
            )

        if right_gripper_traj is not None:
            self.right_gripper_command = float(
                np.asarray(right_gripper_traj, dtype=float).reshape(-1)[-1]
            )

        return True

    def _hold(
        self,
        *,
        viewer: Any | None,
        duration: float,
    ) -> bool:
        """
        Hold current MuJoCo state for a short time.
        """
        if duration <= 0.0:
            return True

        n = max(1, int(np.ceil(duration / self.config.dt)))

        for _ in range(n):
            ok = self._advance_sample(
                viewer=viewer,
                sample_dt=self.config.dt,
                step_simulation=True,
            )

            if not ok:
                return False

        return True

    def execute_stage(
        self,
        stage: StageResult,
        *,
        viewer: Any | None = None,
    ) -> bool:
        """
        Execute one StageResult.

        Execution order:
            1. OCP reference if available.
            2. Else gripper-only trajectory if available.
            3. Return-arm trajectory if available.
            4. Optional stage pause.
        """
        if self.config.print_stage:
            print(f"[MuJoCo] stage: {stage.step_name}")

        # If OCP trajectory exists, replay it first.
        # If gripper command also exists in this stage, run it in parallel
        # with the OCP trajectory.
        if stage.has_ocp_reference:
            ok = self._replay_qv_trajectory(
                q_traj=stage.ocp_q_ref,
                v_traj=stage.ocp_v_ref,
                time_ref=stage.ocp_time_ref,
                left_gripper_traj=stage.left_gripper_command_traj,
                right_gripper_traj=stage.right_gripper_command_traj,
                viewer=viewer,
            )

            if not ok:
                return False

            gripper_was_consumed = stage.has_gripper_command
        else:
            gripper_was_consumed = False

        # If this stage only contains gripper motion, replay it here.
        if stage.has_gripper_command and not gripper_was_consumed:
            ok = self._replay_gripper_only(
                left_gripper_traj=stage.left_gripper_command_traj,
                right_gripper_traj=stage.right_gripper_command_traj,
                viewer=viewer,
            )

            if not ok:
                return False

        # Return-arm trajectory happens after OCP/gripper part.
        if stage.has_return_trajectory:
            ok = self._replay_qv_trajectory(
                q_traj=stage.q_return_traj,
                v_traj=stage.v_return_traj,
                time_ref=None,
                left_gripper_traj=None,
                right_gripper_traj=None,
                viewer=viewer,
            )

            if not ok:
                return False

        if self.config.pause_between_stages:
            ok = self._hold(
                viewer=viewer,
                duration=self.config.stage_pause_time,
            )

            if not ok:
                return False

        return True

    def execute_plan(
        self,
        plan: TaskPlanResult,
    ) -> None:
        """
        Execute a full task plan.

        If config.show_viewer=True, this opens a MuJoCo passive viewer.
        Otherwise execution is headless.
        """
        if self.config.show_viewer:
            if mujoco_viewer is None:
                raise RuntimeError(
                    "mujoco.viewer is not available in this environment."
                )

            with mujoco_viewer.launch_passive(self.model, self.data) as viewer:
                for stage in plan.stage_results:
                    ok = self.execute_stage(stage, viewer=viewer)

                    if not ok:
                        print("[MuJoCo] viewer closed. Stop execution.")
                        return

                self._hold(
                    viewer=viewer,
                    duration=self.config.hold_final_time,
                )

            return

        for stage in plan.stage_results:
            ok = self.execute_stage(stage, viewer=None)

            if not ok:
                return

        self._hold(
            viewer=None,
            duration=self.config.hold_final_time,
        )