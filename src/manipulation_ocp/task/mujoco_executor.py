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
    LEFT_EE_SITE_NAME,
    MUJOCO_GRIPPER_ACTUATOR_NAMES,
    MUJOCO_POSITION_ACTUATOR_NAMES,
    RIGHT_EE_SITE_NAME,
    gripper_command_to_mujoco_ctrl,
    validate_mujoco_replay_model,
)
from manipulation_ocp.task.stages import StageResult, TaskPlanResult
from manipulation_ocp.utils.paths import G1_GRIPPER_SCENE_XML


@dataclass(frozen=True)
class MujocoTaskExecutorConfig:
    """
    MuJoCo task executor config.

    This executor is for visualization/replay, not for OCP solving.

    Defaults:
        - load assets/g1_gripper/scene.xml
        - reset to keyframe "home"
        - draw EE paths
        - draw current EE points
        - draw command target points
    """

    show_viewer: bool = True
    realtime: bool = True

    dt: float = 0.02

    # Reset to keyframe home immediately when loading/resetting MuJoCo.
    reset_keyframe: str | None = "home"

    pause_between_stages: bool = True
    stage_pause_time: float = 0.25
    hold_final_time: float = 0.5

    set_g1_position_ctrl: bool = True

    use_model_timestep_substeps: bool = True
    substeps_per_sample: int | None = None

    print_stage: bool = True

    # EE path line.
    draw_ee_path: bool = True
    ee_path_max_points: int = 3000
    ee_path_line_width: float = 0.006

    left_ee_path_rgba: tuple[float, float, float, float] = (
        0.0,
        0.75,
        1.0,
        1.0,
    )
    right_ee_path_rgba: tuple[float, float, float, float] = (
        1.0,
        0.45,
        0.0,
        1.0,
    )

    # Current EE point.
    draw_current_ee_points: bool = True
    current_ee_point_size: float = 0.018

    left_current_ee_rgba: tuple[float, float, float, float] = (
        0.0,
        1.0,
        1.0,
        1.0,
    )
    right_current_ee_rgba: tuple[float, float, float, float] = (
        1.0,
        0.75,
        0.0,
        1.0,
    )

    # Command target points.
    draw_command_points: bool = True
    command_point_size: float = 0.022

    left_command_point_rgba: tuple[float, float, float, float] = (
        0.0,
        0.2,
        1.0,
        1.0,
    )
    right_command_point_rgba: tuple[float, float, float, float] = (
        1.0,
        0.1,
        0.0,
        1.0,
    )

    # OCP command targets are produced in Pinocchio pelvis/base frame.
    # MuJoCo viewer.user_scn expects world-frame positions.
    command_points_are_in_base_frame: bool = True
    command_frame_body_name: str = "pelvis"

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

        if self.ee_path_max_points <= 1:
            raise ValueError(
                f"ee_path_max_points must be > 1, got {self.ee_path_max_points}"
            )

        for name, value in {
            "ee_path_line_width": self.ee_path_line_width,
            "current_ee_point_size": self.current_ee_point_size,
            "command_point_size": self.command_point_size,
        }.items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")

            if value <= 0.0:
                raise ValueError(f"{name} must be > 0, got {value}")

        if not isinstance(self.command_frame_body_name, str):
            raise TypeError(
                "command_frame_body_name must be a string, "
                f"got {type(self.command_frame_body_name)}"
            )

        if not self.command_frame_body_name.strip():
            raise ValueError("command_frame_body_name must be non-empty")


class MujocoTaskExecutor:
    """
    Replay a TaskPlanResult in MuJoCo.

    Supports:
        - OCP q/v reference replay
        - return-arm q/v replay
        - smooth gripper command replay
        - MuJoCo passive viewer
        - left/right EE path drawing
        - current EE point drawing
        - command target point drawing

    Replay strategy:
        G1 upper-body joints are applied by direct state replay using OCP q/v.
        Gripper is replayed through MuJoCo gripper actuator controls.
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

        self.left_ee_site_id = self._get_site_id(LEFT_EE_SITE_NAME)
        self.right_ee_site_id = self._get_site_id(RIGHT_EE_SITE_NAME)

        self.command_frame_body_id = self._get_body_id(
            self.config.command_frame_body_name,
        )

        self.left_ee_path: list[np.ndarray] = []
        self.right_ee_path: list[np.ndarray] = []

        self.left_command_points: list[np.ndarray] = []
        self.right_command_points: list[np.ndarray] = []

        self.left_gripper_command = self._validate_gripper_command(
            left_gripper_command,
            name="left_gripper_command",
        )
        self.right_gripper_command = self._validate_gripper_command(
            right_gripper_command,
            name="right_gripper_command",
        )

        # Reset to keyframe home immediately after load.
        self.reset()

    @classmethod
    def from_xml_path(
        cls,
        xml_path: str | Path = G1_GRIPPER_SCENE_XML,
        *,
        config: MujocoTaskExecutorConfig | None = None,
        left_gripper_command: float = 0.0,
        right_gripper_command: float = 0.0,
        validate_model: bool = True,
    ) -> MujocoTaskExecutor:
        """
        Load MuJoCo scene/replay model and create executor.

        Default XML:
            assets/g1_gripper/scene.xml
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

    def _get_site_id(self, name: str) -> int:
        site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            name,
        )

        if site_id < 0:
            raise ValueError(f"MuJoCo site not found: {name}")

        return int(site_id)

    def _get_body_id(self, name: str) -> int:
        body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            name,
        )

        if body_id < 0:
            raise ValueError(f"MuJoCo body not found: {name}")

        return int(body_id)

    def launch_viewer(self):
        """
        Open MuJoCo passive viewer.

        Usage:
            with executor.launch_viewer() as viewer:
                ...
        """
        if mujoco_viewer is None:
            raise RuntimeError(
                "mujoco.viewer is not available in this environment."
            )

        return mujoco_viewer.launch_passive(self.model, self.data)

    def reset(self) -> None:
        """
        Reset MuJoCo data to a stable replay state.

        If config.reset_keyframe is "home", MuJoCo is immediately reset to the
        home keyframe when the model is loaded.
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

        self.reset_ee_paths()
        self._record_ee_path_sample()

    def reset_ee_paths(self) -> None:
        """
        Clear recorded end-effector paths.
        """
        self.left_ee_path.clear()
        self.right_ee_path.clear()

    def reset_command_points(self) -> None:
        """
        Clear command target points.
        """
        self.left_command_points.clear()
        self.right_command_points.clear()

    @staticmethod
    def _append_unique_point(
        points: list[np.ndarray],
        point: np.ndarray,
        *,
        atol: float = 1e-9,
    ) -> None:
        p = np.asarray(point, dtype=float).reshape(3)

        for old in points:
            if np.linalg.norm(old - p) <= atol:
                return

        points.append(p.copy())

    def _base_point_to_world(self, point_base: np.ndarray) -> np.ndarray:
        """
        Convert a point from OCP base frame to MuJoCo world frame.

        OCP targets are generated in Pinocchio pelvis/base frame.
        MuJoCo viewer.user_scn expects world-frame positions.
        """
        p_base = np.asarray(point_base, dtype=float).reshape(3)

        body_pos = np.asarray(
            self.data.xpos[self.command_frame_body_id],
            dtype=float,
        ).reshape(3)

        body_rot = np.asarray(
            self.data.xmat[self.command_frame_body_id],
            dtype=float,
        ).reshape(3, 3)

        return body_pos + body_rot @ p_base

    def _command_point_to_world(self, point: np.ndarray) -> np.ndarray:
        """
        Convert command point to world frame before drawing.
        """
        p = np.asarray(point, dtype=float).reshape(3)

        if self.config.command_points_are_in_base_frame:
            return self._base_point_to_world(p)

        return p

    def collect_command_points_from_plan(
        self,
        plan: TaskPlanResult,
    ) -> None:
        """
        Collect left/right EE command target points from OCP solutions.

        OCP targets are in Pinocchio pelvis/base frame, so they are converted
        to MuJoCo world frame before drawing.
        """
        self.reset_command_points()

        mujoco.mj_forward(self.model, self.data)

        for stage in plan.stage_results:
            if stage.ocp_solution is None:
                continue

            left_target_world = self._command_point_to_world(
                stage.ocp_solution.left_ee_target,
            )
            right_target_world = self._command_point_to_world(
                stage.ocp_solution.right_ee_target,
            )

            self._append_unique_point(
                self.left_command_points,
                left_target_world,
            )
            self._append_unique_point(
                self.right_command_points,
                right_target_world,
            )

    def _record_ee_path_sample(self) -> None:
        """
        Record current left/right EE site positions.
        """
        if not self.config.draw_ee_path:
            return

        left_pos = np.asarray(
            self.data.site_xpos[self.left_ee_site_id],
            dtype=float,
        ).copy()

        right_pos = np.asarray(
            self.data.site_xpos[self.right_ee_site_id],
            dtype=float,
        ).copy()

        self.left_ee_path.append(left_pos)
        self.right_ee_path.append(right_pos)

        max_points = int(self.config.ee_path_max_points)

        if len(self.left_ee_path) > max_points:
            self.left_ee_path = self.left_ee_path[-max_points:]

        if len(self.right_ee_path) > max_points:
            self.right_ee_path = self.right_ee_path[-max_points:]

    def _add_line_to_user_scene(
        self,
        *,
        viewer: Any,
        p0: np.ndarray,
        p1: np.ndarray,
        rgba: tuple[float, float, float, float],
    ) -> None:
        """
        Add one line segment to viewer.user_scn.
        """
        if viewer is None or not hasattr(viewer, "user_scn"):
            return

        scn = viewer.user_scn

        if scn.ngeom >= scn.maxgeom:
            return

        p0_arr = np.asarray(p0, dtype=float).reshape(3)
        p1_arr = np.asarray(p1, dtype=float).reshape(3)

        if np.linalg.norm(p1_arr - p0_arr) < 1e-9:
            return

        geom = scn.geoms[scn.ngeom]

        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_LINE,
            float(self.config.ee_path_line_width),
            p0_arr,
            p1_arr,
        )

        geom.rgba[:] = np.asarray(rgba, dtype=float)
        scn.ngeom += 1

    def _add_sphere_to_user_scene(
        self,
        *,
        viewer: Any,
        position: np.ndarray,
        radius: float,
        rgba: tuple[float, float, float, float],
    ) -> None:
        """
        Add one sphere point to viewer.user_scn.
        """
        if viewer is None or not hasattr(viewer, "user_scn"):
            return

        scn = viewer.user_scn

        if scn.ngeom >= scn.maxgeom:
            return

        pos = np.asarray(position, dtype=float).reshape(3)
        size = np.array([float(radius), 0.0, 0.0], dtype=float)
        mat = np.eye(3, dtype=float).reshape(-1)
        rgba_arr = np.asarray(rgba, dtype=float)

        geom = scn.geoms[scn.ngeom]

        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            size,
            pos,
            mat,
            rgba_arr,
        )

        scn.ngeom += 1

    def _draw_ee_paths(self, viewer: Any | None) -> None:
        """
        Draw left/right EE paths into viewer.user_scn.
        """
        if not self.config.draw_ee_path:
            return

        if viewer is None or not hasattr(viewer, "user_scn"):
            return

        for i in range(len(self.left_ee_path) - 1):
            self._add_line_to_user_scene(
                viewer=viewer,
                p0=self.left_ee_path[i],
                p1=self.left_ee_path[i + 1],
                rgba=self.config.left_ee_path_rgba,
            )

        for i in range(len(self.right_ee_path) - 1):
            self._add_line_to_user_scene(
                viewer=viewer,
                p0=self.right_ee_path[i],
                p1=self.right_ee_path[i + 1],
                rgba=self.config.right_ee_path_rgba,
            )

    def _draw_command_points(self, viewer: Any | None) -> None:
        """
        Draw target command points.
        """
        if not self.config.draw_command_points:
            return

        if viewer is None or not hasattr(viewer, "user_scn"):
            return

        for p in self.left_command_points:
            self._add_sphere_to_user_scene(
                viewer=viewer,
                position=p,
                radius=self.config.command_point_size,
                rgba=self.config.left_command_point_rgba,
            )

        for p in self.right_command_points:
            self._add_sphere_to_user_scene(
                viewer=viewer,
                position=p,
                radius=self.config.command_point_size,
                rgba=self.config.right_command_point_rgba,
            )

    def _draw_current_ee_points(self, viewer: Any | None) -> None:
        """
        Draw current left/right EE site points.
        """
        if not self.config.draw_current_ee_points:
            return

        if viewer is None or not hasattr(viewer, "user_scn"):
            return

        left_pos = np.asarray(
            self.data.site_xpos[self.left_ee_site_id],
            dtype=float,
        )

        right_pos = np.asarray(
            self.data.site_xpos[self.right_ee_site_id],
            dtype=float,
        )

        self._add_sphere_to_user_scene(
            viewer=viewer,
            position=left_pos,
            radius=self.config.current_ee_point_size,
            rgba=self.config.left_current_ee_rgba,
        )

        self._add_sphere_to_user_scene(
            viewer=viewer,
            position=right_pos,
            radius=self.config.current_ee_point_size,
            rgba=self.config.right_current_ee_rgba,
        )

    def _draw_overlays(self, viewer: Any | None) -> None:
        """
        Draw all visual overlays.

        This resets viewer.user_scn every frame, then redraws:
            - command target points
            - EE paths
            - current EE points
        """
        if viewer is None or not hasattr(viewer, "user_scn"):
            return

        viewer.user_scn.ngeom = 0

        self._draw_command_points(viewer)
        self._draw_ee_paths(viewer)
        self._draw_current_ee_points(viewer)

    def _set_g1_position_ctrl_from_q(
        self,
        q_ocp: np.ndarray,
    ) -> None:
        """
        Set G1 position actuator controls from OCP q.

        Direct state replay remains the primary replay mechanism.
        This keeps MuJoCo actuators consistent when stepping.
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

        self._draw_overlays(viewer)
        viewer.sync()

        return True

    def keep_viewer_alive_until_closed(
        self,
        viewer: Any,
        *,
        print_message: bool = True,
    ) -> None:
        """
        Keep GUI alive after the plan finishes.

        The GUI only closes when the user manually closes the MuJoCo window.
        """
        if viewer is None:
            return

        if print_message:
            print("")
            print("[MuJoCo] Replay finished. Close the viewer window to exit.")

        while True:
            if hasattr(viewer, "is_running") and not viewer.is_running():
                return

            ok = self._sync_viewer(viewer)

            if not ok:
                return

            time.sleep(0.03)

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

        self._record_ee_path_sample()

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
        Replay a q/v trajectory, optionally with gripper commands.
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

            self._apply_ocp_state(
                q_ocp=q_arr[k],
                v_ocp=v_arr[k],
            )
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

        Order:
            1. OCP reference if available.
            2. Else gripper-only trajectory if available.
            3. Return-arm trajectory if available.
            4. Optional stage pause.
        """
        if self.config.print_stage:
            print(f"[MuJoCo] stage: {stage.step_name}")

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

        if stage.has_gripper_command and not gripper_was_consumed:
            ok = self._replay_gripper_only(
                left_gripper_traj=stage.left_gripper_command_traj,
                right_gripper_traj=stage.right_gripper_command_traj,
                viewer=viewer,
            )

            if not ok:
                return False

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
        *,
        viewer: Any | None = None,
    ) -> None:
        """
        Execute a full task plan.

        If viewer is provided, reuse it.
        If config.show_viewer=True and viewer is None, this opens a MuJoCo
        passive viewer.
        """
        self.reset_ee_paths()
        self.collect_command_points_from_plan(plan)
        self._record_ee_path_sample()

        if viewer is not None:
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

        if self.config.show_viewer:
            with self.launch_viewer() as local_viewer:
                for stage in plan.stage_results:
                    ok = self.execute_stage(stage, viewer=local_viewer)

                    if not ok:
                        print("[MuJoCo] viewer closed. Stop execution.")
                        return

                self._hold(
                    viewer=local_viewer,
                    duration=self.config.hold_final_time,
                )

                self.keep_viewer_alive_until_closed(local_viewer)

            return

        for stage in plan.stage_results:
            ok = self.execute_stage(stage, viewer=None)

            if not ok:
                return

        self._hold(
            viewer=None,
            duration=self.config.hold_final_time,
        )