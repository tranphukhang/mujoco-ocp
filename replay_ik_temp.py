from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from manipulation_ocp.configs.ik import DualEEIKConfig
from manipulation_ocp.configs.initial_guess import InitialGuessConfig
from manipulation_ocp.mujoco.loader import load_model as load_mujoco_model
from manipulation_ocp.mujoco.state_mapping import (
    build_state_mapping,
    get_default_replay_reference_state,
    ocp_trajectory_to_mujoco_trajectory,
)
from manipulation_ocp.pinocchio.ik import build_initial_guess_from_dual_ee_targets
from manipulation_ocp.pinocchio.kinematics import get_ee_positions
from manipulation_ocp.pinocchio.loader import load_model_and_data
from manipulation_ocp.robots.g1_gripper import (
    MUJOCO_REPLAY_XML,
    PINOCCHIO_MODEL_XML,
)


DEFAULT_Q_START = np.array(
    [
        0.0, 0.0, 0.0,
        0.2, 0.7853981634, 0.0, -0.5235987756, 0.0, 0.0, 0.0,
        0.2, -0.7853981634, 0.0, -0.5235987756, 0.0, 0.0, 0.0,
    ],
    dtype=float,
)


def parse_vec3(text: str) -> np.ndarray:
    parts = text.replace(",", " ").split()

    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Expected 3 values, got {len(parts)} from '{text}'"
        )

    try:
        return np.array([float(v) for v in parts], dtype=float)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Could not parse vec3 from '{text}'"
        ) from exc


def interpolate_trajectory(
    traj: np.ndarray,
    t_query: float,
    dt: float,
) -> np.ndarray:
    """
    Linear interpolation for trajectory sampled at dt.

    traj shape:
        (num_nodes, dim)
    """
    num_nodes = traj.shape[0]
    total_time = (num_nodes - 1) * dt

    if t_query <= 0.0:
        return traj[0].copy()

    if t_query >= total_time:
        return traj[-1].copy()

    s = t_query / dt
    k = int(np.floor(s))
    alpha = s - k

    k_next = min(k + 1, num_nodes - 1)

    return (1.0 - alpha) * traj[k] + alpha * traj[k_next]


def build_demo_trajectory(
    *,
    left_offset: np.ndarray,
    right_offset: np.ndarray,
    n_intervals: int,
    dt: float,
    verbose_ik: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """
    Build IK initial guess and map it to MuJoCo replay qpos/qvel trajectory.
    """
    pin_model, pin_data = load_model_and_data(PINOCCHIO_MODEL_XML)

    q_start = DEFAULT_Q_START.copy()

    ee_pos = get_ee_positions(pin_model, pin_data, q_start)
    left_start = ee_pos["left"]
    right_start = ee_pos["right"]

    # Convention:
    #   target positions are in Pinocchio pelvis/base frame.
    left_target = left_start + left_offset
    right_target = right_start + right_offset

    ik_config = DualEEIKConfig(verbose=verbose_ik)
    guess_config = InitialGuessConfig(
        n_intervals=n_intervals,
        dt=dt,
    )

    result = build_initial_guess_from_dual_ee_targets(
        pin_model,
        pin_data,
        q_start,
        left_target,
        right_target,
        ik_config=ik_config,
        guess_config=guess_config,
    )

    mj_model = load_mujoco_model(MUJOCO_REPLAY_XML)
    mapping = build_state_mapping(mj_model)

    qpos_ref, qvel_ref = get_default_replay_reference_state(mj_model)

    qpos_traj, qvel_traj = ocp_trajectory_to_mujoco_trajectory(
        mj_model,
        result.q_guess,
        result.v_guess,
        qpos_reference=qpos_ref,
        qvel_reference=qvel_ref,
        mapping=mapping,
    )

    info: dict[str, object] = {
        "pin_model": pin_model,
        "mj_model": mj_model,
        "q_start": q_start,
        "left_start": left_start,
        "right_start": right_start,
        "left_target": left_target,
        "right_target": right_target,
        "ik_result": result.ik_result,
        "dt": dt,
        "total_time": result.total_time,
        "n_intervals": n_intervals,
        "num_nodes": n_intervals + 1,
    }

    return qpos_traj, qvel_traj, info


def replay_trajectory(
    qpos_traj: np.ndarray,
    qvel_traj: np.ndarray,
    *,
    fps: float,
    traj_dt: float,
    loop: bool,
    pause_first: bool,
) -> None:
    """
    Replay full MuJoCo qpos/qvel trajectory by directly setting state.

    This is state replay, not actuator replay.
    """
    model = load_mujoco_model(MUJOCO_REPLAY_XML)
    data = mujoco.MjData(model)

    total_time = (qpos_traj.shape[0] - 1) * traj_dt
    render_dt = 1.0 / fps

    data.qpos[:] = qpos_traj[0]
    data.qvel[:] = qvel_traj[0]
    mujoco.mj_forward(model, data)

    print("[INFO] MuJoCo replay model:", Path(MUJOCO_REPLAY_XML))
    print("[INFO] qpos_traj shape =", qpos_traj.shape)
    print("[INFO] qvel_traj shape =", qvel_traj.shape)
    print("[INFO] trajectory dt =", traj_dt)
    print("[INFO] trajectory total time =", total_time)
    print("[INFO] render fps =", fps)
    print("[INFO] direct state replay, not actuator replay")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("[INFO] Viewer started")

        if pause_first:
            print("[INFO] Pause first enabled. Press Ctrl+C to stop.")
            print("[INFO] Close the viewer window to exit.")
            while viewer.is_running():
                viewer.sync()
                time.sleep(render_dt)

        print("[INFO] Replaying trajectory")

        while viewer.is_running():
            start_wall_time = time.perf_counter()

            while viewer.is_running():
                elapsed = time.perf_counter() - start_wall_time

                if elapsed > total_time:
                    break

                qpos = interpolate_trajectory(qpos_traj, elapsed, traj_dt)
                qvel = interpolate_trajectory(qvel_traj, elapsed, traj_dt)

                data.qpos[:] = qpos
                data.qvel[:] = qvel

                mujoco.mj_forward(model, data)
                viewer.sync()

                time.sleep(render_dt)

            # Hold final pose briefly.
            data.qpos[:] = qpos_traj[-1]
            data.qvel[:] = qvel_traj[-1]
            mujoco.mj_forward(model, data)
            viewer.sync()

            if not loop:
                print("[INFO] Replay finished. Holding final pose.")
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(render_dt)
                break

            time.sleep(0.5)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Temporary IK initial guess replay for G1 gripper model."
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Viewer render FPS. Default: 30.",
    )

    parser.add_argument(
        "--dt",
        type=float,
        default=0.04,
        help="Initial guess dt. Default: 0.04.",
    )

    parser.add_argument(
        "--n-intervals",
        type=int,
        default=50,
        help="Number of trajectory intervals. Default: 50.",
    )

    parser.add_argument(
        "--left-offset",
        type=parse_vec3,
        default=np.array([0.03, 0.0, 0.02], dtype=float),
        help="Left EE target offset in pelvis frame. Example: 0.03,0,0.02",
    )

    parser.add_argument(
        "--right-offset",
        type=parse_vec3,
        default=np.array([0.03, 0.0, 0.02], dtype=float),
        help="Right EE target offset in pelvis frame. Example: 0.03,0,0.02",
    )

    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop replay.",
    )

    parser.add_argument(
        "--pause-first",
        action="store_true",
        help="Open viewer and hold initial pose before replay.",
    )

    parser.add_argument(
        "--verbose-ik",
        action="store_true",
        help="Print IK iteration log.",
    )

    args = parser.parse_args()

    if args.fps <= 0.0:
        raise ValueError("--fps must be > 0")

    if args.dt <= 0.0:
        raise ValueError("--dt must be > 0")

    if args.n_intervals <= 0:
        raise ValueError("--n-intervals must be > 0")

    qpos_traj, qvel_traj, info = build_demo_trajectory(
        left_offset=args.left_offset,
        right_offset=args.right_offset,
        n_intervals=args.n_intervals,
        dt=args.dt,
        verbose_ik=args.verbose_ik,
    )

    ik_result = info["ik_result"]

    print("\n[IK INFO]")
    print("left_start  =", info["left_start"])
    print("right_start =", info["right_start"])
    print("left_target =", info["left_target"])
    print("right_target=", info["right_target"])
    print("converged   =", ik_result.converged)
    print("accepted    =", ik_result.accepted_for_initial_guess)
    print("fallback    =", ik_result.used_fallback_to_seed)
    print("iterations  =", ik_result.iterations)
    print("init error  =", ik_result.init_error_norm)
    print("final error =", ik_result.final_error_norm)
    print("best iter   =", ik_result.best_iter)

    replay_trajectory(
        qpos_traj,
        qvel_traj,
        fps=args.fps,
        traj_dt=args.dt,
        loop=args.loop,
        pause_first=args.pause_first,
    )


if __name__ == "__main__":
    main()