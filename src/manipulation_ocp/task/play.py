from __future__ import annotations

import argparse
import time

import numpy as np

from manipulation_ocp.casadi.pinocchio_dynamics import (
    build_casadi_pinocchio_dynamics,
)
from manipulation_ocp.casadi.pinocchio_kinematics import (
    build_casadi_pinocchio_kinematics,
)
from manipulation_ocp.configs.ik import DualEEIKConfig
from manipulation_ocp.configs.initial_guess import InitialGuessConfig
from manipulation_ocp.configs.ocp import FastDualEEOCPConfig
from manipulation_ocp.ocp.fast_dual_ee_problem import FastDualEEOCP
from manipulation_ocp.pinocchio.kinematics import get_ee_positions
from manipulation_ocp.pinocchio.loader import load_model_and_data
from manipulation_ocp.robots.g1_gripper import PINOCCHIO_MODEL_XML
from manipulation_ocp.task.mujoco_executor import (
    MujocoTaskExecutor,
    MujocoTaskExecutorConfig,
)
from manipulation_ocp.task.planner import PickPlaceTaskPlanner, TaskPlannerConfig
from manipulation_ocp.task.stages import (
    GripperAction,
    HoldAction,
    LiftAction,
    ParallelArmStep,
    ReachAction,
    ReturnArmAction,
)


def build_default_q0() -> np.ndarray:
    """
    Default 17-DoF OCP initial configuration.

    Joint order:
        0..2   : waist
        3..9   : left arm
        10..16 : right arm
    """
    return np.array(
        [
            0.0,
            0.0,
            0.0,
            0.2,
            0.7853981634,
            0.0,
            -0.5235987756,
            0.0,
            0.0,
            0.0,
            0.2,
            -0.7853981634,
            0.0,
            -0.5235987756,
            0.0,
            0.0,
            0.0,
        ],
        dtype=float,
    )


def build_demo_steps(
    *,
    left_home: np.ndarray,
    right_home: np.ndarray,
) -> list[ParallelArmStep]:
    """
    Build demo staggered dual-arm pipeline.

    Timeline:
        1. left reaches object, right holds home
        2. left closes gripper
        3. left lifts while right reaches object
        4. right closes gripper
        5. left opens while right lifts
        6. left returns home
        7. right opens gripper
        8. right returns home
    """
    left_pick_target = left_home + np.array([0.03, 0.00, 0.02])
    right_pick_target = right_home + np.array([0.03, 0.00, 0.02])

    return [
        ParallelArmStep(
            name="left_reach_object",
            left=ReachAction(side="left", target=left_pick_target),
            right=HoldAction(side="right", mode="home"),
        ),
        ParallelArmStep(
            name="left_close_gripper",
            left=GripperAction(side="left", command=1.0, duration=0.4),
            right=HoldAction(side="right", mode="current"),
        ),
        ParallelArmStep(
            name="left_lift_right_reach",
            left=LiftAction(side="left", dz=0.04),
            right=ReachAction(side="right", target=right_pick_target),
        ),
        ParallelArmStep(
            name="right_close_gripper",
            left=HoldAction(side="left", mode="current"),
            right=GripperAction(side="right", command=1.0, duration=0.4),
        ),
        ParallelArmStep(
            name="left_open_right_lift",
            left=GripperAction(side="left", command=0.0, duration=0.4),
            right=LiftAction(side="right", dz=0.04),
        ),
        ParallelArmStep(
            name="left_return_home",
            left=ReturnArmAction(side="left", duration=0.6),
            right=HoldAction(side="right", mode="current"),
        ),
        ParallelArmStep(
            name="right_open_gripper",
            left=HoldAction(side="left", mode="current"),
            right=GripperAction(side="right", command=0.0, duration=0.4),
        ),
        ParallelArmStep(
            name="right_return_home",
            left=HoldAction(side="left", mode="current"),
            right=ReturnArmAction(side="right", duration=0.6),
        ),
    ]


def print_plan_summary(result) -> None:
    print("")
    print("===== TASK PLAN SUMMARY =====")
    print("num stages:", result.num_stages)
    print("num ocp stages:", result.num_ocp_stages)
    print("successful ocp stages:", result.successful_ocp_stages)
    print("final_state shape:", result.final_state.shape)

    for i, stage in enumerate(result.stage_results):
        print("")
        print("=" * 80)
        print("stage index:", i)
        print("stage name:", stage.step_name)
        print("left action:", type(stage.left_action).__name__)
        print("right action:", type(stage.right_action).__name__)
        print("x_start shape:", stage.x_start.shape)
        print("x_end shape:", stage.x_end.shape)

        print("has ocp solution:", stage.ocp_solution is not None)

        if stage.ocp_solution is not None:
            print("ocp success:", stage.ocp_solution.success)
            print("ocp status:", stage.ocp_solution.status)
            print("ocp objective:", stage.ocp_solution.objective)
            print("ocp tf:", stage.ocp_solution.tf)
            print("ocp h:", stage.ocp_solution.h)

        print("has ocp reference:", stage.has_ocp_reference)

        if stage.ocp_q_ref is not None:
            print("ocp time shape:", stage.ocp_time_ref.shape)
            print("ocp q ref shape:", stage.ocp_q_ref.shape)
            print("ocp v ref shape:", stage.ocp_v_ref.shape)
            print(
                "ocp u ref shape:",
                None if stage.ocp_u_ref is None else stage.ocp_u_ref.shape,
            )
            print(
                "ocp dt unique:",
                np.unique(np.round(np.diff(stage.ocp_time_ref), 10)),
            )

        print("has gripper command:", stage.has_gripper_command)

        if stage.left_gripper_command_traj is not None:
            print("left gripper shape:", stage.left_gripper_command_traj.shape)
            print(
                "left gripper first,last:",
                stage.left_gripper_command_traj[0],
                stage.left_gripper_command_traj[-1],
            )

        if stage.right_gripper_command_traj is not None:
            print("right gripper shape:", stage.right_gripper_command_traj.shape)
            print(
                "right gripper first,last:",
                stage.right_gripper_command_traj[0],
                stage.right_gripper_command_traj[-1],
            )

        print("has return trajectory:", stage.has_return_trajectory)

        if stage.q_return_traj is not None:
            print("return q shape:", stage.q_return_traj.shape)
            print("return v shape:", stage.v_return_traj.shape)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run G1 dual-arm OCP task pipeline.",
    )

    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open MuJoCo passive viewer.",
    )

    parser.add_argument(
        "--no-execute",
        action="store_true",
        help="Only build and run planner. Do not execute MuJoCo replay.",
    )

    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Replay in real time. By default headless replay is not realtime.",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print detailed per-stage summary.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dt = 0.02

    print("Loading Pinocchio/OCP model...")
    pin_model, pin_data = load_model_and_data(PINOCCHIO_MODEL_XML)

    print("Building CasADi dynamics/kinematics...")
    dyn = build_casadi_pinocchio_dynamics(pin_model, name_prefix="g1")
    kin = build_casadi_pinocchio_kinematics(pin_model, name_prefix="g1")

    ocp_cfg = FastDualEEOCPConfig(
        n_intervals=20,
        tf_init=0.4,
        tf_min=0.4,
        tf_max=6.0,
    )

    solver_options = {
        "ipopt.print_level": 0,
        "print_time": True,
        "ipopt.max_iter": 100,
        "ipopt.tol": 1e-4,
        "ipopt.acceptable_tol": 1e-3,
        "ipopt.acceptable_iter": 5,
        "expand": True,
    }

    ocp = FastDualEEOCP(
        dyn,
        kin,
        config=ocp_cfg,
        solver_options=solver_options,
    )

    print("Building OCP problem...")
    t0 = time.perf_counter()
    ocp.build_problem()
    print("ocp build time:", time.perf_counter() - t0)

    q0 = build_default_q0()
    v0 = np.zeros(17)
    x0 = np.concatenate([q0, v0])

    ee_home = get_ee_positions(pin_model, pin_data, q0)
    left_home = np.asarray(ee_home["left"], dtype=float).reshape(3)
    right_home = np.asarray(ee_home["right"], dtype=float).reshape(3)

    planner = PickPlaceTaskPlanner(
        ocp=ocp,
        pin_model=pin_model,
        pin_data=pin_data,
        q_initial=q0,
        config=TaskPlannerConfig(dt=dt),
        ik_config=DualEEIKConfig(verbose=False),
        initial_guess_config=InitialGuessConfig(n_intervals=20, dt=dt),
    )

    steps = build_demo_steps(
        left_home=left_home,
        right_home=right_home,
    )

    print("Running task planner...")
    t0 = time.perf_counter()
    result = planner.run(
        x0_start=x0,
        steps=steps,
    )
    print("planner run time:", time.perf_counter() - t0)

    if not args.quiet:
        print_plan_summary(result)

    if args.no_execute:
        print("")
        print("play.py planner test OK")
        return

    print("")
    print("Executing MuJoCo replay...")

    executor = MujocoTaskExecutor.from_xml_path(
        config=MujocoTaskExecutorConfig(
            show_viewer=bool(args.viewer),
            realtime=bool(args.realtime or args.viewer),
            pause_between_stages=True,
            stage_pause_time=0.25,
            hold_final_time=0.5,
            print_stage=True,
        ),
    )

    executor.execute_plan(result)

    print("")
    print("play.py full pipeline OK")


if __name__ == "__main__":
    main()