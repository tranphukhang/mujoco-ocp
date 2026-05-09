from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FastDualEEOCPConfig:
    """
    Fast dual-end-effector reaching OCP config.

    Design goal:
        Solve quickly enough to generate reference trajectories for RL reward
        shaping, not to build a heavy high-accuracy planner.

    OCP model:
        q ∈ R17
        v ∈ R17
        x = [q; v] ∈ R34
        u = tau ∈ R17

    Discretization:
        Explicit Hermite-Simpson / cubic midpoint collocation.

    Important:
        No posture cost is used.
        Gripper is not part of the optimization.
    """

    # -------------------------------------------------------------------------
    # Mesh / time
    # -------------------------------------------------------------------------
    n_intervals: int = 20

    # Initial guess:
    #   h_init = tf_init / n_intervals = 0.4 / 20 = 0.02 s
    tf_init: float = 0.4
    tf_min: float = 0.4
    tf_max: float = 6.0

    # Keep this zero at the beginning to avoid pushing the solver to shrink
    # time too early.
    time_weight: float = 0.0

    # -------------------------------------------------------------------------
    # Stage EE tracking weights
    # -------------------------------------------------------------------------
    # Stage cost:
    #   ||p_left(q)  - p_left_ref||^2  * stage_ee_left_weight
    # + ||p_right(q) - p_right_ref||^2 * stage_ee_right_weight
    stage_ee_left_weight: float = 1.0
    stage_ee_right_weight: float = 1.0

    # -------------------------------------------------------------------------
    # Terminal EE tracking weights
    # -------------------------------------------------------------------------
    # Terminal cost is stronger than stage cost to make the final state close
    # to the target without using a hard terminal equality constraint.
    terminal_ee_left_weight: float = 10.0
    terminal_ee_right_weight: float = 10.0

    # -------------------------------------------------------------------------
    # Velocity regularization weights
    # -------------------------------------------------------------------------
    # Joint order:
    #   0     : waist_yaw_joint
    #   1, 2  : waist_roll_joint, waist_pitch_joint
    #   3..9  : left arm
    #   10..16: right arm
    #
    # waist_yaw is slightly cheaper so the optimizer can use it more,
    # but not too cheap to avoid abuse.
    stage_v_waist_yaw_weight: float = 0.05
    stage_v_waist_roll_pitch_weight: float = 0.1
    stage_v_arm_weight: float = 0.1

    # -------------------------------------------------------------------------
    # Torque regularization weights
    # -------------------------------------------------------------------------
    # waist_yaw torque is slightly cheaper, but not too cheap.
    stage_u_waist_yaw_weight: float = 0.005
    stage_u_waist_roll_pitch_weight: float = 0.01
    stage_u_arm_weight: float = 0.01

    # -------------------------------------------------------------------------
    # Terminal velocity weights
    # -------------------------------------------------------------------------
    # Do not make waist yaw cheaper here. At the final node, all joints should
    # settle down smoothly.
    terminal_v_waist_weight: float = 10.0
    terminal_v_arm_weight: float = 10.0

    # -------------------------------------------------------------------------
    # Dimensions
    # -------------------------------------------------------------------------
    nq: int = 17
    nv: int = 17
    nu: int = 17

    @property
    def nx(self) -> int:
        return self.nq + self.nv

    @property
    def num_nodes(self) -> int:
        return self.n_intervals + 1

    @property
    def h_init(self) -> float:
        return self.tf_init / self.n_intervals

    def validate(self) -> None:
        if self.n_intervals <= 0:
            raise ValueError("n_intervals must be > 0")

        if self.tf_init <= 0.0:
            raise ValueError("tf_init must be > 0")

        if self.tf_min <= 0.0:
            raise ValueError("tf_min must be > 0")

        if self.tf_max <= self.tf_min:
            raise ValueError("tf_max must be > tf_min")

        if not (self.tf_min <= self.tf_init <= self.tf_max):
            raise ValueError(
                "tf_init must satisfy tf_min <= tf_init <= tf_max. "
                f"Got tf_min={self.tf_min}, tf_init={self.tf_init}, "
                f"tf_max={self.tf_max}"
            )

        if self.time_weight < 0.0:
            raise ValueError("time_weight must be >= 0")

        weights = {
            "stage_ee_left_weight": self.stage_ee_left_weight,
            "stage_ee_right_weight": self.stage_ee_right_weight,
            "terminal_ee_left_weight": self.terminal_ee_left_weight,
            "terminal_ee_right_weight": self.terminal_ee_right_weight,
            "stage_v_waist_yaw_weight": self.stage_v_waist_yaw_weight,
            "stage_v_waist_roll_pitch_weight": (
                self.stage_v_waist_roll_pitch_weight
            ),
            "stage_v_arm_weight": self.stage_v_arm_weight,
            "stage_u_waist_yaw_weight": self.stage_u_waist_yaw_weight,
            "stage_u_waist_roll_pitch_weight": (
                self.stage_u_waist_roll_pitch_weight
            ),
            "stage_u_arm_weight": self.stage_u_arm_weight,
            "terminal_v_waist_weight": self.terminal_v_waist_weight,
            "terminal_v_arm_weight": self.terminal_v_arm_weight,
        }

        for name, value in weights.items():
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0, got {value}")

        if self.nq != 17 or self.nv != 17 or self.nu != 17:
            raise ValueError(
                "Current G1 upper-body OCP expects nq=nv=nu=17. "
                f"Got nq={self.nq}, nv={self.nv}, nu={self.nu}"
            )

    def build_stage_ee_weight_matrix(self) -> np.ndarray:
        """
        Build W_ee for stage cost.

        p_pair = [p_left; p_right] ∈ R6

        Cost:
            (p_pair - p_ref)^T W_ee (p_pair - p_ref)
        """
        W = np.zeros((6, 6), dtype=float)

        W[0:3, 0:3] = self.stage_ee_left_weight * np.eye(3)
        W[3:6, 3:6] = self.stage_ee_right_weight * np.eye(3)

        return W

    def build_terminal_ee_weight_matrix(self) -> np.ndarray:
        """
        Build W_ee_f for terminal EE cost.
        """
        W = np.zeros((6, 6), dtype=float)

        W[0:3, 0:3] = self.terminal_ee_left_weight * np.eye(3)
        W[3:6, 3:6] = self.terminal_ee_right_weight * np.eye(3)

        return W

    def build_stage_velocity_weight_matrix(self) -> np.ndarray:
        """
        Build W_v for stage velocity regularization.

        Joint order:
            0      : waist yaw
            1..2   : waist roll/pitch
            3..16  : arms
        """
        weights = np.zeros(self.nv, dtype=float)

        weights[0] = self.stage_v_waist_yaw_weight
        weights[1:3] = self.stage_v_waist_roll_pitch_weight
        weights[3:17] = self.stage_v_arm_weight

        return np.diag(weights)

    def build_stage_control_weight_matrix(self) -> np.ndarray:
        """
        Build W_u for stage torque regularization.

        Joint order:
            0      : waist yaw
            1..2   : waist roll/pitch
            3..16  : arms
        """
        weights = np.zeros(self.nu, dtype=float)

        weights[0] = self.stage_u_waist_yaw_weight
        weights[1:3] = self.stage_u_waist_roll_pitch_weight
        weights[3:17] = self.stage_u_arm_weight

        return np.diag(weights)

    def build_terminal_velocity_weight_matrix(self) -> np.ndarray:
        """
        Build W_v_f for terminal velocity cost.

        At the terminal node, all joints should settle down smoothly.
        """
        weights = np.zeros(self.nv, dtype=float)

        weights[0:3] = self.terminal_v_waist_weight
        weights[3:17] = self.terminal_v_arm_weight

        return np.diag(weights)


def get_default_fast_dual_ee_ocp_config() -> FastDualEEOCPConfig:
    """
    Return the default fast OCP config.
    """
    config = FastDualEEOCPConfig()
    config.validate()
    return config