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

    Robot model:
        q ∈ R17
        v ∈ R17
        x = [q; v] ∈ R34
        u = tau ∈ R17

    Transcription:
        Implicit midpoint direct collocation.

    Time discretization:
        The horizon is divided into N intervals.

        There are:
            N + 1 state nodes:
                X_0, X_1, ..., X_N

            N + 1 control nodes:
                U_0, U_1, ..., U_N

        For each interval k, implicit midpoint uses:

            x_mid    = 0.5 * (X_k + X_{k+1})
            u_mid    = 0.5 * (U_k + U_{k+1})
            xdot_mid = (X_{k+1} - X_k) / h

        and enforces:

            F_impl(x_mid, xdot_mid, u_mid) = 0

        where:

            F_impl =
            [
                qdot_mid - v_mid
                RNEA(q_mid, v_mid, vdot_mid) - u_mid
            ]

    Cost:
        Stage cost is evaluated at interval midpoints.

        Terminal cost is evaluated at the final node.

    Important:
        No hard terminal end-effector equality constraint is used.
        The end-effector target is tracked through stage and terminal costs.

        No posture cost is used.

        Gripper is not part of the optimization.
        Gripper commands are handled outside the OCP by the task planner.
    """

    # -------------------------------------------------------------------------
    # Mesh / time
    # -------------------------------------------------------------------------
    n_intervals: int = 20

    # Initial guess:
    #   h_init = tf_init / n_intervals = 0.4 / 20 = 0.02 s
    #
    # This matches the intended RL sampling time dt = 0.02 s.
    tf_init: float = 0.4
    tf_min: float = 0.4
    tf_max: float = 6.0

    # Keep this zero at the beginning to avoid pushing the solver to shrink
    # time too early. If this is increased, the optimizer will be encouraged
    # to reduce Tf.
    time_weight: float = 0.0

    # -------------------------------------------------------------------------
    # Stage EE tracking weights
    # -------------------------------------------------------------------------
    # Stage cost at midpoint:
    #
    #   ||p_left(q_mid)  - p_left_ref||^2  * stage_ee_left_weight
    # + ||p_right(q_mid) - p_right_ref||^2 * stage_ee_right_weight
    #
    # These weights affect the entire trajectory, not only the final node.
    stage_ee_left_weight: float = 1.0
    stage_ee_right_weight: float = 1.0

    # -------------------------------------------------------------------------
    # Terminal EE tracking weights
    # -------------------------------------------------------------------------
    # Terminal cost at final node:
    #
    #   ||p_left(q_N)  - p_left_ref||^2  * terminal_ee_left_weight
    # + ||p_right(q_N) - p_right_ref||^2 * terminal_ee_right_weight
    #
    # This is stronger than the stage cost to make the final state close to
    # the target without using a hard terminal equality constraint.
    terminal_ee_left_weight: float = 10.0
    terminal_ee_right_weight: float = 10.0

    # -------------------------------------------------------------------------
    # Velocity regularization weights
    # -------------------------------------------------------------------------
    # Joint order:
    #   0      : waist_yaw_joint
    #   1, 2   : waist_roll_joint, waist_pitch_joint
    #   3..9   : left arm
    #   10..16 : right arm
    #
    # waist_yaw is slightly cheaper so the optimizer can use it more,
    # but not too cheap to avoid abuse.
    stage_v_waist_yaw_weight: float = 0.05
    stage_v_waist_roll_pitch_weight: float = 0.1
    stage_v_arm_weight: float = 0.1

    # -------------------------------------------------------------------------
    # Torque regularization weights
    # -------------------------------------------------------------------------
    # Joint order:
    #   0      : waist_yaw_joint
    #   1, 2   : waist_roll_joint, waist_pitch_joint
    #   3..9   : left arm
    #   10..16 : right arm
    #
    # waist_yaw torque is slightly cheaper, but not too cheap.
    stage_u_waist_yaw_weight: float = 0.005
    stage_u_waist_roll_pitch_weight: float = 0.01
    stage_u_arm_weight: float = 0.01

    # -------------------------------------------------------------------------
    # Terminal velocity weights
    # -------------------------------------------------------------------------
    # At the final node, all joints should settle down smoothly.
    # Do not make waist yaw cheaper here.
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
        """
        State dimension.

        x = [q; v]
        """
        return self.nq + self.nv

    @property
    def num_nodes(self) -> int:
        """
        Number of state/control nodes.

        With N intervals, direct collocation uses N + 1 nodes.
        """
        return self.n_intervals + 1

    @property
    def h_init(self) -> float:
        """
        Initial time step guess.

        h_init = tf_init / n_intervals
        """
        return self.tf_init / self.n_intervals

    def validate(self) -> None:
        """
        Validate config values.
        """
        if self.n_intervals <= 0:
            raise ValueError("n_intervals must be > 0")

        time_values = {
            "tf_init": self.tf_init,
            "tf_min": self.tf_min,
            "tf_max": self.tf_max,
            "time_weight": self.time_weight,
        }

        for name, value in time_values.items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")

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
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")

            if value < 0.0:
                raise ValueError(f"{name} must be >= 0, got {value}")

        if self.nq != 17 or self.nv != 17 or self.nu != 17:
            raise ValueError(
                "Current G1 upper-body OCP expects nq=nv=nu=17. "
                f"Got nq={self.nq}, nv={self.nv}, nu={self.nu}"
            )

    def build_stage_ee_weight_matrix(self) -> np.ndarray:
        """
        Build W_ee for midpoint stage cost.

        p_pair = [p_left; p_right] ∈ R6

        Cost term:
            (p_pair - p_ref)^T W_ee (p_pair - p_ref)
        """
        W = np.zeros((6, 6), dtype=float)

        W[0:3, 0:3] = self.stage_ee_left_weight * np.eye(3)
        W[3:6, 3:6] = self.stage_ee_right_weight * np.eye(3)

        return W

    def build_terminal_ee_weight_matrix(self) -> np.ndarray:
        """
        Build W_ee_f for terminal EE cost.

        Terminal cost is evaluated at q_N.
        """
        W = np.zeros((6, 6), dtype=float)

        W[0:3, 0:3] = self.terminal_ee_left_weight * np.eye(3)
        W[3:6, 3:6] = self.terminal_ee_right_weight * np.eye(3)

        return W

    def build_stage_velocity_weight_matrix(self) -> np.ndarray:
        """
        Build W_v for midpoint velocity regularization.

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
        Build W_u for midpoint torque regularization.

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
    Return the default fast dual-EE OCP config.
    """
    config = FastDualEEOCPConfig()
    config.validate()
    return config