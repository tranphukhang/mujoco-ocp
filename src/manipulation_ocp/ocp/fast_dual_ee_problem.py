from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import casadi as ca
import numpy as np

from manipulation_ocp.casadi.pinocchio_dynamics import (
    CasadiPinocchioDynamics,
)
from manipulation_ocp.casadi.pinocchio_kinematics import (
    CasadiPinocchioKinematics,
)
from manipulation_ocp.configs.ocp import (
    FastDualEEOCPConfig,
    get_default_fast_dual_ee_ocp_config,
)
from manipulation_ocp.ocp.costs import (
    build_dual_ee_stage_cost,
    build_dual_ee_terminal_cost,
)
from manipulation_ocp.ocp.direct_collocation import (
    compute_hermite_simpson_interval,
)
from manipulation_ocp.pinocchio.limits import (
    get_effort_limits,
    get_position_limits,
    get_velocity_limits,
)
from manipulation_ocp.utils.numerics import as_vector


@dataclass(frozen=True)
class OCPInitialGuessData:
    """
    Initial guess data for the OCP NLP.

    Shapes:
        X_nodes = (N + 1, nx)
        U_nodes = (N + 1, nu)
    """

    tf: float
    X_nodes: np.ndarray
    U_nodes: np.ndarray


@dataclass(frozen=True)
class FastDualEEOCPSolution:
    """
    Parsed OCP solution.
    """

    success: bool
    status: str
    objective: float

    tf: float
    h: float

    X_nodes: np.ndarray
    U_nodes: np.ndarray

    q_nodes: np.ndarray
    v_nodes: np.ndarray

    raw_solution: dict[str, Any]


def make_initial_guess_data(
    q_guess: np.ndarray,
    v_guess: np.ndarray,
    u_guess: np.ndarray,
    *,
    tf: float,
) -> OCPInitialGuessData:
    """
    Build OCP initial guess from IK/RNEA result.

    Expected:
        q_guess = (N + 1, nq)
        v_guess = (N + 1, nv)
        u_guess = (N + 1, nu)

    Output:
        X_nodes = [q_guess, v_guess]
    """
    q_arr = np.asarray(q_guess, dtype=float)
    v_arr = np.asarray(v_guess, dtype=float)
    u_arr = np.asarray(u_guess, dtype=float)

    if q_arr.ndim != 2:
        raise ValueError(f"q_guess must be 2D, got {q_arr.shape}")

    if v_arr.shape != q_arr.shape:
        raise ValueError(
            f"v_guess must have shape {q_arr.shape}, got {v_arr.shape}"
        )

    if u_arr.ndim != 2:
        raise ValueError(f"u_guess must be 2D, got {u_arr.shape}")

    if u_arr.shape[0] == q_arr.shape[0] - 1:
        # Allow interval controls; pad final control.
        u_arr = np.vstack([u_arr, u_arr[-1:]])

    if u_arr.shape[0] != q_arr.shape[0]:
        raise ValueError(
            "u_guess must have either N+1 or N rows. "
            f"Got q rows={q_arr.shape[0]}, u rows={u_arr.shape[0]}"
        )

    X_nodes = np.hstack([q_arr, v_arr])

    return OCPInitialGuessData(
        tf=float(tf),
        X_nodes=X_nodes,
        U_nodes=u_arr,
    )


def _as_matrix(
    value: Sequence[Sequence[float]] | np.ndarray,
    *,
    shape: tuple[int, int],
    name: str,
) -> np.ndarray:
    arr = np.asarray(value, dtype=float)

    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")

    return arr


class FastDualEEOCP:
    """
    Fast dual-end-effector reaching OCP.

    Design goal:
        Generate reference trajectories for RL reward shaping.
        Prioritize speed and stability over heavy planner accuracy.

    This version intentionally has:
        - no collision constraints
        - no SQP outer loop
        - no trust region
        - no hard terminal EE equality
        - no posture cost
        - no gripper decision variables

    Variables:
        Tf
        X_k = [q_k; v_k] ∈ R34, k = 0,...,N
        U_k = tau_k       ∈ R17, k = 0,...,N

    Dynamics discretization:
        explicit Hermite-Simpson / cubic midpoint.

    Dynamics:
        xdot = f(x, u) = [v; ABA(q, v, tau)]
    """

    def __init__(
        self,
        dynamics: CasadiPinocchioDynamics,
        kinematics: CasadiPinocchioKinematics,
        *,
        config: FastDualEEOCPConfig | None = None,
        solver_name: str = "ipopt",
        solver_options: dict[str, Any] | None = None,
    ) -> None:
        if config is None:
            config = get_default_fast_dual_ee_ocp_config()

        config.validate()

        if dynamics.nq != config.nq or dynamics.nv != config.nv:
            raise ValueError(
                "Dynamics dimensions do not match config. "
                f"dyn nq={dynamics.nq}, nv={dynamics.nv}; "
                f"cfg nq={config.nq}, nv={config.nv}"
            )

        if kinematics.nq != config.nq or kinematics.nv != config.nv:
            raise ValueError(
                "Kinematics dimensions do not match config. "
                f"kin nq={kinematics.nq}, nv={kinematics.nv}; "
                f"cfg nq={config.nq}, nv={config.nv}"
            )

        self.dynamics = dynamics
        self.kinematics = kinematics
        self.config = config

        self.nq = config.nq
        self.nv = config.nv
        self.nu = config.nu
        self.nx = config.nx

        self.N = config.n_intervals
        self.num_nodes = config.num_nodes

        self.solver_name = solver_name

        default_solver_options: dict[str, Any] = {
            # Fast/debug-friendly defaults.
            "ipopt.print_level": 5,
            "ipopt.max_iter": 300,
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-3,
            "ipopt.acceptable_iter": 10,
            "print_time": True,
        }

        if solver_options is not None:
            default_solver_options.update(solver_options)

        self.solver_options = default_solver_options

        self.nlp: dict[str, Any] | None = None
        self.solver: ca.Function | None = None

        self.problem_data: dict[str, Any] | None = None
        self.solution: dict[str, Any] | None = None

    def _build_default_bounds(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build x/u bounds from Pinocchio model.

        x = [q; v]
        u = tau
        """
        model = self.dynamics.model

        q_min, q_max = get_position_limits(model)

        v_min, v_max = get_velocity_limits(
            model,
            default_if_invalid=10.0,
        )

        u_min, u_max = get_effort_limits(model)

        x_min = np.concatenate([q_min, v_min])
        x_max = np.concatenate([q_max, v_max])

        return x_min, x_max, u_min, u_max

    def _parse_initial_guess(
        self,
        *,
        x0_value: np.ndarray,
        initial_guess: OCPInitialGuessData | None,
    ) -> OCPInitialGuessData:
        """
        Normalize initial guess.

        If no initial guess is provided:
            X_k = x0
            U_k = 0
        """
        if initial_guess is None:
            X_nodes = np.tile(
                x0_value.reshape(1, self.nx),
                (self.num_nodes, 1),
            )
            U_nodes = np.zeros((self.num_nodes, self.nu), dtype=float)

            return OCPInitialGuessData(
                tf=self.config.tf_init,
                X_nodes=X_nodes,
                U_nodes=U_nodes,
            )

        X_nodes = _as_matrix(
            initial_guess.X_nodes,
            shape=(self.num_nodes, self.nx),
            name="initial_guess.X_nodes",
        )

        U_nodes = _as_matrix(
            initial_guess.U_nodes,
            shape=(self.num_nodes, self.nu),
            name="initial_guess.U_nodes",
        )

        tf = float(initial_guess.tf)

        if not np.isfinite(tf):
            raise ValueError(f"initial_guess.tf must be finite, got {tf}")

        return OCPInitialGuessData(
            tf=tf,
            X_nodes=X_nodes,
            U_nodes=U_nodes,
        )

    def _clip_initial_guess_to_bounds(
        self,
        guess: OCPInitialGuessData,
        *,
        x_min: np.ndarray,
        x_max: np.ndarray,
        u_min: np.ndarray,
        u_max: np.ndarray,
    ) -> OCPInitialGuessData:
        tf = float(np.clip(guess.tf, self.config.tf_min, self.config.tf_max))
        X_nodes = np.clip(guess.X_nodes, x_min[None, :], x_max[None, :])
        U_nodes = np.clip(guess.U_nodes, u_min[None, :], u_max[None, :])

        return OCPInitialGuessData(
            tf=tf,
            X_nodes=X_nodes,
            U_nodes=U_nodes,
        )

    def build_problem(
        self,
        *,
        x0_value: Sequence[float] | np.ndarray,
        left_ee_target: Sequence[float] | np.ndarray,
        right_ee_target: Sequence[float] | np.ndarray,
        initial_guess: OCPInitialGuessData | None = None,
        x_min: Sequence[float] | np.ndarray | None = None,
        x_max: Sequence[float] | np.ndarray | None = None,
        u_min: Sequence[float] | np.ndarray | None = None,
        u_max: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        """
        Build the fast dual-EE OCP NLP.

        Parameters
        ----------
        x0_value:
            Initial state, shape (34,).

        left_ee_target, right_ee_target:
            Target positions in Pinocchio pelvis/base frame, shape (3,).

        initial_guess:
            Warm-start data from IK/RNEA.

        x_min, x_max:
            Optional state bounds, shape (34,).

        u_min, u_max:
            Optional torque bounds, shape (17,).
        """
        cfg = self.config

        x0_vec = as_vector(x0_value, size=self.nx, name="x0_value")

        left_target = as_vector(left_ee_target, size=3, name="left_ee_target")
        right_target = as_vector(
            right_ee_target,
            size=3,
            name="right_ee_target",
        )

        p_ref_np = np.concatenate([left_target, right_target])
        p_ref = ca.DM(p_ref_np)

        default_x_min, default_x_max, default_u_min, default_u_max = (
            self._build_default_bounds()
        )

        x_min_vec = (
            default_x_min
            if x_min is None
            else as_vector(x_min, size=self.nx, name="x_min")
        )

        x_max_vec = (
            default_x_max
            if x_max is None
            else as_vector(x_max, size=self.nx, name="x_max")
        )

        u_min_vec = (
            default_u_min
            if u_min is None
            else as_vector(u_min, size=self.nu, name="u_min")
        )

        u_max_vec = (
            default_u_max
            if u_max is None
            else as_vector(u_max, size=self.nu, name="u_max")
        )

        guess = self._parse_initial_guess(
            x0_value=x0_vec,
            initial_guess=initial_guess,
        )

        guess = self._clip_initial_guess_to_bounds(
            guess,
            x_min=x_min_vec,
            x_max=x_max_vec,
            u_min=u_min_vec,
            u_max=u_max_vec,
        )

        # ---------------------------------------------------------------------
        # Weight matrices
        # ---------------------------------------------------------------------
        W_ee = ca.DM(cfg.build_stage_ee_weight_matrix())
        W_ee_f = ca.DM(cfg.build_terminal_ee_weight_matrix())
        W_v = ca.DM(cfg.build_stage_velocity_weight_matrix())
        W_u = ca.DM(cfg.build_stage_control_weight_matrix())
        W_v_f = ca.DM(cfg.build_terminal_velocity_weight_matrix())

        # ---------------------------------------------------------------------
        # NLP containers
        # ---------------------------------------------------------------------
        w: list[ca.MX] = []
        w0: list[float] = []
        lbw: list[float] = []
        ubw: list[float] = []

        g: list[ca.MX] = []
        lbg: list[float] = []
        ubg: list[float] = []

        index_map: dict[str, Any] = {
            "Tf": None,
            "X_nodes": [],
            "U_nodes": [],
        }

        current_index = 0
        J = 0

        # ---------------------------------------------------------------------
        # Time variable
        # ---------------------------------------------------------------------
        Tf = ca.MX.sym("Tf")
        w.append(Tf)

        w0.append(float(guess.tf))
        lbw.append(float(cfg.tf_min))
        ubw.append(float(cfg.tf_max))

        index_map["Tf"] = (current_index, current_index + 1)
        current_index += 1

        h = Tf / cfg.n_intervals

        if cfg.time_weight > 0.0:
            J += float(cfg.time_weight) * Tf

        # ---------------------------------------------------------------------
        # State nodes
        # ---------------------------------------------------------------------
        X: list[ca.MX] = []

        for k in range(self.num_nodes):
            Xk = ca.MX.sym(f"X_{k}", self.nx)
            X.append(Xk)
            w.append(Xk)

            if k == 0:
                # Initial state enforced as variable bounds.
                lbw.extend(x0_vec.tolist())
                ubw.extend(x0_vec.tolist())
                w0.extend(x0_vec.tolist())
            else:
                lbw.extend(x_min_vec.tolist())
                ubw.extend(x_max_vec.tolist())
                w0.extend(guess.X_nodes[k].tolist())

            index_map["X_nodes"].append(
                (current_index, current_index + self.nx)
            )
            current_index += self.nx

        # ---------------------------------------------------------------------
        # Control nodes
        # ---------------------------------------------------------------------
        U: list[ca.MX] = []

        for k in range(self.num_nodes):
            Uk = ca.MX.sym(f"U_{k}", self.nu)
            U.append(Uk)
            w.append(Uk)

            lbw.extend(u_min_vec.tolist())
            ubw.extend(u_max_vec.tolist())
            w0.extend(guess.U_nodes[k].tolist())

            index_map["U_nodes"].append(
                (current_index, current_index + self.nu)
            )
            current_index += self.nu

        # ---------------------------------------------------------------------
        # Dynamics constraints + Simpson stage cost
        # ---------------------------------------------------------------------
        f_dyn = self.dynamics.explicit_dynamics_fn

        for k in range(cfg.n_intervals):
            xk = X[k]
            xkp1 = X[k + 1]

            uk = U[k]
            ukp1 = U[k + 1]

            defect, xc, uc, _, _, _ = compute_hermite_simpson_interval(
                f_dyn=f_dyn,
                xk=xk,
                xk_next=xkp1,
                uk=uk,
                uk_next=ukp1,
                h=h,
            )

            g.append(defect)
            lbg.extend(np.zeros(self.nx).tolist())
            ubg.extend(np.zeros(self.nx).tolist())

            Lk = build_dual_ee_stage_cost(
                x=xk,
                u=uk,
                kinematics=self.kinematics,
                p_ref=p_ref,
                W_ee=W_ee,
                W_v=W_v,
                W_u=W_u,
                nq=self.nq,
                nv=self.nv,
            )

            Lc = build_dual_ee_stage_cost(
                x=xc,
                u=uc,
                kinematics=self.kinematics,
                p_ref=p_ref,
                W_ee=W_ee,
                W_v=W_v,
                W_u=W_u,
                nq=self.nq,
                nv=self.nv,
            )

            Lkp1 = build_dual_ee_stage_cost(
                x=xkp1,
                u=ukp1,
                kinematics=self.kinematics,
                p_ref=p_ref,
                W_ee=W_ee,
                W_v=W_v,
                W_u=W_u,
                nq=self.nq,
                nv=self.nv,
            )

            J += (h / 6.0) * (Lk + 4.0 * Lc + Lkp1)

        # ---------------------------------------------------------------------
        # Terminal cost
        # ---------------------------------------------------------------------
        J += build_dual_ee_terminal_cost(
            x=X[-1],
            kinematics=self.kinematics,
            p_ref=p_ref,
            W_ee_f=W_ee_f,
            W_v_f=W_v_f,
            nq=self.nq,
            nv=self.nv,
        )

        # ---------------------------------------------------------------------
        # Build NLP solver
        # ---------------------------------------------------------------------
        w_cat = ca.vertcat(*w)
        g_cat = ca.vertcat(*g) if len(g) > 0 else ca.MX.zeros(0, 1)

        self.nlp = {
            "x": w_cat,
            "f": J,
            "g": g_cat,
        }

        self.solver = ca.nlpsol(
            "solver",
            self.solver_name,
            self.nlp,
            self.solver_options,
        )

        self.problem_data = {
            "N": cfg.n_intervals,
            "num_nodes": self.num_nodes,
            "nq": self.nq,
            "nv": self.nv,
            "nu": self.nu,
            "nx": self.nx,
            "method": "explicit_hermite_simpson",
            "collision_enabled": False,
            "tf_init": float(guess.tf),
            "tf_min": float(cfg.tf_min),
            "tf_max": float(cfg.tf_max),
            "time_weight": float(cfg.time_weight),
            "left_ee_target": left_target.copy(),
            "right_ee_target": right_target.copy(),
            "p_ref": p_ref_np.copy(),
            "x0_value": x0_vec.copy(),
            "x_min": x_min_vec.copy(),
            "x_max": x_max_vec.copy(),
            "u_min": u_min_vec.copy(),
            "u_max": u_max_vec.copy(),
            "W_ee": np.asarray(W_ee),
            "W_ee_f": np.asarray(W_ee_f),
            "W_v": np.asarray(W_v),
            "W_u": np.asarray(W_u),
            "W_v_f": np.asarray(W_v_f),
            "w0": np.asarray(w0, dtype=float),
            "lbw": np.asarray(lbw, dtype=float),
            "ubw": np.asarray(ubw, dtype=float),
            "lbg": np.asarray(lbg, dtype=float),
            "ubg": np.asarray(ubg, dtype=float),
            "index_map": index_map,
        }

    def solve(self) -> FastDualEEOCPSolution:
        """
        Solve the NLP.
        """
        if self.solver is None or self.problem_data is None:
            raise RuntimeError("Call build_problem(...) before solve().")

        pd = self.problem_data

        sol = self.solver(
            x0=pd["w0"],
            lbx=pd["lbw"],
            ubx=pd["ubw"],
            lbg=pd["lbg"],
            ubg=pd["ubg"],
        )

        self.solution = sol

        return self.extract_solution(sol)

    def extract_solution(
        self,
        sol: dict[str, Any] | None = None,
    ) -> FastDualEEOCPSolution:
        """
        Parse NLP solution into trajectories.
        """
        if self.problem_data is None:
            raise RuntimeError("No problem_data. Call build_problem first.")

        if sol is None:
            if self.solution is None:
                raise RuntimeError("No solution available.")
            sol = self.solution

        w_opt = np.asarray(sol["x"], dtype=float).reshape(-1)

        index_map = self.problem_data["index_map"]

        tf_slice = index_map["Tf"]
        tf = float(w_opt[tf_slice[0] : tf_slice[1]][0])
        h = tf / self.config.n_intervals

        X_nodes = np.zeros((self.num_nodes, self.nx), dtype=float)
        U_nodes = np.zeros((self.num_nodes, self.nu), dtype=float)

        for k, sl in enumerate(index_map["X_nodes"]):
            X_nodes[k] = w_opt[sl[0] : sl[1]]

        for k, sl in enumerate(index_map["U_nodes"]):
            U_nodes[k] = w_opt[sl[0] : sl[1]]

        q_nodes = X_nodes[:, 0 : self.nq]
        v_nodes = X_nodes[:, self.nq : self.nq + self.nv]

        stats = self.solver.stats() if self.solver is not None else {}
        status = str(stats.get("return_status", "unknown"))
        success = bool(stats.get("success", False))

        return FastDualEEOCPSolution(
            success=success,
            status=status,
            objective=float(sol["f"]),
            tf=tf,
            h=h,
            X_nodes=X_nodes,
            U_nodes=U_nodes,
            q_nodes=q_nodes,
            v_nodes=v_nodes,
            raw_solution=sol,
        )

    def print_summary(self) -> None:
        """
        Print compact OCP summary.
        """
        print("===== FAST DUAL-EE OCP =====")
        print("method              : explicit Hermite-Simpson")
        print("collision_enabled   : False")
        print(f"N                   : {self.N}")
        print(f"num_nodes           : {self.num_nodes}")
        print(f"nq                  : {self.nq}")
        print(f"nv                  : {self.nv}")
        print(f"nu                  : {self.nu}")
        print(f"nx                  : {self.nx}")
        print(f"solver_name         : {self.solver_name}")
        print(f"solver_options      : {self.solver_options}")

        if self.problem_data is not None:
            print(f"tf_init             : {self.problem_data['tf_init']}")
            print(f"tf_min              : {self.problem_data['tf_min']}")
            print(f"tf_max              : {self.problem_data['tf_max']}")
            print(f"time_weight         : {self.problem_data['time_weight']}")
            print(f"left_ee_target      : {self.problem_data['left_ee_target']}")
            print(f"right_ee_target     : {self.problem_data['right_ee_target']}")
            print(f"num_decision_vars   : {len(self.problem_data['w0'])}")
            print(f"num_constraints     : {len(self.problem_data['lbg'])}")

        if self.solution is not None:
            print(f"objective_opt       : {float(self.solution['f'])}")