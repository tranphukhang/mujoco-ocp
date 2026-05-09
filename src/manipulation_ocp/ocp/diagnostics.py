from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from manipulation_ocp.utils.numerics import as_vector


@dataclass(frozen=True)
class BoundDiagnostics:
    """
    Bound checking result.

    For variables y with lower <= y <= upper:

        lower_margin = y - lower
        upper_margin = upper - y

    If all constraints are satisfied:
        min_lower_margin >= 0
        min_upper_margin >= 0
        max_violation = 0
    """

    name: str
    min_lower_margin: float
    min_upper_margin: float
    max_violation: float


@dataclass(frozen=True)
class ImplicitMidpointResidualDiagnostics:
    """
    Diagnostics for implicit midpoint dynamics residual.

    For each interval k:

        x_mid    = 0.5 * (X_k + X_{k+1})
        u_mid    = 0.5 * (U_k + U_{k+1})
        xdot_mid = (X_{k+1} - X_k) / h

        residual_k = F_impl(x_mid, xdot_mid, u_mid)

    where:

        F_impl =
        [
            qdot_mid - v_mid
            RNEA(q_mid, v_mid, vdot_mid) - u_mid
        ]

    residual:
        Full implicit dynamics residual, shape (N, nx).

    kinematic:
        qdot_mid - v_mid, shape (N, nq).

    inverse_dynamics:
        RNEA(...) - u_mid, shape (N, nv).
    """

    max_residual_l2: float
    mean_residual_l2: float
    max_residual_inf: float

    max_kinematic_l2: float
    mean_kinematic_l2: float
    max_kinematic_inf: float

    max_inverse_dynamics_l2: float
    mean_inverse_dynamics_l2: float
    max_inverse_dynamics_inf: float


@dataclass(frozen=True)
class OCPDiagnosticsReport:
    """
    Diagnostics report for one solved OCP stage.
    """

    success: bool
    status: str
    objective: float

    tf: float
    h: float
    num_nodes: int
    method: str

    initial_state_l2_error: float
    initial_state_inf_error: float

    terminal_left_position: np.ndarray
    terminal_right_position: np.ndarray

    terminal_left_target: np.ndarray
    terminal_right_target: np.ndarray

    terminal_left_error: np.ndarray
    terminal_right_error: np.ndarray

    terminal_left_error_norm: float
    terminal_right_error_norm: float
    terminal_pair_error_norm: float

    implicit_midpoint_residual: ImplicitMidpointResidualDiagnostics

    q_bounds: BoundDiagnostics
    v_bounds: BoundDiagnostics
    u_bounds: BoundDiagnostics

    max_abs_q: float
    max_abs_v: float
    max_abs_u: float

    def to_dict(self) -> dict[str, Any]:
        """
        Convert report to a plain dict.

        Numpy arrays are kept as arrays.
        """
        return asdict(self)


def _bound_diagnostics(
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    name: str,
) -> BoundDiagnostics:
    """
    Check bounds over a trajectory.

    values:
        shape = (num_nodes, dim)

    lower, upper:
        shape = (dim,)
    """
    values = np.asarray(values, dtype=float)
    lower = np.asarray(lower, dtype=float).reshape(-1)
    upper = np.asarray(upper, dtype=float).reshape(-1)

    if values.ndim != 2:
        raise ValueError(f"{name} values must be 2D, got shape {values.shape}")

    if values.shape[1] != lower.size:
        raise ValueError(
            f"{name} dim mismatch: values dim={values.shape[1]}, "
            f"lower dim={lower.size}"
        )

    if upper.size != lower.size:
        raise ValueError(
            f"{name} bound mismatch: lower dim={lower.size}, "
            f"upper dim={upper.size}"
        )

    lower_margin = values - lower[None, :]
    upper_margin = upper[None, :] - values

    lower_violation = np.maximum(lower[None, :] - values, 0.0)
    upper_violation = np.maximum(values - upper[None, :], 0.0)

    max_violation = max(
        float(np.max(lower_violation)),
        float(np.max(upper_violation)),
    )

    return BoundDiagnostics(
        name=name,
        min_lower_margin=float(np.min(lower_margin)),
        min_upper_margin=float(np.min(upper_margin)),
        max_violation=max_violation,
    )


def compute_implicit_midpoint_residuals(
    *,
    dynamics: Any,
    X_nodes: np.ndarray,
    U_nodes: np.ndarray,
    h: float,
) -> np.ndarray:
    """
    Compute numeric implicit midpoint residuals for a solved trajectory.

    For each interval k:

        x_mid    = 0.5 * (X_k + X_{k+1})
        u_mid    = 0.5 * (U_k + U_{k+1})
        xdot_mid = (X_{k+1} - X_k) / h

        residual_k = implicit_dynamics(x_mid, xdot_mid, u_mid)

    Returns
    -------
    residuals:
        shape = (N, nx)
    """
    X = np.asarray(X_nodes, dtype=float)
    U = np.asarray(U_nodes, dtype=float)

    if X.ndim != 2:
        raise ValueError(f"X_nodes must be 2D, got shape {X.shape}")

    if U.ndim != 2:
        raise ValueError(f"U_nodes must be 2D, got shape {U.shape}")

    if X.shape[0] != U.shape[0]:
        raise ValueError(
            f"X_nodes and U_nodes must have same number of nodes. "
            f"Got X={X.shape}, U={U.shape}"
        )

    if h <= 0.0:
        raise ValueError(f"h must be > 0, got {h}")

    num_nodes, nx = X.shape
    num_intervals = num_nodes - 1

    residuals = np.zeros((num_intervals, nx), dtype=float)

    for k in range(num_intervals):
        xk = X[k]
        xkp1 = X[k + 1]

        uk = U[k]
        ukp1 = U[k + 1]

        x_mid = 0.5 * (xk + xkp1)
        u_mid = 0.5 * (uk + ukp1)

        xdot_mid = (xkp1 - xk) / h

        residuals[k] = dynamics.evaluate_implicit_dynamics(
            x_mid,
            xdot_mid,
            u_mid,
        )

    return residuals


def _implicit_midpoint_residual_diagnostics(
    *,
    residuals: np.ndarray,
    nq: int,
    nv: int,
) -> ImplicitMidpointResidualDiagnostics:
    """
    Build compact diagnostics from implicit midpoint residuals.

    residuals:
        shape = (N, nq + nv)

    first nq entries:
        qdot_mid - v_mid

    next nv entries:
        RNEA(q_mid, v_mid, vdot_mid) - u_mid
    """
    residuals = np.asarray(residuals, dtype=float)

    if residuals.ndim != 2:
        raise ValueError(f"residuals must be 2D, got shape {residuals.shape}")

    if residuals.shape[1] != nq + nv:
        raise ValueError(
            f"residual dim mismatch: expected {nq + nv}, "
            f"got {residuals.shape[1]}"
        )

    kin_res = residuals[:, :nq]
    dyn_res = residuals[:, nq : nq + nv]

    residual_l2 = np.linalg.norm(residuals, axis=1)
    kin_l2 = np.linalg.norm(kin_res, axis=1)
    dyn_l2 = np.linalg.norm(dyn_res, axis=1)

    return ImplicitMidpointResidualDiagnostics(
        max_residual_l2=float(np.max(residual_l2)),
        mean_residual_l2=float(np.mean(residual_l2)),
        max_residual_inf=float(np.max(np.abs(residuals))),

        max_kinematic_l2=float(np.max(kin_l2)),
        mean_kinematic_l2=float(np.mean(kin_l2)),
        max_kinematic_inf=float(np.max(np.abs(kin_res))),

        max_inverse_dynamics_l2=float(np.max(dyn_l2)),
        mean_inverse_dynamics_l2=float(np.mean(dyn_l2)),
        max_inverse_dynamics_inf=float(np.max(np.abs(dyn_res))),
    )


def evaluate_ocp_solution(
    *,
    ocp: Any,
    solution: Any,
    x0_value: Sequence[float] | np.ndarray | None = None,
) -> OCPDiagnosticsReport:
    """
    Evaluate one solved OCP stage.

    This diagnostics module assumes the OCP uses implicit midpoint dynamics.

    Parameters
    ----------
    ocp:
        FastDualEEOCP object.

    solution:
        FastDualEEOCPSolution object.

    x0_value:
        Optional initial state used for the stage.

        If None, this function tries to use:
            ocp.last_stage_data["x0_value"]

    Returns
    -------
    report:
        OCPDiagnosticsReport
    """
    if ocp.problem_data is None:
        raise RuntimeError("ocp.problem_data is None. Build the OCP first.")

    pd = ocp.problem_data

    method = str(pd.get("method", "unknown"))

    if method != "implicit_midpoint":
        raise ValueError(
            "This diagnostics file expects method='implicit_midpoint'. "
            f"Got method={method!r}."
        )

    X_nodes = np.asarray(solution.X_nodes, dtype=float)
    U_nodes = np.asarray(solution.U_nodes, dtype=float)

    q_nodes = np.asarray(solution.q_nodes, dtype=float)
    v_nodes = np.asarray(solution.v_nodes, dtype=float)

    if x0_value is None:
        if getattr(ocp, "last_stage_data", None) is not None:
            x0_value = ocp.last_stage_data.get("x0_value")

    if x0_value is None:
        initial_state_l2_error = float("nan")
        initial_state_inf_error = float("nan")
    else:
        x0_vec = as_vector(x0_value, size=ocp.nx, name="x0_value")
        x0_error = X_nodes[0] - x0_vec
        initial_state_l2_error = float(np.linalg.norm(x0_error))
        initial_state_inf_error = float(np.max(np.abs(x0_error)))

    q_terminal = q_nodes[-1]

    terminal_left_position = ocp.kinematics.evaluate_left_position(q_terminal)
    terminal_right_position = ocp.kinematics.evaluate_right_position(q_terminal)

    terminal_left_target = np.asarray(solution.left_ee_target, dtype=float)
    terminal_right_target = np.asarray(solution.right_ee_target, dtype=float)

    terminal_left_error = terminal_left_position - terminal_left_target
    terminal_right_error = terminal_right_position - terminal_right_target

    terminal_left_error_norm = float(np.linalg.norm(terminal_left_error))
    terminal_right_error_norm = float(np.linalg.norm(terminal_right_error))

    terminal_pair_error_norm = float(
        np.linalg.norm(
            np.concatenate(
                [
                    terminal_left_error,
                    terminal_right_error,
                ]
            )
        )
    )

    residuals = compute_implicit_midpoint_residuals(
        dynamics=ocp.dynamics,
        X_nodes=X_nodes,
        U_nodes=U_nodes,
        h=solution.h,
    )

    residual_report = _implicit_midpoint_residual_diagnostics(
        residuals=residuals,
        nq=ocp.nq,
        nv=ocp.nv,
    )

    x_min = np.asarray(pd["x_min"], dtype=float)
    x_max = np.asarray(pd["x_max"], dtype=float)

    q_min = x_min[: ocp.nq]
    q_max = x_max[: ocp.nq]

    v_min = x_min[ocp.nq : ocp.nq + ocp.nv]
    v_max = x_max[ocp.nq : ocp.nq + ocp.nv]

    u_min = np.asarray(pd["u_min"], dtype=float)
    u_max = np.asarray(pd["u_max"], dtype=float)

    q_bounds = _bound_diagnostics(
        q_nodes,
        q_min,
        q_max,
        name="q",
    )

    v_bounds = _bound_diagnostics(
        v_nodes,
        v_min,
        v_max,
        name="v",
    )

    u_bounds = _bound_diagnostics(
        U_nodes,
        u_min,
        u_max,
        name="u",
    )

    return OCPDiagnosticsReport(
        success=bool(solution.success),
        status=str(solution.status),
        objective=float(solution.objective),
        tf=float(solution.tf),
        h=float(solution.h),
        num_nodes=int(X_nodes.shape[0]),
        method=method,
        initial_state_l2_error=initial_state_l2_error,
        initial_state_inf_error=initial_state_inf_error,
        terminal_left_position=terminal_left_position,
        terminal_right_position=terminal_right_position,
        terminal_left_target=terminal_left_target,
        terminal_right_target=terminal_right_target,
        terminal_left_error=terminal_left_error,
        terminal_right_error=terminal_right_error,
        terminal_left_error_norm=terminal_left_error_norm,
        terminal_right_error_norm=terminal_right_error_norm,
        terminal_pair_error_norm=terminal_pair_error_norm,
        implicit_midpoint_residual=residual_report,
        q_bounds=q_bounds,
        v_bounds=v_bounds,
        u_bounds=u_bounds,
        max_abs_q=float(np.max(np.abs(q_nodes))),
        max_abs_v=float(np.max(np.abs(v_nodes))),
        max_abs_u=float(np.max(np.abs(U_nodes))),
    )


def print_ocp_diagnostics(report: OCPDiagnosticsReport) -> None:
    """
    Print a compact human-readable diagnostics report.
    """
    print("===== OCP SOLUTION DIAGNOSTICS =====")
    print(f"success                  : {report.success}")
    print(f"status                   : {report.status}")
    print(f"objective                : {report.objective:.6e}")
    print(f"method                   : {report.method}")
    print(f"Tf                       : {report.tf:.6f}")
    print(f"h                        : {report.h:.6f}")
    print(f"num_nodes                : {report.num_nodes}")

    print("")
    print("[initial state]")
    print(f"L2 error                 : {report.initial_state_l2_error:.6e}")
    print(f"inf error                : {report.initial_state_inf_error:.6e}")

    print("")
    print("[terminal EE]")
    print(f"left position            : {report.terminal_left_position}")
    print(f"left target              : {report.terminal_left_target}")
    print(f"left error               : {report.terminal_left_error}")
    print(f"left error norm          : {report.terminal_left_error_norm:.6e}")

    print(f"right position           : {report.terminal_right_position}")
    print(f"right target             : {report.terminal_right_target}")
    print(f"right error              : {report.terminal_right_error}")
    print(f"right error norm         : {report.terminal_right_error_norm:.6e}")

    print(f"pair error norm          : {report.terminal_pair_error_norm:.6e}")

    residual = report.implicit_midpoint_residual

    print("")
    print("[implicit midpoint residual]")
    print(f"max residual L2          : {residual.max_residual_l2:.6e}")
    print(f"mean residual L2         : {residual.mean_residual_l2:.6e}")
    print(f"max residual inf         : {residual.max_residual_inf:.6e}")

    print("")
    print("[kinematic residual: qdot - v]")
    print(f"max kinematic L2         : {residual.max_kinematic_l2:.6e}")
    print(f"mean kinematic L2        : {residual.mean_kinematic_l2:.6e}")
    print(f"max kinematic inf        : {residual.max_kinematic_inf:.6e}")

    print("")
    print("[inverse dynamics residual: RNEA - u]")
    print(f"max inverse dyn L2       : {residual.max_inverse_dynamics_l2:.6e}")
    print(f"mean inverse dyn L2      : {residual.mean_inverse_dynamics_l2:.6e}")
    print(f"max inverse dyn inf      : {residual.max_inverse_dynamics_inf:.6e}")

    print("")
    print("[bounds]")
    for bound in [report.q_bounds, report.v_bounds, report.u_bounds]:
        print(f"{bound.name} min lower margin : {bound.min_lower_margin:.6e}")
        print(f"{bound.name} min upper margin : {bound.min_upper_margin:.6e}")
        print(f"{bound.name} max violation    : {bound.max_violation:.6e}")

    print("")
    print("[max absolute values]")
    print(f"max |q|                  : {report.max_abs_q:.6e}")
    print(f"max |v|                  : {report.max_abs_v:.6e}")
    print(f"max |u|                  : {report.max_abs_u:.6e}")