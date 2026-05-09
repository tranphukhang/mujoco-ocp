from __future__ import annotations

from typing import Sequence

import numpy as np
import pinocchio as pin

from manipulation_ocp.utils.numerics import as_vector


def rnea(
    model: pin.Model,
    data: pin.Data,
    q: Sequence[float] | np.ndarray,
    v: Sequence[float] | np.ndarray,
    a: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """
    Recursive Newton-Euler Algorithm.

    Inverse dynamics:
        tau = RNEA(q, v, a)

    Parameters
    ----------
    q:
        Joint positions, shape (nq,)

    v:
        Joint velocities, shape (nv,)

    a:
        Joint accelerations, shape (nv,)

    Returns
    -------
    tau:
        Joint torques, shape (nv,)
    """
    q_vec = as_vector(q, size=model.nq, name="q")
    v_vec = as_vector(v, size=model.nv, name="v")
    a_vec = as_vector(a, size=model.nv, name="a")

    tau = pin.rnea(model, data, q_vec, v_vec, a_vec)

    return np.asarray(tau, dtype=float).reshape(model.nv)


def compute_rnea_trajectory(
    model: pin.Model,
    data: pin.Data,
    q_traj: np.ndarray,
    v_traj: np.ndarray,
    a_traj: np.ndarray,
) -> np.ndarray:
    """
    Compute RNEA torques for a trajectory.

    Expected shapes:
        q_traj = (num_nodes, nq)
        v_traj = (num_nodes, nv)
        a_traj = (num_nodes, nv)

    Returns
    -------
    u_traj:
        shape = (num_nodes, nv)
    """
    q_arr = np.asarray(q_traj, dtype=float)
    v_arr = np.asarray(v_traj, dtype=float)
    a_arr = np.asarray(a_traj, dtype=float)

    if q_arr.ndim != 2 or q_arr.shape[1] != model.nq:
        raise ValueError(
            f"q_traj must have shape (num_nodes, {model.nq}), "
            f"got {q_arr.shape}"
        )

    if v_arr.ndim != 2 or v_arr.shape[1] != model.nv:
        raise ValueError(
            f"v_traj must have shape (num_nodes, {model.nv}), "
            f"got {v_arr.shape}"
        )

    if a_arr.ndim != 2 or a_arr.shape[1] != model.nv:
        raise ValueError(
            f"a_traj must have shape (num_nodes, {model.nv}), "
            f"got {a_arr.shape}"
        )

    if not (q_arr.shape[0] == v_arr.shape[0] == a_arr.shape[0]):
        raise ValueError(
            "q_traj, v_traj, and a_traj must have the same number of nodes. "
            f"Got {q_arr.shape[0]}, {v_arr.shape[0]}, {a_arr.shape[0]}"
        )

    num_nodes = q_arr.shape[0]
    u_traj = np.zeros((num_nodes, model.nv), dtype=float)

    for k in range(num_nodes):
        u_traj[k] = rnea(
            model,
            data,
            q_arr[k],
            v_arr[k],
            a_arr[k],
        )

    return u_traj