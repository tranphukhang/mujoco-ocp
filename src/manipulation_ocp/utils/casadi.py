from __future__ import annotations

from typing import Any

import casadi as ca
import numpy as np


def sx_vector(name: str, size: int) -> ca.SX:
    """
    Create a CasADi SX column vector.

    Returns
    -------
    vec:
        CasADi SX with shape (size, 1)
    """
    if size <= 0:
        raise ValueError(f"size must be > 0, got {size}")

    return ca.SX.sym(name, size, 1)


def ensure_positive_int(value: int, *, name: str) -> int:
    """Validate a positive integer."""
    value = int(value)

    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")

    return value


def casadi_output_to_numpy(
    value: Any,
    *,
    expected_size: int | None = None,
    name: str = "value",
) -> np.ndarray:
    """
    Convert a CasADi output to a 1D numpy array.

    Useful for testing CasADi functions against numeric Pinocchio outputs.
    """
    arr = np.asarray(value, dtype=float).reshape(-1)

    if expected_size is not None and arr.size != expected_size:
        raise ValueError(
            f"{name} must have size {expected_size}, got {arr.size}"
        )

    return arr


def evaluate_vector_function(
    fn: ca.Function,
    *args: Any,
    expected_size: int | None = None,
    name: str = "output",
) -> np.ndarray:
    """
    Evaluate a single-output CasADi function and return a 1D numpy vector.
    """
    value = fn(*args)

    return casadi_output_to_numpy(
        value,
        expected_size=expected_size,
        name=name,
    )