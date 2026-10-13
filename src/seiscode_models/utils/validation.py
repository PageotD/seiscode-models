"""Model validation utilities."""

import numpy as np


def validate_model(
    v: np.ndarray,
    dx: float = 1.0,
    dz: float = 1.0,
    v_min: float = 0.0,
    v_max: float = 1.0e12,
) -> None:
    """Validate a velocity model array and its parameters.

    Parameters
    ----------
    v : ndarray
        Velocity model array (nx, nz).
    dx, dz : float
        Grid spacing in meters.
    v_min, v_max : float
        Acceptable velocity limits in m/s.

    Raises
    ------
    ValueError
        If the model structure, dtype, or bounds are invalid.
    """
    if not isinstance(v, np.ndarray):
        raise ValueError("Model must be a NumPy array.")
    if v.ndim != 2:
        raise ValueError(f"Model must be 2D, got {v.ndim}D.")
    if v.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(f"Model dtype must be float32 or float64, got {v.dtype}.")
    if not np.all(np.isfinite(v)):
        raise ValueError("Model contains non-finite values (NaN or Inf).")
    if np.any(v < v_min) or np.any(v > v_max):
        raise ValueError(f"Velocities out of range [{v_min}, {v_max}] m/s.")
    if dx <= 0 or dz <= 0:
        raise ValueError("Grid spacing dx and dz must be positive.")
