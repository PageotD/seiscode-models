"""Model validation utilities."""

import numpy as np


def validate_model(
    v: np.ndarray,
    dx: float = 1.0,
    dz: float = 1.0,
    v_min: float = 0.0,
    v_max: float = 1.0e12,
) -> None:
    """Validate a velocity model array and its spatial parameters.

    Checks that the model is a 2D NumPy array, that its data type is strictly
    single (float32) or double (float64) precision, that all values are finite
    (no NaN or Inf), that velocities stay within physical bounds [v_min, v_max],
    and that grid intervals are positive.

    Parameters
    ----------
    v : ndarray of shape (nx, nz)
        2D velocity field array in m/s.
    dx : float, optional
        Grid spacing along horizontal x-axis in meters. Default is 1.0.
    dz : float, optional
        Grid spacing along vertical z-axis in meters. Default is 1.0.
    v_min : float, optional
        Minimum allowable physical velocity in m/s. Default is 0.0.
    v_max : float, optional
        Maximum allowable physical velocity in m/s. Default is 1.0e12.

    Raises
    ------
    ValueError
        If `v` is not a 2D NumPy array, if its dtype is not float32 or float64,
        if non-finite values (NaN/Inf) exist, if velocities fall outside [v_min, v_max],
        or if `dx`/`dz` are not positive.

    Examples
    --------
    >>> import numpy as np
    >>> import seiscode_models as sm
    >>> arr = np.ones((10, 10), dtype=np.float32) * 2000.0
    >>> sm.validate_model(arr, dx=5.0, dz=5.0)  # Passes silently
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
