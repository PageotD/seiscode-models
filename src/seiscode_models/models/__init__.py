"""Primitive building blocks for subsurface velocity models with dtype support."""

import numpy as np

from ..utils.dtypes import parse_dtype


def gradient_background(
    nx: int, nz: int, v0: float = 1500.0, v1: float = 4500.0, dtype=np.float32
) -> np.ndarray:
    """Generate a linear velocity gradient with depth.

    Parameters
    ----------
    nx : int
        Number of cells along x (horizontal direction).
    nz : int
        Number of cells along z (vertical direction, positive downward).
    v0 : float, optional
        Velocity at the surface, in m/s. Default is 1500.0.
    v1 : float, optional
        Velocity at the bottom of the model, in m/s. Default is 4500.0.
    dtype : str or np.dtype, optional
        Data type precision ('f32', 'f64', etc.). Default is np.float32.

    Returns
    -------
    ndarray (nx, nz)
        Velocity field in m/s with the requested dtype.
    """
    dt = parse_dtype(dtype)
    z = np.linspace(0.0, 1.0, nz, dtype=dt)
    v_col = v0 + (v1 - v0) * z
    return np.broadcast_to(v_col, (nx, nz)).copy()


def add_gaussian_anomaly(
    v: np.ndarray,
    cx: float,
    cz: float,
    sigma_x: float,
    sigma_z: float,
    dv: float,
    vmin: float = 200.0,
    vmax: float = 8000.0,
) -> np.ndarray:
    """Add a Gaussian velocity anomaly to an existing model.

    Parameters
    ----------
    v : ndarray (nx, nz)
        Velocity field to modify (not modified in place).
    cx, cz : float
        Center position along x and z, in cells.
    sigma_x, sigma_z : float
        Gaussian standard deviation along x and z, in cells.
    dv : float
        Amplitude of the anomaly at the center, in m/s.
    vmin, vmax : float, optional
        Clipping velocity bounds, in m/s.

    Returns
    -------
    ndarray (nx, nz)
        Copy of v with the anomaly added, preserving dtype and clipped.
    """
    nx, nz = v.shape
    dt = v.dtype
    ii = np.arange(nx, dtype=dt)[:, None]
    jj = np.arange(nz, dtype=dt)[None, :]
    g = dv * np.exp(
        -((ii - cx) ** 2 / (2 * sigma_x**2) + (jj - cz) ** 2 / (2 * sigma_z**2))
    )
    return np.clip(v + g, vmin, vmax).astype(dt)


def add_fault_offset(v: np.ndarray, ix_fault: int, throw: int) -> np.ndarray:
    """Shift layers on either side of a vertical fault.

    Parameters
    ----------
    v : ndarray (nx, nz)
        Stratified velocity field to shift.
    ix_fault : int
        Column index of the fault.
    throw : int
        Fault throw in cells along z.

    Returns
    -------
    ndarray (nx, nz)
        Shifted field preserving original dtype.
    """
    out = v.copy()
    left, right = slice(None, ix_fault), slice(ix_fault, None)
    if throw >= 0:
        out[left, throw:] = v[left, : out.shape[1] - throw]
    else:
        out[right, :throw] = v[right, -throw:]
    return out
