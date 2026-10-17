"""Primitive building blocks for subsurface velocity models with dtype and physical coordinate support."""

from typing import Optional, Union

import numpy as np

from ..utils.dtypes import parse_dtype


def gradient_background(
    nx: int,
    nz: int,
    dx: float = 1.0,
    dz: float = 1.0,
    v0: float = 1500.0,
    v1: float = 4500.0,
    dtype: Union[str, np.dtype] = np.float32,
) -> np.ndarray:
    """Generate a linear velocity gradient with depth.

    Parameters
    ----------
    nx : int
        Number of cells along x (horizontal direction).
    nz : int
        Number of cells along z (vertical direction, positive downward).
    dx : float, optional
        Grid spacing along x, in meters. Default is 1.0.
    dz : float, optional
        Grid spacing along z, in meters. Default is 1.0.
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
    if dx <= 0 or dz <= 0:
        raise ValueError("Grid spacing dx and dz must be positive.")
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
    dx: float = 1.0,
    dz: float = 1.0,
    vmin: float = 200.0,
    vmax: float = 8000.0,
) -> np.ndarray:
    """Add a Gaussian velocity anomaly to an existing model.

    Parameters
    ----------
    v : ndarray (nx, nz)
        Velocity field to modify (not modified in place).
    cx, cz : float
        Center position along x and z in physical coordinates (meters) if dx/dz are provided,
        or in grid cells if dx=1.0 and dz=1.0.
    sigma_x, sigma_z : float
        Gaussian standard deviation along x and z (in meters if dx/dz provided, else cells).
    dv : float
        Amplitude of the anomaly at the center, in m/s.
    dx, dz : float, optional
        Grid spacing along x and z in meters. Default is 1.0.
    vmin, vmax : float, optional
        Clipping velocity bounds, in m/s.

    Returns
    -------
    ndarray (nx, nz)
        Copy of v with the anomaly added, preserving dtype and clipped.
    """
    if dx <= 0 or dz <= 0:
        raise ValueError("Grid spacing dx and dz must be positive.")
    nx, nz = v.shape
    dt = v.dtype
    x = (np.arange(nx, dtype=dt) * dx)[:, None]
    z = (np.arange(nz, dtype=dt) * dz)[None, :]
    g = dv * np.exp(
        -((x - cx) ** 2 / (2 * sigma_x**2) + (z - cz) ** 2 / (2 * sigma_z**2))
    )
    return np.clip(v + g, vmin, vmax).astype(dt)


def add_fault_offset(
    v: np.ndarray,
    x_fault: Optional[float] = None,
    throw: Optional[float] = None,
    dx: float = 1.0,
    dz: float = 1.0,
    ix_fault: Optional[int] = None,
) -> np.ndarray:
    """Shift layers on either side of a vertical fault.

    Parameters
    ----------
    v : ndarray (nx, nz)
        Stratified velocity field to shift.
    x_fault : float, optional
        Horizontal position of the fault in physical units (meters).
    throw : float, optional
        Fault throw along vertical axis in physical units (meters if dz provided, else cells).
    dx, dz : float, optional
        Grid spacing along x and z in meters. Default is 1.0.
    ix_fault : int, optional
        Deprecated: column index of the fault. Use x_fault instead.

    Returns
    -------
    ndarray (nx, nz)
        Shifted field preserving original dtype.
    """
    if dx <= 0 or dz <= 0:
        raise ValueError("Grid spacing dx and dz must be positive.")

    if x_fault is None:
        if ix_fault is not None:
            col_fault = int(ix_fault)
        else:
            raise ValueError("Either 'x_fault' or 'ix_fault' must be provided.")
    else:
        col_fault = int(round(x_fault / dx))

    if throw is None:
        raise ValueError("'throw' must be provided.")

    shift_cells = int(round(throw / dz))

    out = v.copy()
    left, right = slice(None, col_fault), slice(col_fault, None)
    if shift_cells >= 0:
        if shift_cells < out.shape[1]:
            out[left, shift_cells:] = v[left, : out.shape[1] - shift_cells]
        else:
            out[left, :] = v[left, 0]
    else:
        if -shift_cells < out.shape[1]:
            out[right, :shift_cells] = v[right, -shift_cells:]
        else:
            out[right, :] = v[right, -1]
    return out


def add_elliptical_inclusion(
    v: np.ndarray,
    cx: float,
    cz: float,
    radius_x: float,
    radius_z: float,
    cv: float,
    dx: float = 1.0,
    dz: float = 1.0,
) -> np.ndarray:
    """Add an elliptical velocity anomaly and return a new model.

    Parameters
    ----------
    v : ndarray (nx, nz)
        Velocity field to modify (not modified in place).
    cx, cz : float
        Center position along x and z in physical coordinates (meters).
    radius_x, radius_z : float
        Semi-axes of the ellipse in meters.
    cv : float
        Velocity inside the ellipse, in m/s.
    dx, dz : float, optional
        Grid spacing along x and z in meters. Default is 1.0.

    Returns
    -------
    ndarray (nx, nz)
        Copy of v containing the elliptical anomaly.
    """
    if radius_x <= 0 or radius_z <= 0:
        raise ValueError("Ellipse radius_x and radius_z must be positive.")
    if dx <= 0 or dz <= 0:
        raise ValueError("Grid spacing dx and dz must be positive.")

    out = v.copy()
    nx, nz = v.shape
    dt = v.dtype

    x = (np.arange(nx, dtype=dt) * dx)[:, None]
    z = (np.arange(nz, dtype=dt) * dz)[None, :]

    mask = ((x - cx) / radius_x) ** 2 + ((z - cz) / radius_z) ** 2 < 1.0
    out[mask] = cv

    return out.astype(v.dtype, copy=False)
