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
    r"""Generate a linear velocity gradient with depth.

    Computes a 2D velocity background that linearly increases with depth
    from surface velocity :math:`v_0` to base velocity :math:`v_1`:

    .. math::

        v(x, z) = v_0 + (v_1 - v_0) \cdot \frac{z}{z_{\max}}

    Parameters
    ----------
    nx : int
        Number of grid cells along x (horizontal direction).
    nz : int
        Number of grid cells along z (vertical direction, positive downward).
    dx : float, optional
        Grid spacing along x in meters. Default is 1.0.
    dz : float, optional
        Grid spacing along z in meters. Default is 1.0.
    v0 : float, optional
        Velocity at the surface (:math:`z=0`), in m/s. Default is 1500.0.
    v1 : float, optional
        Velocity at the bottom of the model (:math:`z=(nz-1)dz`), in m/s. Default is 4500.0.
    dtype : str or np.dtype, optional
        NumPy floating-point data type ('f32', 'f64', np.float32, np.float64). Default is np.float32.

    Returns
    -------
    ndarray of shape (nx, nz)
        2D velocity field in m/s with the requested dtype.

    Raises
    ------
    ValueError
        If `dx` or `dz` are not strictly positive.

    Examples
    --------
    >>> import seiscode_models as sm
    >>> v = sm.gradient_background(100, 50, dx=5.0, dz=5.0, v0=1500.0, v1=3500.0)
    >>> v.shape
    (100, 50)
    >>> v[0, 0], v[0, -1]
    (1500.0, 3500.0)
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
    r"""Add a smooth 2D Gaussian velocity perturbation to an existing model.

    Superimposes a localized Gaussian anomaly centered at :math:`(c_x, c_z)`
    with spatial widths :math:`\sigma_x, \sigma_z`:

    .. math::

        \Delta v(x, z) = dv \cdot \exp\left( - \left[ \frac{(x - c_x)^2}{2\sigma_x^2}
        + \frac{(z - c_z)^2}{2\sigma_z^2} \right] \right)

    Velocities are subsequently clipped to the physical interval :math:`[v_{\min}, v_{\max}]`.

    Parameters
    ----------
    v : ndarray of shape (nx, nz)
        Input velocity field to perturb (not modified in place).
    cx : float
        Center coordinate along horizontal axis in meters (or cells if dx=1).
    cz : float
        Center coordinate along vertical axis in meters (or cells if dz=1).
    sigma_x : float
        Gaussian standard deviation (half-width) along x in meters (or cells).
    sigma_z : float
        Gaussian standard deviation (half-width) along z in meters (or cells).
    dv : float
        Peak velocity perturbation amplitude at the center in m/s (can be positive or negative).
    dx : float, optional
        Horizontal grid spacing in meters. Default is 1.0.
    dz : float, optional
        Vertical grid spacing in meters. Default is 1.0.
    vmin : float, optional
        Lower physical velocity clamp in m/s. Default is 200.0.
    vmax : float, optional
        Upper physical velocity clamp in m/s. Default is 8000.0.

    Returns
    -------
    ndarray of shape (nx, nz)
        A new velocity array containing the Gaussian perturbation, maintaining original dtype.

    Raises
    ------
    ValueError
        If `dx` or `dz` are not strictly positive.

    Examples
    --------
    >>> import seiscode_models as sm
    >>> bg = sm.gradient_background(80, 40, dx=10.0, dz=10.0)
    >>> anom = sm.add_gaussian_anomaly(bg, cx=400.0, cz=200.0, sigma_x=50.0, sigma_z=30.0, dv=500.0, dx=10.0, dz=10.0)
    >>> anom.shape
    (80, 40)
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
    """Shift velocity layers vertically across a planar vertical fault line.

    Simulates tectonic fault displacement by shifting the left or right structural
    block vertically along the fault position `x_fault`.

    Parameters
    ----------
    v : ndarray of shape (nx, nz)
        Stratified 2D velocity field to offset (not modified in place).
    x_fault : float, optional
        Horizontal position of the vertical fault in meters.
    throw : float, optional
        Vertical fault throw (displacement) in meters. Positive throws displace the downthrown block.
    dx : float, optional
        Horizontal grid spacing in meters. Default is 1.0.
    dz : float, optional
        Vertical grid spacing in meters. Default is 1.0.
    ix_fault : int, optional
        Deprecated: column index of the fault. Prefer using `x_fault` in physical units (meters).

    Returns
    -------
    ndarray of shape (nx, nz)
        Shifted velocity model preserving the original shape and dtype.

    Raises
    ------
    ValueError
        If neither `x_fault` nor `ix_fault` is provided, if `throw` is None, or if `dx`/`dz` <= 0.

    Examples
    --------
    >>> import seiscode_models as sm
    >>> bg = sm.gradient_background(60, 40, dx=5.0, dz=5.0)
    >>> faulted = sm.add_fault_offset(bg, x_fault=150.0, throw=25.0, dx=5.0, dz=5.0)
    >>> faulted.shape
    (60, 40)
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
    r"""Insert a sharp elliptical velocity body into an existing model.

    Replaces all velocity values inside the ellipse defined by:

    .. math::

        \left(\frac{x - c_x}{r_x}\right)^2 + \left(\frac{z - c_z}{r_z}\right)^2 < 1

    with the constant inclusion velocity `cv`.

    Parameters
    ----------
    v : ndarray of shape (nx, nz)
        Input velocity field to modify (not modified in place).
    cx : float
        Horizontal center position of the ellipse in meters.
    cz : float
        Vertical center position of the ellipse in meters.
    radius_x : float
        Horizontal semi-axis radius :math:`r_x` in meters. Must be positive.
    radius_z : float
        Vertical semi-axis radius :math:`r_z` in meters. Must be positive.
    cv : float
        Constant velocity inside the inclusion body in m/s.
    dx : float, optional
        Horizontal grid spacing in meters. Default is 1.0.
    dz : float, optional
        Vertical grid spacing in meters. Default is 1.0.

    Returns
    -------
    ndarray of shape (nx, nz)
        A copy of `v` containing the embedded elliptical inclusion.

    Raises
    ------
    ValueError
        If `radius_x` or `radius_z` are non-positive, or if `dx`/`dz` <= 0.

    Examples
    --------
    >>> import seiscode_models as sm
    >>> bg = sm.gradient_background(50, 50, dx=10.0, dz=10.0)
    >>> incl = sm.add_elliptical_inclusion(bg, cx=250.0, cz=250.0, radius_x=100.0, radius_z=50.0, cv=5000.0, dx=10.0, dz=10.0)
    >>> incl[25, 25]
    5000.0
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
