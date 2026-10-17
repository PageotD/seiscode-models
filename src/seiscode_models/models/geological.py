"""Geological velocity models supporting f32/f64 precision and physical coordinates."""

from typing import Optional, Sequence, Union

import numpy as np

from ..utils.dtypes import parse_dtype
from .primitives import add_fault_offset, add_gaussian_anomaly, gradient_background


def dipping_layers(
    nx: int = 201,
    nz: int = 51,
    dx: float = 1.0,
    dz: float = 1.0,
    v: Sequence[float] = (1800.0, 2400.0, 3200.0, 4200.0),
    dip_deg: float = 8.0,
    thickness: float = 10.0,
    dtype: Union[str, np.dtype] = np.float32,
) -> np.ndarray:
    """Generate dipping planar layers with the same dip angle.

    Parameters
    ----------
    nx : int, optional
        Number of grid cells along x. Default is 201.
    nz : int, optional
        Number of grid cells along z. Default is 51.
    dx : float, optional
        Grid spacing along x in meters. Default is 1.0.
    dz : float, optional
        Grid spacing along z in meters. Default is 1.0.
    v : sequence of float, optional
        Velocities of layers from top to bottom in m/s.
    dip_deg : float, optional
        Dip angle in degrees. Default is 8.0.
    thickness : float, optional
        Thickness of each dipping layer in meters. Default is 10.0.
    dtype : str or np.dtype, optional
        Output data type. Default is np.float32.

    Returns
    -------
    ndarray (nx, nz)
        Dipping layers velocity model.
    """
    if dx <= 0 or dz <= 0:
        raise ValueError("Grid spacing dx and dz must be positive.")
    dt = parse_dtype(dtype)
    v_arr = np.asarray(v, dtype=dt)
    x = np.arange(nx, dtype=dt) * dx
    dip = np.tan(np.deg2rad(dip_deg))
    depths = np.cumsum(np.arange(1, v_arr.size) * thickness).astype(dt)
    k_int = np.clip(
        ((depths[:, None] + (x - x.mean()) * dip) / dz).astype(int), 0, nz - 1
    )
    v_out = np.full((nx, nz), v_arr[-1], dtype=dt)
    for k in range(v_arr.size - 2, -1, -1):
        mask = np.arange(nz)[None, :] < k_int[k][:, None]  # (nx, nz)
        v_out = np.where(mask, v_arr[k], v_out).astype(dt)
    return v_out


def sinusoidal_interface(
    nx: int = 201,
    nz: int = 51,
    dx: float = 1.0,
    dz: float = 1.0,
    period: float = 100.0,
    amplitude: float = 8.0,
    z_mean: Optional[float] = None,
    v_top: float = 2000.0,
    v_bottom: float = 3500.0,
    dtype: Union[str, np.dtype] = np.float32,
) -> np.ndarray:
    """Generate two layers separated by a sinusoidal interface.

    Parameters
    ----------
    nx : int, optional
        Number of grid cells along x. Default is 201.
    nz : int, optional
        Number of grid cells along z. Default is 51.
    dx : float, optional
        Grid spacing along x in meters. Default is 1.0.
    dz : float, optional
        Grid spacing along z in meters. Default is 1.0.
    period : float, optional
        Spatial wavelength of the sinusoid in meters. Default is 100.0.
    amplitude : float, optional
        Amplitude of the interface oscillation in meters. Default is 8.0.
    z_mean : float, optional
        Mean depth of the interface in meters. Default is 0.45 * (nz * dz).
    v_top : float, optional
        Velocity of upper layer in m/s. Default is 2000.0.
    v_bottom : float, optional
        Velocity of lower layer in m/s. Default is 3500.0.
    dtype : str or np.dtype, optional
        Output data type. Default is np.float32.

    Returns
    -------
    ndarray (nx, nz)
        Sinusoidal interface velocity model.
    """
    if dx <= 0 or dz <= 0:
        raise ValueError("Grid spacing dx and dz must be positive.")
    dt = parse_dtype(dtype)
    x = np.arange(nx, dtype=dt) * dx
    if z_mean is None:
        mean_depth = (nz * dz) * 0.45
    else:
        mean_depth = float(z_mean)

    z_interface = mean_depth + amplitude * np.sin(2 * np.pi * x / period)
    k_int = np.clip((z_interface / dz).astype(int), 0, nz)
    v = np.full((nx, nz), v_bottom, dtype=dt)
    v[k_int[:, None] > np.arange(nz)[None, :]] = v_top
    return v


def faulted_layers(
    nx: int = 201,
    nz: int = 51,
    dx: float = 1.0,
    dz: float = 1.0,
    x_fault: Optional[float] = None,
    throw: Optional[float] = None,
    v: Sequence[float] = (1800.0, 2400.0, 3200.0, 4200.0),
    dtype: Union[str, np.dtype] = np.float32,
    ix_fault: Optional[int] = None,
) -> np.ndarray:
    """Generate stratigraphic layers with a normal fault.

    Parameters
    ----------
    nx : int, optional
        Number of grid cells along x. Default is 201.
    nz : int, optional
        Number of grid cells along z. Default is 51.
    dx : float, optional
        Grid spacing along x in meters. Default is 1.0.
    dz : float, optional
        Grid spacing along z in meters. Default is 1.0.
    x_fault : float, optional
        Horizontal position of the fault in meters. If None and ix_fault is None, defaults to 0.6 * (nx * dx).
    throw : float, optional
        Vertical fault throw in meters. Default is 8.0 * dz.
    v : sequence of float, optional
        Velocities of layers from top to bottom in m/s.
    dtype : str or np.dtype, optional
        Output data type. Default is np.float32.
    ix_fault : int, optional
        Deprecated: column index of the fault. Use x_fault instead.

    Returns
    -------
    ndarray (nx, nz)
        Faulted layers velocity model.
    """
    if dx <= 0 or dz <= 0:
        raise ValueError("Grid spacing dx and dz must be positive.")
    dt = parse_dtype(dtype)
    vs = np.asarray(v, dtype=dt)
    if vs.ndim != 1 or vs.size < 2:
        raise ValueError(f"'v' must be a 1D sequence of >= 2 velocities, got {v}.")

    if x_fault is None:
        if ix_fault is not None:
            fault_pos = float(ix_fault * dx)
        else:
            fault_pos = float(120 * dx if nx > 120 else (nx // 2) * dx)
    else:
        fault_pos = float(x_fault)

    if throw is None:
        throw_val = 8.0 * dz
    else:
        throw_val = float(throw)

    edges = np.linspace(0, nz, vs.size + 1).astype(int)
    base = np.full((nx, nz), vs[-1], dtype=dt)
    for k in range(vs.size - 1):
        base[:, edges[k] : edges[k + 1]] = vs[k]

    return add_fault_offset(base, x_fault=fault_pos, throw=throw_val, dx=dx, dz=dz).astype(dt)


def checkerboard(
    nx: int = 201,
    nz: int = 51,
    dx: float = 1.0,
    dz: float = 1.0,
    cell_x: Optional[float] = None,
    cell_z: Optional[float] = None,
    cell: Optional[float] = None,
    v0: float = 3000.0,
    dv: float = 600.0,
    dtype: Union[str, np.dtype] = np.float32,
) -> np.ndarray:
    """Generate a checkerboard pattern for resolution testing.

    Parameters
    ----------
    nx : int, optional
        Number of grid cells along x. Default is 201.
    nz : int, optional
        Number of grid cells along z. Default is 51.
    dx : float, optional
        Grid spacing along x in meters. Default is 1.0.
    dz : float, optional
        Grid spacing along z in meters. Default is 1.0.
    cell_x : float, optional
        Horizontal cell size in meters. Default is 20.0 * dx.
    cell_z : float, optional
        Vertical cell size in meters. Default is 20.0 * dz.
    cell : float, optional
        Uniform cell size (in meters if dx/dz provided, else cells).
    v0 : float, optional
        Background velocity in m/s. Default is 3000.0.
    dv : float, optional
        Velocity perturbation amplitude in m/s. Default is 600.0.
    dtype : str or np.dtype, optional
        Output data type. Default is np.float32.

    Returns
    -------
    ndarray (nx, nz)
        Checkerboard velocity model.
    """
    if dx <= 0 or dz <= 0:
        raise ValueError("Grid spacing dx and dz must be positive.")
    dt = parse_dtype(dtype)

    if cell is not None:
        cx_m = float(cell) * dx if cell_x is None else float(cell_x)
        cz_m = float(cell) * dz if cell_z is None else float(cell_z)
    else:
        cx_m = float(cell_x) if cell_x is not None else 20.0 * dx
        cz_m = float(cell_z) if cell_z is not None else 20.0 * dz

    x = (np.arange(nx, dtype=dt) * dx)[:, None]
    z = (np.arange(nz, dtype=dt) * dz)[None, :]
    pat = ((x // cx_m) + (z // cz_m)) % 2
    return np.where(pat == 0, v0 + dv, v0 - dv).astype(dt)


def salt_dome(
    nx: int = 201,
    nz: int = 51,
    dx: float = 1.0,
    dz: float = 1.0,
    cx: Optional[float] = None,
    width: Optional[float] = None,
    v_salt: float = 4500.0,
    v_sed: float = 2500.0,
    dtype: Union[str, np.dtype] = np.float32,
) -> np.ndarray:
    """Generate a salt dome model with semi-elliptical shape.

    Parameters
    ----------
    nx : int, optional
        Number of grid cells along x. Default is 201.
    nz : int, optional
        Number of grid cells along z. Default is 51.
    dx : float, optional
        Grid spacing along x in meters. Default is 1.0.
    dz : float, optional
        Grid spacing along z in meters. Default is 1.0.
    cx : float, optional
        Horizontal center position of dome in meters. Default is 140.0 * dx.
    width : float, optional
        Horizontal semi-axis width in meters. Default is 40.0 * dx.
    v_salt : float, optional
        Velocity inside salt body in m/s. Default is 4500.0.
    v_sed : float, optional
        Sediment velocity at surface in m/s. Default is 2500.0.
    dtype : str or np.dtype, optional
        Output data type. Default is np.float32.

    Returns
    -------
    ndarray (nx, nz)
        Salt dome velocity model.
    """
    if dx <= 0 or dz <= 0:
        raise ValueError("Grid spacing dx and dz must be positive.")
    dt = parse_dtype(dtype)
    v = gradient_background(nx, nz, dx=dx, dz=dz, v0=v_sed, v1=4000.0, dtype=dt)

    center_x = float(cx * dx if cx is not None and cx < nx else (cx if cx is not None else 140.0 * dx))
    semi_width = float(width * dx if width is not None and width < nx else (width if width is not None else 40.0 * dx))

    x = (np.arange(nx, dtype=dt) * dx)[:, None]
    z = (np.arange(nz, dtype=dt) * dz)[None, :]
    z_max = nz * dz * 0.9

    r = ((x - center_x) / semi_width) ** 2 + np.clip((z_max - z) / z_max, 0, None) ** 2
    v[r < 1.0] = v_salt
    return v


def low_velocity_layer(
    nx: int = 201,
    nz: int = 51,
    dx: float = 1.0,
    dz: float = 1.0,
    z0: Optional[float] = None,
    thickness: Optional[float] = None,
    v_slow: float = 1200.0,
    dtype: Union[str, np.dtype] = np.float32,
    k0: Optional[int] = None,
) -> np.ndarray:
    """Generate a model with a buried low-velocity channel.

    Parameters
    ----------
    nx : int, optional
        Number of grid cells along x. Default is 201.
    nz : int, optional
        Number of grid cells along z. Default is 51.
    dx : float, optional
        Grid spacing along x in meters. Default is 1.0.
    dz : float, optional
        Grid spacing along z in meters. Default is 1.0.
    z0 : float, optional
        Top depth of the low velocity layer in meters. Default is 20.0 * dz.
    thickness : float, optional
        Layer thickness in meters. Default is 8.0 * dz.
    v_slow : float, optional
        Velocity in the low-velocity channel in m/s. Default is 1200.0.
    dtype : str or np.dtype, optional
        Output data type. Default is np.float32.
    k0 : int, optional
        Deprecated: row cell index of layer top. Use z0 instead.

    Returns
    -------
    ndarray (nx, nz)
        Low velocity layer model.
    """
    if dx <= 0 or dz <= 0:
        raise ValueError("Grid spacing dx and dz must be positive.")
    dt = parse_dtype(dtype)
    v = gradient_background(nx, nz, dx=dx, dz=dz, v0=1800.0, v1=4500.0, dtype=dt)

    if z0 is None:
        depth_top = float(k0 * dz) if k0 is not None else 20.0 * dz
    else:
        depth_top = float(z0)

    thick = float(thickness) if thickness is not None else 8.0 * dz

    k_start = int(round(depth_top / dz))
    k_end = int(round((depth_top + thick) / dz))
    v[:, k_start:k_end] = v_slow
    return v


def heterogeneous_anomaly_model(
    nx: int = 384,
    nz: int = 122,
    dx: float = 1.0,
    dz: float = 1.0,
    seed: int = 7,
    dtype: Union[str, np.dtype] = np.float32,
) -> np.ndarray:
    """Generate a complex synthetic model.

    Parameters
    ----------
    nx : int, optional
        Number of grid cells along x. Default is 384.
    nz : int, optional
        Number of grid cells along z. Default is 122.
    dx : float, optional
        Grid spacing along x in meters. Default is 1.0.
    dz : float, optional
        Grid spacing along z in meters. Default is 1.0.
    seed : int, optional
        Random seed for reproducibility. Default is 7.
    dtype : str or np.dtype, optional
        Output data type. Default is np.float32.

    Returns
    -------
    ndarray (nx, nz)
        Complex heterogeneous velocity model.
    """
    if dx <= 0 or dz <= 0:
        raise ValueError("Grid spacing dx and dz must be positive.")
    dt = parse_dtype(dtype)
    rng = np.random.default_rng(seed)
    v = gradient_background(nx, nz, dx=dx, dz=dz, v0=1500.0, v1=4800.0, dtype=dt)
    v = add_gaussian_anomaly(v, (nx * dx) * 0.72, (nz * dz) * 0.35, 30.0 * dx, 12.0 * dz, 1200.0, dx=dx, dz=dz)
    v = add_gaussian_anomaly(v, (nx * dx) * 0.25, (nz * dz) * 0.55, 25.0 * dx, 10.0 * dz, -700.0, dx=dx, dz=dz)
    v = add_gaussian_anomaly(v, (nx * dx) * 0.5, (nz * dz) * 0.75, 40.0 * dx, 15.0 * dz, 900.0, dx=dx, dz=dz)
    noise = rng.normal(0.0, 1.0, (nx, nz)).astype(dt)
    k = 7
    pad = k // 2
    padded = np.pad(noise, pad, mode="edge")
    smooth = np.zeros_like(noise)
    for di in range(k):
        for dj in range(k):
            smooth += padded[di : di + nx, dj : dj + nz]
    smooth /= k * k
    return np.clip(v * (1.0 + 0.05 * smooth / smooth.std()), 500.0, 6000.0).astype(dt)

