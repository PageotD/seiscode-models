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
    """Generate dipping planar layers with a uniform dip angle.

    Creates a 2D velocity model composed of planar geological strata dipping
    at an angle `dip_deg`. Depths are referenced relative to the horizontal
    midpoint of the model.

    Parameters
    ----------
    nx : int, optional
        Number of grid cells along horizontal x-axis. Default is 201.
    nz : int, optional
        Number of grid cells along vertical z-axis (depth). Default is 51.
    dx : float, optional
        Grid spacing along x in meters. Default is 1.0.
    dz : float, optional
        Grid spacing along z in meters. Default is 1.0.
    v : sequence of float, optional
        Layer velocities from top to bottom in m/s. Default is (1800.0, 2400.0, 3200.0, 4200.0).
    dip_deg : float, optional
        Dip angle in degrees (positive dips downward towards increasing x). Default is 8.0.
    thickness : float, optional
        Perpendicular or vertical layer thickness in meters. Default is 10.0.
    dtype : str or np.dtype, optional
        Output NumPy floating-point precision ('f32', 'f64', np.float32, np.float64).
        Default is np.float32.

    Returns
    -------
    ndarray of shape (nx, nz)
        2D velocity field in m/s with the specified dtype.

    Raises
    ------
    ValueError
        If `dx` or `dz` are not strictly positive.

    Examples
    --------
    >>> import seiscode_models as sm
    >>> v = sm.dipping_layers(nx=100, nz=50, dx=5.0, dz=5.0, dip_deg=12.0)
    >>> v.shape
    (100, 50)
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
    """Generate a two-layer velocity model separated by a sinusoidal undulating interface.

    Simulates folded strata or an undulating basement boundary defined by:
    ``z(x) = z_mean + amplitude * sin(2 * pi * x / period)``.

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
        Peak amplitude of the interface oscillation in meters. Default is 8.0.
    z_mean : float, optional
        Mean depth of the interface in meters. Defaults to 45% of total depth (``0.45 * nz * dz``).
    v_top : float, optional
        Velocity of the upper medium in m/s. Default is 2000.0.
    v_bottom : float, optional
        Velocity of the lower medium in m/s. Default is 3500.0.
    dtype : str or np.dtype, optional
        Output NumPy floating-point precision ('f32', 'f64', np.float32, np.float64).
        Default is np.float32.

    Returns
    -------
    ndarray of shape (nx, nz)
        2D velocity field in m/s with the specified dtype.

    Raises
    ------
    ValueError
        If `dx` or `dz` are not strictly positive.

    Examples
    --------
    >>> import seiscode_models as sm
    >>> v = sm.sinusoidal_interface(nx=150, nz=60, dx=10.0, dz=5.0, period=500.0, amplitude=50.0)
    >>> v.shape
    (150, 60)
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
    """Generate stratified horizontal layers offset by a vertical normal fault.

    Creates horizontally stratified layers and shifts the layers across a vertical
    fault line located at `x_fault`.

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
        Horizontal position of the fault in meters. If None and `ix_fault` is None,
        defaults to ``120 * dx`` (or model midpoint).
    throw : float, optional
        Vertical fault displacement (throw) in meters. Positive values displace the downthrown block.
        Default is ``8.0 * dz``.
    v : sequence of float, optional
        Layer velocities from top to bottom in m/s. Default is (1800.0, 2400.0, 3200.0, 4200.0).
    dtype : str or np.dtype, optional
        Output NumPy floating-point precision ('f32', 'f64', np.float32, np.float64).
        Default is np.float32.
    ix_fault : int, optional
        Deprecated: column index of the fault. Prefer using `x_fault` in physical units (meters).

    Returns
    -------
    ndarray of shape (nx, nz)
        2D velocity model with faulted stratigraphic layers.

    Raises
    ------
    ValueError
        If `v` contains fewer than 2 velocities, or if `dx`/`dz` are not positive.

    Examples
    --------
    >>> import seiscode_models as sm
    >>> v = sm.faulted_layers(nx=120, nz=60, dx=10.0, dz=10.0, x_fault=600.0, throw=50.0)
    >>> v.shape
    (120, 60)
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

    return add_fault_offset(
        base, x_fault=fault_pos, throw=throw_val, dx=dx, dz=dz
    ).astype(dt)


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
    """Generate an alternating checkerboard pattern for seismic resolution tests.

    Alternates positive and negative velocity perturbations (+dv / -dv) around
    a background velocity v0 in rectangular cells of dimension `cell_x` by `cell_z`.

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
        Horizontal dimension of each checkerboard tile in meters. Default is ``20.0 * dx``.
    cell_z : float, optional
        Vertical dimension of each checkerboard tile in meters. Default is ``20.0 * dz``.
    cell : float, optional
        Deprecated uniform cell size in meters (or cells if dx=1, dz=1).
    v0 : float, optional
        Base background velocity in m/s. Default is 3000.0.
    dv : float, optional
        Velocity perturbation amplitude in m/s. Default is 600.0.
    dtype : str or np.dtype, optional
        Output NumPy floating-point precision ('f32', 'f64', np.float32, np.float64).
        Default is np.float32.

    Returns
    -------
    ndarray of shape (nx, nz)
        Checkerboard velocity model alternating between `v0 + dv` and `v0 - dv`.

    Raises
    ------
    ValueError
        If `dx` or `dz` are not strictly positive.

    Examples
    --------
    >>> import seiscode_models as sm
    >>> v = sm.checkerboard(nx=100, nz=100, dx=5.0, dz=5.0, cell_x=50.0, cell_z=50.0)
    >>> v.min(), v.max()
    (2400.0, 3600.0)
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
    """Generate a realistic salt dome intrusion embedded in compacted sediments.

    Creates a background gradient representing sedimentary compaction, intruded
    by a semi-elliptical high-velocity salt body rooted at the base of the model.

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
        Horizontal center position of the salt dome in meters. Default is ``140.0 * dx``.
    width : float, optional
        Horizontal half-width (semi-axis) of the salt body in meters. Default is ``40.0 * dx``.
    v_salt : float, optional
        Constant velocity inside the salt dome in m/s. Default is 4500.0.
    v_sed : float, optional
        Sediment velocity at the surface in m/s (increases with depth to 4000 m/s). Default is 2500.0.
    dtype : str or np.dtype, optional
        Output NumPy floating-point precision ('f32', 'f64', np.float32, np.float64).
        Default is np.float32.

    Returns
    -------
    ndarray of shape (nx, nz)
        2D salt dome velocity model.

    Raises
    ------
    ValueError
        If `dx` or `dz` are not strictly positive.

    Examples
    --------
    >>> import seiscode_models as sm
    >>> v = sm.salt_dome(nx=200, nz=80, dx=10.0, dz=10.0, cx=1000.0, width=300.0)
    >>> v.shape
    (200, 80)
    """
    if dx <= 0 or dz <= 0:
        raise ValueError("Grid spacing dx and dz must be positive.")
    dt = parse_dtype(dtype)
    v = gradient_background(nx, nz, dx=dx, dz=dz, v0=v_sed, v1=4000.0, dtype=dt)

    center_x = float(
        cx * dx
        if cx is not None and cx < nx
        else (cx if cx is not None else 140.0 * dx)
    )
    semi_width = float(
        width * dx
        if width is not None and width < nx
        else (width if width is not None else 40.0 * dx)
    )

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
    """Generate a velocity model with a buried low-velocity layer or waveguide channel.

    Embeds a low-velocity horizontal layer (e.g. gas sand, weathered layer, or deep waveguide)
    within a linear gradient background.

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
        Top depth of the low-velocity layer in meters. Default is ``20.0 * dz``.
    thickness : float, optional
        Thickness of the low-velocity layer in meters. Default is ``8.0 * dz``.
    v_slow : float, optional
        Constant velocity inside the layer in m/s. Default is 1200.0.
    dtype : str or np.dtype, optional
        Output NumPy floating-point precision ('f32', 'f64', np.float32, np.float64).
        Default is np.float32.
    k0 : int, optional
        Deprecated: vertical grid index of the layer top. Use `z0` in meters instead.

    Returns
    -------
    ndarray of shape (nx, nz)
        2D velocity model containing the buried low-velocity channel.

    Raises
    ------
    ValueError
        If `dx` or `dz` are not strictly positive.

    Examples
    --------
    >>> import seiscode_models as sm
    >>> v = sm.low_velocity_layer(nx=100, nz=50, dx=10.0, dz=10.0, z0=200.0, thickness=80.0)
    >>> v.min()
    1200.0
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
    """Generate a complex synthetic model featuring multiple Gaussian anomalies and stochastic clutter.

    Combines a vertical linear gradient, positive and negative Gaussian anomalies,
    and a smoothed random Gaussian noise perturbation. Frequently used as a benchmark
    for Full Waveform Inversion (FWI) algorithms.

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
        Random seed for the stochastic noise generation. Default is 7.
    dtype : str or np.dtype, optional
        Output NumPy floating-point precision ('f32', 'f64', np.float32, np.float64).
        Default is np.float32.

    Returns
    -------
    ndarray of shape (nx, nz)
        Complex heterogeneous velocity model clipped to [500.0, 6000.0] m/s.

    Raises
    ------
    ValueError
        If `dx` or `dz` are not strictly positive.

    Examples
    --------
    >>> import seiscode_models as sm
    >>> v = sm.heterogeneous_anomaly_model(nx=200, nz=80, seed=42)
    >>> v.shape
    (200, 80)
    """
    if dx <= 0 or dz <= 0:
        raise ValueError("Grid spacing dx and dz must be positive.")
    dt = parse_dtype(dtype)
    rng = np.random.default_rng(seed)
    v = gradient_background(nx, nz, dx=dx, dz=dz, v0=1500.0, v1=4800.0, dtype=dt)
    v = add_gaussian_anomaly(
        v,
        (nx * dx) * 0.72,
        (nz * dz) * 0.35,
        30.0 * dx,
        12.0 * dz,
        1200.0,
        dx=dx,
        dz=dz,
    )
    v = add_gaussian_anomaly(
        v,
        (nx * dx) * 0.25,
        (nz * dz) * 0.55,
        25.0 * dx,
        10.0 * dz,
        -700.0,
        dx=dx,
        dz=dz,
    )
    v = add_gaussian_anomaly(
        v, (nx * dx) * 0.5, (nz * dz) * 0.75, 40.0 * dx, 15.0 * dz, 900.0, dx=dx, dz=dz
    )
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


def layered_model(
    nx: int = 201,
    nz: int = 51,
    dx: float = 1.0,
    dz: float = 1.0,
    thicknesses: Sequence[float] = (50.0, 50.0, 100.0),
    velocities: Sequence[float] = (1500.0, 2200.0, 3000.0, 4000.0),
    dtype: Union[str, np.dtype] = np.float32,
) -> np.ndarray:
    """Generate a 1D/2D tabular horizontally layered velocity model.

    Creates horizontally stratified geological layers defined by explicit thicknesses
    in meters. The last velocity in `velocities` acts as the basement half-space
    if the model depth exceeds the cumulative thickness of the upper layers.

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
    thicknesses : sequence of float, optional
        Thicknesses of the upper layers in meters. Length should be `len(velocities) - 1`.
        Default is (50.0, 50.0, 100.0).
    velocities : sequence of float, optional
        Velocities of each layer from top to bottom (including basement) in m/s.
        Default is (1500.0, 2200.0, 3000.0, 4000.0).
    dtype : str or np.dtype, optional
        Output NumPy floating-point precision ('f32', 'f64', np.float32, np.float64).
        Default is np.float32.

    Returns
    -------
    ndarray of shape (nx, nz)
        2D horizontally layered velocity model in m/s.

    Raises
    ------
    ValueError
        If `dx` or `dz` are not positive, or if `len(velocities) != len(thicknesses) + 1`.

    Examples
    --------
    >>> import seiscode_models as sm
    >>> v = sm.layered_model(nx=50, nz=40, dx=5.0, dz=5.0, thicknesses=(50.0, 50.0), velocities=(1500.0, 2500.0, 3500.0))
    >>> v[0, 0]
    1500.0
    """
    if dx <= 0 or dz <= 0:
        raise ValueError("Grid spacing dx and dz must be positive.")
    if len(velocities) != len(thicknesses) + 1:
        raise ValueError(
            f"Expected len(velocities) == len(thicknesses) + 1, got "
            f"{len(velocities)} velocities and {len(thicknesses)} thicknesses."
        )

    dt = parse_dtype(dtype)
    v_out = np.full((nx, nz), velocities[-1], dtype=dt)

    depth_top = 0.0
    for thick, vel in zip(thicknesses, velocities[:-1]):
        depth_bottom = depth_top + thick
        k_top = int(round(depth_top / dz))
        k_bottom = int(round(depth_bottom / dz))
        if k_top < nz:
            v_out[:, k_top : min(k_bottom, nz)] = vel
        depth_top = depth_bottom

    return v_out


def compacted_sediments(
    nx: int = 201,
    nz: int = 51,
    dx: float = 1.0,
    dz: float = 1.0,
    v0: float = 1600.0,
    k: float = 0.6,
    dtype: Union[str, np.dtype] = np.float32,
) -> np.ndarray:
    r"""Generate a realistic sediment compaction velocity model :math:`v(z) = v_0 + k \cdot z`.

    Models continuous mechanical compaction in sedimentary basins where velocity
    increases linearly with depth :math:`z` at compaction gradient rate :math:`k` (s\ :sup:`-1`).

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
    v0 : float, optional
        Velocity at the top surface in m/s. Default is 1600.0.
    k : float, optional
        Compaction velocity gradient in s\ :sup:`-1` (m/s per meter of depth). Default is 0.6.
    dtype : str or np.dtype, optional
        Output NumPy floating-point precision ('f32', 'f64', np.float32, np.float64).
        Default is np.float32.

    Returns
    -------
    ndarray of shape (nx, nz)
        2D compacted sediment velocity field in m/s.

    Raises
    ------
    ValueError
        If `dx` or `dz` are not positive.

    Examples
    --------
    >>> import seiscode_models as sm
    >>> v = sm.compacted_sediments(nx=100, nz=50, dx=10.0, dz=10.0, v0=1500.0, k=0.5)
    >>> v[0, 0]
    1500.0
    >>> v[0, -1]  # depth = 490m -> 1500 + 0.5 * 490 = 1745 m/s
    1745.0
    """
    if dx <= 0 or dz <= 0:
        raise ValueError("Grid spacing dx and dz must be positive.")
    dt = parse_dtype(dtype)
    z = (np.arange(nz, dtype=dt) * dz)[None, :]
    v_col = v0 + k * z
    return np.broadcast_to(v_col, (nx, nz)).copy()


def add_water_bottom(
    v: np.ndarray,
    water_depth: Union[float, Sequence[float], np.ndarray] = 100.0,
    v_water: float = 1500.0,
    dx: float = 1.0,
    dz: float = 1.0,
) -> np.ndarray:
    """Impose a water layer (bathymetry) at the top of an existing velocity model.

    Sets all velocities above the seafloor depth to the constant water velocity `v_water`.
    Accepts either a flat water depth in meters, or a 1D bathymetric depth profile `z_seafloor(x)`.

    Parameters
    ----------
    v : ndarray of shape (nx, nz)
        Input subsurface velocity model to modify (not modified in place).
    water_depth : float or sequence of float or 1D ndarray, optional
        Depth of the seabed in meters. Can be a single float (flat water bottom)
        or a sequence of length `nx` representing irregular bathymetry. Default is 100.0.
    v_water : float, optional
        Acoustic velocity in water in m/s. Default is 1500.0.
    dx : float, optional
        Grid spacing along x in meters. Default is 1.0.
    dz : float, optional
        Grid spacing along z in meters. Default is 1.0.

    Returns
    -------
    ndarray of shape (nx, nz)
        A copy of `v` with the water layer imposed above the seafloor.

    Raises
    ------
    ValueError
        If `dx` or `dz` are not positive, or if `water_depth` sequence length != `nx`.

    Examples
    --------
    >>> import seiscode_models as sm
    >>> bg = sm.gradient_background(50, 40, dx=5.0, dz=5.0, v0=2000.0, v1=4000.0)
    >>> marine = sm.add_water_bottom(bg, water_depth=50.0, v_water=1500.0, dx=5.0, dz=5.0)
    >>> marine[0, 0]
    1500.0
    """
    if dx <= 0 or dz <= 0:
        raise ValueError("Grid spacing dx and dz must be positive.")
    nx, nz = v.shape
    dt = v.dtype
    out = v.copy()

    if np.isscalar(water_depth):
        k_water = int(round(float(water_depth) / dz))
        k_clamp = max(0, min(k_water, nz))
        out[:, :k_clamp] = v_water
    else:
        bathymetry = np.asarray(water_depth, dtype=dt)
        if bathymetry.ndim != 1 or bathymetry.size != nx:
            raise ValueError(
                f"Bathymetry profile length ({bathymetry.size}) must match model nx ({nx})."
            )
        k_water = np.clip(np.round(bathymetry / dz).astype(int), 0, nz)
        z_indices = np.arange(nz)[None, :]
        mask = z_indices < k_water[:, None]
        out[mask] = v_water

    return out
