"""Geological velocity models supporting f32/f64 precision."""

import numpy as np

from ..utils.dtypes import parse_dtype
from .primitives import add_fault_offset, add_gaussian_anomaly, gradient_background


def dipping_layers(
    nx: int = 201,
    nz: int = 51,
    dx: float = 1.0,
    dz: float = 1.0,
    v: tuple = (1800.0, 2400.0, 3200.0, 4200.0),
    dip_deg: float = 8.0,
    thickness: float = 10.0,
    dtype=np.float32,
) -> np.ndarray:
    """Generate dipping planar layers with the same dip angle."""
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
    period: float = 100.0,
    amplitude: float = 8.0,
    v_top: float = 2000.0,
    v_bottom: float = 3500.0,
    dtype=np.float32,
) -> np.ndarray:
    """Generate two layers separated by a sinusoidal interface."""
    dt = parse_dtype(dtype)
    x = np.arange(nx, dtype=dt)
    z_int = nz * 0.45 + amplitude * np.sin(2 * np.pi * x / period)
    k_int = np.clip(z_int.astype(int), 0, nz)
    v = np.full((nx, nz), v_bottom, dtype=dt)
    v[k_int[:, None] > np.arange(nz)[None, :]] = v_top
    return v


def faulted_layers(
    nx: int = 201,
    nz: int = 51,
    ix_fault: int = 120,
    throw: int = 8,
    v: tuple = (1800.0, 2400.0, 3200.0, 4200.0),
    dtype=np.float32,
) -> np.ndarray:
    """Generate stratigraphic layers with a normal fault."""
    dt = parse_dtype(dtype)
    vs = np.asarray(v, dtype=dt)
    if vs.ndim != 1 or vs.size < 2:
        raise ValueError(f"'v' must be a 1D sequence of >= 2 velocities, got {v}.")
    edges = np.linspace(0, nz, vs.size + 1).astype(int)
    base = np.full((nx, nz), vs[-1], dtype=dt)
    for k in range(vs.size - 1):
        base[:, edges[k] : edges[k + 1]] = vs[k]
    return add_fault_offset(base, ix_fault, throw).astype(dt)


def checkerboard(
    nx: int = 201,
    nz: int = 51,
    cell: int = 20,
    v0: float = 3000.0,
    dv: float = 600.0,
    dtype=np.float32,
) -> np.ndarray:
    """Generate a checkerboard pattern for resolution testing."""
    dt = parse_dtype(dtype)
    ii = np.arange(nx, dtype=dt)[:, None]
    jj = np.arange(nz, dtype=dt)[None, :]
    pat = ((ii // cell) + (jj // cell)) % 2
    return np.where(pat == 0, v0 + dv, v0 - dv).astype(dt)


def salt_dome(
    nx: int = 201,
    nz: int = 51,
    cx: int = 140,
    width: float = 40.0,
    v_salt: float = 4500.0,
    v_sed: float = 2500.0,
    dtype=np.float32,
) -> np.ndarray:
    """Generate a salt dome model with semi-elliptical shape."""
    dt = parse_dtype(dtype)
    v = gradient_background(nx, nz, v_sed, 4000.0, dtype=dt)
    ii = np.arange(nx, dtype=dt)[:, None]
    jj = np.arange(nz, dtype=dt)[None, :]
    r = ((ii - cx) / width) ** 2 + np.clip((nz * 0.9 - jj) / (nz * 0.9), 0, None) ** 2
    v[r < 1.0] = v_salt
    return v


def low_velocity_layer(
    nx: int = 201,
    nz: int = 51,
    k0: int = 20,
    thickness: int = 8,
    v_slow: float = 1200.0,
    dtype=np.float32,
) -> np.ndarray:
    """Generate a model with a buried low-velocity channel."""
    dt = parse_dtype(dtype)
    v = gradient_background(nx, nz, 1800.0, 4500.0, dtype=dt)
    v[:, k0 : k0 + thickness] = v_slow
    return v


def heterogeneous_anomaly_model(
    nx: int = 384, nz: int = 122, seed: int = 7, dtype=np.float32
) -> np.ndarray:
    """Generate a complex synthetic model."""
    dt = parse_dtype(dtype)
    rng = np.random.default_rng(seed)
    v = gradient_background(nx, nz, 1500.0, 4800.0, dtype=dt)
    v = add_gaussian_anomaly(v, nx * 0.72, nz * 0.35, 30.0, 12.0, 1200.0)
    v = add_gaussian_anomaly(v, nx * 0.25, nz * 0.55, 25.0, 10.0, -700.0)
    v = add_gaussian_anomaly(v, nx * 0.5, nz * 0.75, 40.0, 15.0, 900.0)
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
