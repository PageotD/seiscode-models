"""Object-oriented interface for 2D subsurface velocity models."""
from pathlib import Path
from typing import Union, Tuple, Optional
import numpy as np

from .models.primitives import gradient_background, add_gaussian_anomaly, add_fault_offset
from .models.geological import (
    dipping_layers,
    sinusoidal_interface,
    faulted_layers,
    checkerboard,
    salt_dome,
    low_velocity_layer,
    heterogeneous_anomaly_model,
)
from .utils.dtypes import parse_dtype
from .utils.validation import validate_model
from .utils.io import export_model
from .utils.visualization import quick_view

class VelocityModel:
    """Represents a 2D subsurface velocity model with grid metadata and method chaining."""

    def __init__(self, data: np.ndarray, dx: float = 1.0, dz: float = 1.0) -> None:
        """Initialize a VelocityModel from an existing NumPy array.

        Parameters
        ----------
        data : ndarray (nx, nz)
            Velocity model array in m/s.
        dx, dz : float, optional
            Grid spacing in meters. Default is 1.0.
        """
        validate_model(data, dx=dx, dz=dz)
        self._data: np.ndarray = data
        self._dx: float = float(dx)
        self._dz: float = float(dz)

    @property
    def data(self) -> np.ndarray:
        """Underlying NumPy array (nx, nz)."""
        return self._data

    @property
    def nx(self) -> int:
        """Number of grid cells along x."""
        return self._data.shape[0]

    @property
    def nz(self) -> int:
        """Number of grid cells along z."""
        return self._data.shape[1]

    @property
    def dx(self) -> float:
        """Grid spacing along x in meters."""
        return self._dx

    @property
    def dz(self) -> float:
        """Grid spacing along z in meters."""
        return self._dz

    @property
    def dtype(self) -> np.dtype:
        """NumPy data type precision (float32 or float64)."""
        return self._data.dtype

    # --- Factory Methods (Constructeurs géologiques orientés objet) ---

    @classmethod
    def gradient(
        cls, 
        nx: int = 201, 
        nz: int = 51, 
        dx: float = 1.0, 
        dz: float = 1.0, 
        v0: float = 1500.0, 
        v1: float = 4500.0, 
        dtype: Union[str, np.dtype] = np.float32
    ) -> "VelocityModel":
        """Create a model with a linear velocity gradient with depth."""
        arr = gradient_background(nx, nz, v0=v0, v1=v1, dtype=dtype)
        return cls(arr, dx=dx, dz=dz)

    @classmethod
    def salt_dome(
        cls, 
        nx: int = 201, 
        nz: int = 51, 
        dx: float = 1.0, 
        dz: float = 1.0, 
        cx: int = 140, 
        width: float = 40.0, 
        v_salt: float = 4500.0, 
        v_sed: float = 2500.0, 
        dtype: Union[str, np.dtype] = np.float32
    ) -> "VelocityModel":
        """Create a salt dome model."""
        arr = salt_dome(nx, nz, cx=cx, width=width, v_salt=v_salt, v_sed=v_sed, dtype=dtype)
        return cls(arr, dx=dx, dz=dz)

    @classmethod
    def dipping_layers(
        cls, 
        nx: int = 201, 
        nz: int = 51, 
        dx: float = 1.0, 
        dz: float = 1.0, 
        v: Tuple[float, ...] = (1800.0, 2400.0, 3200.0, 4200.0), 
        dip_deg: float = 8.0, 
        dtype: Union[str, np.dtype] = np.float32
    ) -> "VelocityModel":
        """Create a dipping layers model."""
        arr = dipping_layers(nx=nx, nz=nz, dx=dx, dz=dz, v=v, dip_deg=dip_deg, dtype=dtype)
        return cls(arr, dx=dx, dz=dz)

    @classmethod
    def sinusoidal_interface(
        cls, 
        nx: int = 201, 
        nz: int = 51, 
        dx: float = 1.0, 
        dz: float = 1.0, 
        period: float = 100.0, 
        amplitude: float = 8.0, 
        v_top: float = 2000.0, 
        v_bottom: float = 3500.0, 
        dtype: Union[str, np.dtype] = np.float32
    ) -> "VelocityModel":
        """Create a model with a sinusoidal interface."""
        arr = sinusoidal_interface(nx=nx, nz=nz, period=period, amplitude=amplitude, v_top=v_top, v_bottom=v_bottom, dtype=dtype)
        return cls(arr, dx=dx, dz=dz)

    @classmethod
    def faulted_layers(
        cls, 
        nx: int = 201, 
        nz: int = 51, 
        dx: float = 1.0, 
        dz: float = 1.0, 
        ix_fault: int = 120, 
        throw: int = 8, 
        v: Tuple[float, ...] = (1800.0, 2400.0, 3200.0, 4200.0), 
        dtype: Union[str, np.dtype] = np.float32
    ) -> "VelocityModel":
        """Create a faulted stratigraphic layers model."""
        arr = faulted_layers(nx=nx, nz=nz, ix_fault=ix_fault, throw=throw, v=v, dtype=dtype)
        return cls(arr, dx=dx, dz=dz)

    @classmethod
    def checkerboard(
        cls, 
        nx: int = 201, 
        nz: int = 51, 
        dx: float = 1.0, 
        dz: float = 1.0, 
        cell: int = 20, 
        v0: float = 3000.0, 
        dv: float = 600.0, 
        dtype: Union[str, np.dtype] = np.float32
    ) -> "VelocityModel":
        """Create a checkerboard resolution test model."""
        arr = checkerboard(nx=nx, nz=nz, cell=cell, v0=v0, dv=dv, dtype=dtype)
        return cls(arr, dx=dx, dz=dz)

    @classmethod
    def low_velocity_layer(
        cls, 
        nx: int = 201, 
        nz: int = 51, 
        dx: float = 1.0, 
        dz: float = 1.0, 
        k0: int = 20, 
        thickness: int = 8, 
        v_slow: float = 1200.0, 
        dtype: Union[str, np.dtype] = np.float32
    ) -> "VelocityModel":
        """Create a model with a buried low-velocity channel."""
        arr = low_velocity_layer(nx=nx, nz=nz, k0=k0, thickness=thickness, v_slow=v_slow, dtype=dtype)
        return cls(arr, dx=dx, dz=dz)

    @classmethod
    def heterogeneous_anomaly_model(
        cls, 
        nx: int = 384, 
        nz: int = 122, 
        dx: float = 1.0, 
        dz: float = 1.0, 
        seed: int = 7, 
        dtype: Union[str, np.dtype] = np.float32
    ) -> "VelocityModel":
        """Create a complex heterogeneous synthetic anomaly model."""
        arr = heterogeneous_anomaly_model(nx=nx, nz=nz, seed=seed, dtype=dtype)
        return cls(arr, dx=dx, dz=dz)

    # --- Modifier Methods (Fluent Interface / Method Chaining) ---

    def add_anomaly(
        self, 
        cx: float, 
        cz: float, 
        sigma_x: float, 
        sigma_z: float, 
        dv: float, 
        vmin: float = 200.0, 
        vmax: float = 8000.0
    ) -> "VelocityModel":
        """Add a Gaussian velocity anomaly in place and return self for method chaining."""
        self._data = add_gaussian_anomaly(self._data, cx, cz, sigma_x, sigma_z, dv, vmin=vmin, vmax=vmax)
        return self

    def add_fault(self, ix_fault: int, throw: int) -> "VelocityModel":
        """Apply a fault offset in place and return self for method chaining."""
        self._data = add_fault_offset(self._data, ix_fault, throw)
        return self

    # --- Utility Methods ---

    def export(self, filepath: Union[str, Path], format: str = "bin") -> None:
        """Export the model to a file compatible with seismic solvers."""
        export_model(self._data, filepath, format=format, dx=self._dx, dz=self._dz)

    def plot(self, title: str = "Velocity model") -> None:
        """Display the velocity model."""
        quick_view(self._data, dx=self._dx, dz=self._dz, title=title)