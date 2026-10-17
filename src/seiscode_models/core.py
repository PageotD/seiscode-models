"""Object-oriented interface for 2D subsurface velocity models."""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from .models.geological import (
    checkerboard,
    dipping_layers,
    faulted_layers,
    heterogeneous_anomaly_model,
    layered_model,
    low_velocity_layer,
    salt_dome,
    sinusoidal_interface,
)
from .models.primitives import (
    add_elliptical_inclusion,
    add_fault_offset,
    add_gaussian_anomaly,
    gradient_background,
)
from .utils.io import export_model
from .utils.validation import validate_model
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
        dtype: Union[str, np.dtype] = np.float32,
    ) -> "VelocityModel":
        """Create a VelocityModel with a linear velocity gradient with depth.

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
            Velocity at surface (z=0) in m/s. Default is 1500.0.
        v1 : float, optional
            Velocity at model bottom in m/s. Default is 4500.0.
        dtype : str or np.dtype, optional
            Output data type. Default is np.float32.

        Returns
        -------
        VelocityModel
            A new VelocityModel instance with gradient velocity field.
        """
        arr = gradient_background(nx=nx, nz=nz, dx=dx, dz=dz, v0=v0, v1=v1, dtype=dtype)
        return cls(arr, dx=dx, dz=dz)

    @classmethod
    def salt_dome(
        cls,
        nx: int = 201,
        nz: int = 51,
        dx: float = 1.0,
        dz: float = 1.0,
        cx: Optional[float] = None,
        width: Optional[float] = None,
        v_salt: float = 4500.0,
        v_sed: float = 2500.0,
        dtype: Union[str, np.dtype] = np.float32,
    ) -> "VelocityModel":
        """Create a VelocityModel featuring a salt dome intrusion in compacted sediments.

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
            Horizontal center coordinate of dome in meters. Default is 140.0 * dx.
        width : float, optional
            Horizontal half-width of the dome in meters. Default is 40.0 * dx.
        v_salt : float, optional
            Velocity of the salt body in m/s. Default is 4500.0.
        v_sed : float, optional
            Sediment velocity at surface in m/s. Default is 2500.0.
        dtype : str or np.dtype, optional
            Output data type. Default is np.float32.

        Returns
        -------
        VelocityModel
            A new VelocityModel instance with salt dome.
        """
        arr = salt_dome(
            nx=nx,
            nz=nz,
            dx=dx,
            dz=dz,
            cx=cx,
            width=width,
            v_salt=v_salt,
            v_sed=v_sed,
            dtype=dtype,
        )
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
        thickness: float = 10.0,
        dtype: Union[str, np.dtype] = np.float32,
    ) -> "VelocityModel":
        """Create a VelocityModel with dipping planar geological strata.

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
        v : tuple of float, optional
            Layer velocities from top to bottom in m/s.
        dip_deg : float, optional
            Dip angle in degrees. Default is 8.0.
        thickness : float, optional
            Thickness of each layer in meters. Default is 10.0.
        dtype : str or np.dtype, optional
            Output data type. Default is np.float32.

        Returns
        -------
        VelocityModel
            A new VelocityModel instance with dipping layers.
        """
        arr = dipping_layers(
            nx=nx,
            nz=nz,
            dx=dx,
            dz=dz,
            v=v,
            dip_deg=dip_deg,
            thickness=thickness,
            dtype=dtype,
        )
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
        z_mean: Optional[float] = None,
        v_top: float = 2000.0,
        v_bottom: float = 3500.0,
        dtype: Union[str, np.dtype] = np.float32,
    ) -> "VelocityModel":
        """Create a VelocityModel with two media separated by a sinusoidal boundary.

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
            Wavelength of the sinusoid in meters. Default is 100.0.
        amplitude : float, optional
            Amplitude of the oscillation in meters. Default is 8.0.
        z_mean : float, optional
            Mean depth of the interface in meters.
        v_top : float, optional
            Velocity of the upper medium in m/s. Default is 2000.0.
        v_bottom : float, optional
            Velocity of the lower medium in m/s. Default is 3500.0.
        dtype : str or np.dtype, optional
            Output data type. Default is np.float32.

        Returns
        -------
        VelocityModel
            A new VelocityModel instance with sinusoidal interface.
        """
        arr = sinusoidal_interface(
            nx=nx,
            nz=nz,
            dx=dx,
            dz=dz,
            period=period,
            amplitude=amplitude,
            z_mean=z_mean,
            v_top=v_top,
            v_bottom=v_bottom,
            dtype=dtype,
        )
        return cls(arr, dx=dx, dz=dz)

    @classmethod
    def layered(
        cls,
        nx: int = 201,
        nz: int = 51,
        dx: float = 1.0,
        dz: float = 1.0,
        thicknesses: Tuple[float, ...] = (50.0, 50.0, 100.0),
        velocities: Tuple[float, ...] = (1500.0, 2200.0, 3000.0, 4000.0),
        dtype: Union[str, np.dtype] = np.float32,
    ) -> "VelocityModel":
        """Create a VelocityModel with horizontal tabular layers.

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
        thicknesses : tuple of float, optional
            Layer thicknesses from top to bottom in meters. Length = len(velocities) - 1.
        velocities : tuple of float, optional
            Layer velocities in m/s including basement half-space.
        dtype : str or np.dtype, optional
            Output data type. Default is np.float32.

        Returns
        -------
        VelocityModel
            A new VelocityModel instance with horizontal layers.
        """
        arr = layered_model(
            nx=nx,
            nz=nz,
            dx=dx,
            dz=dz,
            thicknesses=thicknesses,
            velocities=velocities,
            dtype=dtype,
        )
        return cls(arr, dx=dx, dz=dz)

    @classmethod
    def faulted_layers(
        cls,
        nx: int = 201,
        nz: int = 51,
        dx: float = 1.0,
        dz: float = 1.0,
        x_fault: Optional[float] = None,
        throw: Optional[float] = None,
        v: Tuple[float, ...] = (1800.0, 2400.0, 3200.0, 4200.0),
        dtype: Union[str, np.dtype] = np.float32,
        ix_fault: Optional[int] = None,
    ) -> "VelocityModel":
        """Create a VelocityModel with stratified layers offset by a vertical normal fault.

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
            Horizontal fault position in meters. Default is 120.0 * dx.
        throw : float, optional
            Vertical fault displacement (throw) in meters. Default is 8.0 * dz.
        v : tuple of float, optional
            Layer velocities from top to bottom in m/s.
        dtype : str or np.dtype, optional
            Output data type. Default is np.float32.
        ix_fault : int, optional
            Deprecated: fault column index. Prefer using `x_fault` in meters.

        Returns
        -------
        VelocityModel
            A new VelocityModel instance with faulted layers.
        """
        arr = faulted_layers(
            nx=nx,
            nz=nz,
            dx=dx,
            dz=dz,
            x_fault=x_fault,
            throw=throw,
            v=v,
            dtype=dtype,
            ix_fault=ix_fault,
        )
        return cls(arr, dx=dx, dz=dz)

    @classmethod
    def checkerboard(
        cls,
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
    ) -> "VelocityModel":
        """Create a VelocityModel with a checkerboard resolution test pattern.

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
            Horizontal dimension of each cell in meters. Default is 20.0 * dx.
        cell_z : float, optional
            Vertical dimension of each cell in meters. Default is 20.0 * dz.
        cell : float, optional
            Uniform cell dimension.
        v0 : float, optional
            Base velocity in m/s. Default is 3000.0.
        dv : float, optional
            Velocity perturbation amplitude in m/s. Default is 600.0.
        dtype : str or np.dtype, optional
            Output data type. Default is np.float32.

        Returns
        -------
        VelocityModel
            A new VelocityModel instance with checkerboard pattern.
        """
        arr = checkerboard(
            nx=nx,
            nz=nz,
            dx=dx,
            dz=dz,
            cell_x=cell_x,
            cell_z=cell_z,
            cell=cell,
            v0=v0,
            dv=dv,
            dtype=dtype,
        )
        return cls(arr, dx=dx, dz=dz)

    @classmethod
    def low_velocity_layer(
        cls,
        nx: int = 201,
        nz: int = 51,
        dx: float = 1.0,
        dz: float = 1.0,
        z0: Optional[float] = None,
        thickness: Optional[float] = None,
        v_slow: float = 1200.0,
        dtype: Union[str, np.dtype] = np.float32,
        k0: Optional[int] = None,
    ) -> "VelocityModel":
        """Create a VelocityModel with a buried low-velocity channel.

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
            Top depth of the low-velocity channel in meters. Default is 20.0 * dz.
        thickness : float, optional
            Layer thickness in meters. Default is 8.0 * dz.
        v_slow : float, optional
            Velocity in the channel in m/s. Default is 1200.0.
        dtype : str or np.dtype, optional
            Output data type. Default is np.float32.
        k0 : int, optional
            Deprecated: vertical index of layer top. Prefer `z0` in meters.

        Returns
        -------
        VelocityModel
            A new VelocityModel instance with low-velocity channel.
        """
        arr = low_velocity_layer(
            nx=nx,
            nz=nz,
            dx=dx,
            dz=dz,
            z0=z0,
            thickness=thickness,
            v_slow=v_slow,
            dtype=dtype,
            k0=k0,
        )
        return cls(arr, dx=dx, dz=dz)

    @classmethod
    def heterogeneous_anomaly_model(
        cls,
        nx: int = 384,
        nz: int = 122,
        dx: float = 1.0,
        dz: float = 1.0,
        seed: int = 7,
        dtype: Union[str, np.dtype] = np.float32,
    ) -> "VelocityModel":
        """Create a VelocityModel with complex heterogeneous anomalies and stochastic noise.

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
            Random seed for noise generation. Default is 7.
        dtype : str or np.dtype, optional
            Output data type. Default is np.float32.

        Returns
        -------
        VelocityModel
            A new VelocityModel instance with heterogeneous anomalies.
        """
        arr = heterogeneous_anomaly_model(
            nx=nx, nz=nz, dx=dx, dz=dz, seed=seed, dtype=dtype
        )
        return cls(arr, dx=dx, dz=dz)

    # --- Modifier Methods (Fluent Interface / Method Chaining) ---

    def add_gaussian_anomaly(
        self,
        cx: float,
        cz: float,
        sigma_x: float,
        sigma_z: float,
        dv: float,
        vmin: float = 200.0,
        vmax: float = 8000.0,
    ) -> "VelocityModel":
        """Add a Gaussian velocity anomaly in place and return self for method chaining.

        Parameters
        ----------
        cx : float
            Horizontal center coordinate of the anomaly in meters.
        cz : float
            Vertical center coordinate (depth) of the anomaly in meters.
        sigma_x : float
            Horizontal standard deviation in meters.
        sigma_z : float
            Vertical standard deviation in meters.
        dv : float
            Peak velocity anomaly at center in m/s.
        vmin : float, optional
            Lower physical velocity bound in m/s. Default is 200.0.
        vmax : float, optional
            Upper physical velocity bound in m/s. Default is 8000.0.

        Returns
        -------
        VelocityModel
            Self reference for fluent method chaining.
        """
        self._data = add_gaussian_anomaly(
            self._data,
            cx,
            cz,
            sigma_x,
            sigma_z,
            dv,
            dx=self._dx,
            dz=self._dz,
            vmin=vmin,
            vmax=vmax,
        )
        return self

    def add_elliptical_inclusion(
        self,
        cx: float,
        cz: float,
        radius_x: float,
        radius_z: float,
        cv: float,
    ) -> "VelocityModel":
        """Add an elliptical velocity inclusion in place and return self for method chaining.

        Parameters
        ----------
        cx : float
            Horizontal center coordinate in meters.
        cz : float
            Vertical center coordinate (depth) in meters.
        radius_x : float
            Horizontal semi-axis in meters.
        radius_z : float
            Vertical semi-axis in meters.
        cv : float
            Constant velocity inside the ellipse in m/s.

        Returns
        -------
        VelocityModel
            Self reference for fluent method chaining.
        """
        self._data = add_elliptical_inclusion(
            self._data,
            cx,
            cz,
            radius_x,
            radius_z,
            cv,
            dx=self._dx,
            dz=self._dz,
        )
        return self

    def add_fault(
        self,
        x_fault: Optional[float] = None,
        throw: Optional[float] = None,
        ix_fault: Optional[int] = None,
    ) -> "VelocityModel":
        """Apply a vertical fault displacement in place and return self for chaining.

        Parameters
        ----------
        x_fault : float, optional
            Horizontal fault position in meters.
        throw : float, optional
            Vertical fault displacement in meters.
        ix_fault : int, optional
            Deprecated: fault column index. Prefer `x_fault` in meters.

        Returns
        -------
        VelocityModel
            Self reference for fluent method chaining.
        """
        self._data = add_fault_offset(
            self._data,
            x_fault=x_fault,
            throw=throw,
            dx=self._dx,
            dz=self._dz,
            ix_fault=ix_fault,
        )
        return self

    def clip(
        self,
        vmin: float = 200.0,
        vmax: float = 8000.0,
    ) -> "VelocityModel":
        """Clip velocities to physical bounds in place and return self for chaining.

        Parameters
        ----------
        vmin : float, optional
            Minimum velocity bound in m/s. Default is 200.0.
        vmax : float, optional
            Maximum velocity bound in m/s. Default is 8000.0.

        Returns
        -------
        VelocityModel
            Self reference for fluent method chaining.

        Raises
        ------
        ValueError
            If bounds are not finite or if vmin > vmax.
        """
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            raise ValueError("vmin and vmax must be finite.")

        if vmin > vmax:
            raise ValueError("vmin must be <= vmax.")

        self._data = np.clip(self._data, vmin, vmax)
        return self

    def smooth(self, k: int = 5) -> "VelocityModel":
        """Apply a k x k moving-average smoothing filter in place and return self.

        Parameters
        ----------
        k : int, optional
            Size of the square smoothing kernel. Must be an odd integer >= 3. Default is 5.

        Returns
        -------
        VelocityModel
            Self reference for fluent method chaining.

        Raises
        ------
        ValueError
            If k is even or less than 3.
        """
        if k < 3 or k % 2 == 0:
            raise ValueError(f"Smoothing kernel size must be odd and >= 3, got k={k}.")
        pad = k // 2
        padded = np.pad(self._data, pad, mode="edge")
        out = np.zeros_like(self._data)
        for di in range(k):
            for dj in range(k):
                out += padded[di : di + self.nx, dj : dj + self.nz]
        self._data = (out / (k * k)).astype(self._data.dtype)
        return self

    # --- Utility Methods ---
    @classmethod
    def load(
        cls,
        filepath: Union[str, Path],
        shape: Optional[Tuple[int, int]] = None,
        dtype: Union[str, np.dtype] = np.float32,
        dx: float = 1.0,
        dz: float = 1.0,
    ) -> "VelocityModel":
        """Load a velocity model from a raw binary (.bin) or NumPy (.npy) file.

        Parameters
        ----------
        filepath : str or Path
            Path to the model file.
        shape : tuple of int, optional
            Grid dimensions (nx, nz). Required for raw binary files.
        dtype : str or np.dtype, optional
            Data type of the stored array. Default is np.float32.
        dx : float, optional
            Horizontal grid spacing in meters. Default is 1.0.
        dz : float, optional
            Vertical grid spacing in meters. Default is 1.0.

        Returns
        -------
        VelocityModel
            A new VelocityModel populated with loaded array and grid metadata.
        """
        from .utils.io import load_model

        arr = load_model(filepath, shape=shape, dtype=dtype)
        return cls(arr, dx=dx, dz=dz)

    def copy(self) -> "VelocityModel":
        """Return an independent deep copy of the velocity model.

        Returns
        -------
        VelocityModel
            A cloned VelocityModel with copied array buffer and identical metadata.
        """
        return VelocityModel(self._data.copy(), dx=self._dx, dz=self._dz)

    def export(
        self, filepath: Union[str, Path], format: str = "bin", order: str = "C"
    ) -> None:
        """Export the velocity model to disk for seismic solvers.

        Parameters
        ----------
        filepath : str or Path
            Target file path.
        format : str, optional
            Export format ('bin', 'raw', or 'npy'). Default is 'bin'.
        order : str, optional
            Memory layout for binary export ('C' or 'F'). Default is 'C'.
        """
        export_model(
            self._data, filepath, format=format, dx=self._dx, dz=self._dz, order=order
        )

    def plot(self, title: str = "Velocity model") -> None:
        """Display the velocity model using matplotlib.

        Parameters
        ----------
        title : str, optional
            Title for the plot figure. Default is "Velocity model".
        """
        quick_view(self._data, dx=self._dx, dz=self._dz, title=title)

    # --- Statistics ---

    @property
    def vmin(self) -> float:
        """Minimum velocity in m/s."""
        return float(self._data.min())

    @property
    def vmax(self) -> float:
        """Maximum velocity in m/s."""
        return float(self._data.max())

    def stats(self) -> dict:
        """Return summary statistics of the velocity field."""
        return {
            "shape": (self.nx, self.nz),
            "dx": self._dx,
            "dz": self._dz,
            "dtype": str(self._data.dtype),
            "vmin": self.vmin,
            "vmax": self.vmax,
            "vmean": float(self._data.mean()),
            "vstd": float(self._data.std()),
        }

    # --- Dunder ---

    def __repr__(self) -> str:
        return (
            f"VelocityModel(nx={self.nx}, nz={self.nz}, dx={self._dx}, "
            f"dz={self._dz}, dtype={self._data.dtype})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VelocityModel):
            return NotImplemented
        return (
            self._data.shape == other._data.shape
            and np.array_equal(self._data, other._data)
            and self._dx == other._dx
            and self._dz == other._dz
        )
