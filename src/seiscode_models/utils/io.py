"""Input/Output utilities for exporting velocity models to seismic solvers."""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from .validation import validate_model


def export_model(
    v: np.ndarray,
    filepath: Union[str, Path],
    format: str = "bin",
    dx: float = 1.0,
    dz: float = 1.0,
    order: str = "C",
) -> None:
    """Export a 2D velocity model to disk in binary format for seismic forward solvers.

    Supports flat raw binary format (for C or Fortran solvers like SPECFEM or Madagascar)
    as well as NumPy `.npy` format.

    Parameters
    ----------
    v : ndarray of shape (nx, nz)
        2D velocity field array to export.
    filepath : str or Path
        Destination file path on disk. Parent directories are created automatically.
    format : str, optional
        Target storage format:
        - 'bin' or 'raw': Flat raw binary file matching memory layout specified by `order`.
        - 'npy': Standard NumPy binary format preserving shape and precision dtype.
        Default is 'bin'.
    dx : float, optional
        Horizontal grid spacing in meters. Validated before export. Default is 1.0.
    dz : float, optional
        Vertical grid spacing in meters. Validated before export. Default is 1.0.
    order : str, optional
        Memory storage order for raw binary output:
        - 'C': Row-major order (standard in C / Python).
        - 'F': Column-major order (standard in Fortran).
        Ignored for 'npy' format. Default is 'C'.

    Raises
    ------
    ValueError
        If `v` fails validation or if an unsupported format string is provided.

    Examples
    --------
    >>> import seiscode_models as sm
    >>> v = sm.gradient_background(50, 50)
    >>> sm.export_model(v, "/tmp/model.bin", format="bin", order="F")
    """
    validate_model(v, dx=dx, dz=dz)
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    fmt = format.lower()
    if fmt in ("bin", "raw"):
        if order.upper() == "F":
            np.asfortranarray(v).tofile(filepath)
        else:
            v.tofile(filepath)
    elif fmt == "npy":
        if filepath.suffix != ".npy":
            filepath = filepath.with_suffix(".npy")
        np.save(filepath, v)
    else:
        raise ValueError(
            f"Unsupported export format: '{format}'. Use 'bin', 'raw', or 'npy'."
        )


def load_model(
    filepath: Union[str, Path],
    shape: Optional[Tuple[int, int]] = None,
    dtype: Union[str, np.dtype] = np.float32,
) -> np.ndarray:
    """Load a 2D velocity model from a raw binary or NumPy (.npy) file.

    Parameters
    ----------
    filepath : str or Path
        Path to the binary file on disk.
    shape : tuple of int, optional
        Target 2D dimensions `(nx, nz)`. Mandatory for raw binary files (.bin/.raw);
        optional for NumPy `.npy` files.
    dtype : str or np.dtype, optional
        Floating-point data type expected in the raw binary file. Default is np.float32.

    Returns
    -------
    ndarray of shape (nx, nz)
        Loaded 2D velocity model array.

    Raises
    ------
    ValueError
        If `shape` is missing for raw binary, or if file size does not match expected elements.

    Examples
    --------
    >>> import seiscode_models as sm
    >>> v = sm.load_model("/tmp/model.bin", shape=(50, 50))
    >>> v.shape
    (50, 50)
    """
    filepath = Path(filepath)
    if filepath.suffix == ".npy" or (shape is None and filepath.suffix == ".npy"):
        return np.load(filepath)
    if shape is None:
        raise ValueError("Shape must be provided for raw binary files.")
    v = np.fromfile(filepath, dtype=dtype)
    expected_size = shape[0] * shape[1]
    if v.size != expected_size:
        raise ValueError(
            f"File size mismatch: expected {expected_size} elements for shape {shape}, got {v.size}."
        )
    return v.reshape(shape)
