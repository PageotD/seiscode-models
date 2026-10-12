"""Input/Output utilities for exporting velocity models to seismic solvers."""
from pathlib import Path
import numpy as np
from typing import Union, Tuple
from .validation import validate_model

def export_model(
    v: np.ndarray, 
    filepath: Union[str, Path], 
    format: str = "bin", 
    dx: float = 1.0, 
    dz: float = 1.0
) -> None:
    """Export a velocity model to a file compatible with seismic solvers.

    Parameters
    ----------
    v : ndarray (nx, nz)
        Velocity model array.
    filepath : str or Path
        Destination file path.
    format : str, optional
        Export format:
        - 'bin' or 'raw': Flat binary file (C-contiguous or Fortran-contiguous depending on solver needs).
        - 'npy': Standard NumPy binary format preserving dtype and shape.
        Default is 'bin'.
    dx, dz : float, optional
        Grid spacing (validated before export).
    """
    validate_model(v, dx=dx, dz=dz)
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    fmt = format.lower()
    if fmt in ("bin", "raw"):
        # Enregistrement en binaire brut (souvent attendu en float32/float64 pur par les solveurs C/Fortran)
        v.tofile(filepath)
    elif fmt == "npy":
        np.save(filepath, v)
    else:
        raise ValueError(f"Unsupported export format: '{format}'. Use 'bin', 'raw', or 'npy'.")

def load_model(filepath: Union[str, Path], shape: Tuple[int, int], dtype=np.float32) -> np.ndarray:
    """Load a velocity model from a raw binary file.

    Parameters
    ----------
    filepath : str or Path
        Path to the binary file.
    shape : tuple of int
        Expected shape (nx, nz) of the model.
    dtype : str or np.dtype, optional
        Data type of the binary file. Default is np.float32.

    Returns
    -------
    ndarray (nx, nz)
        Loaded velocity model.
    """
    filepath = Path(filepath)
    if filepath.suffix == ".npy":
        return np.load(filepath)
    
    v = np.fromfile(filepath, dtype=dtype)
    expected_size = shape[0] * shape[1]
    if v.size != expected_size:
        raise ValueError(f"File size mismatch: expected {expected_size} elements for shape {shape}, got {v.size}.")
    return v.reshape(shape)