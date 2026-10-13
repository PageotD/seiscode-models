"""Precision management utilities for velocity models."""

from typing import Union

import numpy as np


def parse_dtype(dtype: Union[str, np.dtype] = np.float32) -> np.dtype:
    """Parse and validate the requested floating-point precision dtype.

    Parameters
    ----------
    dtype : str or np.dtype, optional
        Target precision, e.g., 'f32', 'float32', 'f64', 'float64',
        or np.float32, np.float64. Default is np.float32.

    Returns
    -------
    np.dtype
        Normalized NumPy floating-point dtype.

    Raises
    ------
    ValueError
        If the dtype is not supported or not a floating-point type.
    """
    if isinstance(dtype, str):
        dtype_lower = dtype.lower()

        if dtype_lower in ("f32", "float32"):
            return np.dtype(np.float32)
        elif dtype_lower in ("f64", "float64"):
            return np.dtype(np.float64)
        else:
            raise ValueError(
                f"Unsupported string dtype: '{dtype}'. Use 'f32' or 'f64'."
            )

    dt = np.dtype(dtype)

    if dt == np.dtype(np.float32) or dt == np.dtype(np.float64):
        return dt

    raise ValueError(
        f"Unsupported dtype: {dt}. Only float32 (f32) and float64 (f64) are allowed."
    )
