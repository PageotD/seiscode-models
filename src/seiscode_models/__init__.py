"""SeisCode Models: Synthetic subsurface velocity model generators."""
from .models.primitives import (
    gradient_background,
    add_gaussian_anomaly,
    add_fault_offset,
)
from .utils.dtypes import parse_dtype
from .utils.validation import validate_model

__version__ = "0.1.0"

__all__ = [
    "gradient_background",
    "add_gaussian_anomaly",
    "add_fault_offset",
    "parse_dtype",
    "validate_model",
]