"""SeisCode Models: Synthetic subsurface velocity model generators."""

from .models.primitives import (
    gradient_background,
    add_gaussian_anomaly,
    add_fault_offset,
)
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
from .utils.io import export_model, load_model
from .utils.visualization import quick_view
from .core import VelocityModel

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:  # Python3.8
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("seiscode-models")
except PackageNotFoundError:
    raise RuntimeError(
        "seiscode-models package is not installed. Please install it to use this module."
    )


__all__ = [
    "VelocityModel",
    "gradient_background",
    "add_gaussian_anomaly",
    "add_fault_offset",
    "parse_dtype",
    "validate_model",
    "dipping_layers",
    "sinusoidal_interface",
    "faulted_layers",
    "checkerboard",
    "salt_dome",
    "low_velocity_layer",
    "heterogeneous_anomaly_model",
    "export_model",
    "load_model",
    "quick_view",
]
