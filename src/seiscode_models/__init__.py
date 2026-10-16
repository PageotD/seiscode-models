"""SeisCode Models: Synthetic subsurface velocity model generators."""

from .core import VelocityModel
from .models.geological import (
    checkerboard,
    dipping_layers,
    faulted_layers,
    heterogeneous_anomaly_model,
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
from .utils.dtypes import parse_dtype
from .utils.io import export_model, load_model
from .utils.validation import validate_model
from .utils.visualization import quick_view

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # Python3.8
    from importlib_metadata import PackageNotFoundError, version

try:
    __version__ = version("seiscode-models")
except PackageNotFoundError as err:
    raise RuntimeError(
        "seiscode-models package is not installed. Please install it to use this module."
    ) from err


__all__ = [
    "VelocityModel",
    "gradient_background",
    "add_gaussian_anomaly",
    "add_fault_offset",
    "add_elliptical_inclusion",
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
