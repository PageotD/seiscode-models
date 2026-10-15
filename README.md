# SeisCode Models

[![PyPI - Version](https://img.shields.io/pypi/v/seiscode-models)](https://pypi.org/project/seiscode-models/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/seiscode-models)](https://pypi.org/project/seiscode-models/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**SeisCode Models** is a lightweight Python library for generating, manipulating, visualizing, and exporting synthetic 2D subsurface velocity models.

It is designed for seismic modeling, seismic inversion experiments, resolution tests, and the preparation of velocity fields for numerical solvers, including eikonal solvers and finite-difference wave propagation codes.

Built on NumPy, the library provides geological model generators and an object-oriented interface for working with velocity fields.

## Features

- **Geological model generation:** Create dipping layers, faulted strata, salt domes, sinusoidal interfaces, and low-velocity layers.
- **Synthetic anomalies:** Add Gaussian velocity anomalies to simulate heterogeneous subsurface structures.
- **Customizable velocity fields:** Generate vertical velocity gradients and checkerboard patterns for resolution tests.
- **Object-oriented API:** Build and manipulate models through the `VelocityModel` class.
- **Model processing:** Apply fault offsets, clip velocity values, and smooth velocity fields.
- **Numerical precision:** Work with `float32` or `float64` velocity arrays.
- **Visualization:** Display velocity models with the built-in plotting utility.
- **Data export and loading:** Save models as raw binary or NumPy files for use in external applications.
- **Lightweight dependencies:** Requires NumPy and Matplotlib.

## Installation

Install the latest published version from PyPI:

```bash
pip install seiscode-models
```

To install the development version from the repository:

```bash
git clone https://github.com/PageotD/seiscode-models.git
cd seiscode-models
pip install -e ".[dev]"
```

The development dependencies include pytest, pytest-cov, Ruff, and Python packaging tools.

## Quick start

Generate a velocity model with a linear velocity gradient, add a Gaussian anomaly, inspect its properties, and visualize the result.

```python
from seiscode_models import VelocityModel

# Generate a 2D velocity model.
model = VelocityModel.gradient(
    nx=201,
    nz=101,
    dx=10.0,
    dz=10.0,
    v0=1500.0,
    v1=4500.0,
)

# Add a positive Gaussian velocity anomaly.
model.add_anomaly(
    cx=100,
    cz=50,
    sigma_x=15,
    sigma_z=10,
    dv=1000.0,
)

# Inspect model properties.
print(model)
print(model.stats())

# Visualize the velocity field.
model.plot(title="Velocity Model with Gaussian Anomaly")

# Export the model as a NumPy file.
model.export("velocity_model.npy", format="npy")
```

The resulting velocity field is a NumPy array with shape `(nx, nz)`. Velocity values are expressed in meters per second, while `dx` and `dz` represent grid spacing in meters.

The anomaly coordinates and standard deviations are specified in grid cells, not physical distances.

## Generating geological models

`VelocityModel` provides class methods for creating several common synthetic subsurface structures.

All examples below use the public API.

### Linear velocity gradient

Create a model in which velocity increases linearly with depth.

```python
from seiscode_models import VelocityModel

model = VelocityModel.gradient(
    nx=201,
    nz=101,
    dx=10.0,
    dz=10.0,
    v0=1500.0,
    v1=4500.0,
)

model.plot(title="Linear Velocity Gradient")
```

Parameters:

- `nx`, `nz`: Number of grid cells along the horizontal and vertical axes.
- `dx`, `dz`: Grid spacing in meters.
- `v0`: Velocity at the top of the model, in m/s.
- `v1`: Velocity at the bottom of the model, in m/s.
- `dtype`: NumPy-compatible data type; defaults to `numpy.float32`.

### Dipping layers

Generate a stratified model with inclined geological interfaces.

```python
model = VelocityModel.dipping_layers(
    nx=201,
    nz=101,
    dx=10.0,
    dz=10.0,
    v=(1800.0, 2400.0, 3200.0, 4200.0),
    dip_deg=8.0,
)

model.plot(title="Dipping Geological Layers")
```

The `v` parameter specifies the velocity values associated with the layers, while `dip_deg` controls their dip angle.

### Faulted layers

Create a stratified velocity model with a vertical fault offset.

```python
model = VelocityModel.faulted_layers(
    nx=201,
    nz=101,
    dx=10.0,
    dz=10.0,
    ix_fault=120,
    throw=8,
    v=(1800.0, 2400.0, 3200.0, 4200.0),
)

model.plot(title="Faulted Geological Layers")
```

- `ix_fault`: Horizontal grid index of the fault.
- `throw`: Vertical displacement in grid cells.
- `v`: Sequence of layer velocities in m/s.

### Salt dome

Generate a synthetic salt structure embedded in sedimentary material.

```python
model = VelocityModel.salt_dome(
    nx=201,
    nz=101,
    dx=10.0,
    dz=10.0,
    cx=140,
    width=40.0,
    v_salt=4500.0,
    v_sed=2500.0,
)

model.plot(title="Salt Dome")
```

The dome's horizontal position is controlled by `cx`, and its approximate width by `width`. The velocities of the salt and surrounding sediments are configurable.

### Sinusoidal interface

Create a two-layer model separated by a sinusoidal geological interface.

```python
model = VelocityModel.sinusoidal_interface(
    nx=201,
    nz=101,
    dx=10.0,
    dz=10.0,
    period=100.0,
    amplitude=8.0,
    v_top=2000.0,
    v_bottom=3500.0,
)

model.plot(title="Sinusoidal Interface")
```

### Checkerboard model

Generate alternating velocity anomalies for resolution tests and inversion benchmarks.

```python
model = VelocityModel.checkerboard(
    nx=201,
    nz=101,
    dx=10.0,
    dz=10.0,
    cell=20,
    v0=3000.0,
    dv=600.0,
)

model.plot(title="Checkerboard Resolution Model")
```

The `cell` parameter controls the checkerboard block size in grid cells. The two alternating velocity values are `v0 + dv` and `v0 - dv`.

### Low-velocity layer

Introduce a buried low-velocity channel into a background velocity gradient.

```python
model = VelocityModel.low_velocity_layer(
    nx=201,
    nz=101,
    dx=10.0,
    dz=10.0,
    k0=30,
    thickness=8,
    v_slow=1200.0,
)

model.plot(title="Buried Low-Velocity Layer")
```

- `k0`: Starting vertical grid index of the low-velocity layer.
- `thickness`: Layer thickness in grid cells.
- `v_slow`: Velocity within the layer, in m/s.

### Heterogeneous anomaly model

Generate a more complex synthetic model containing multiple velocity anomalies and smoothed random perturbations.

```python
model = VelocityModel.heterogeneous_anomaly_model(
    nx=384,
    nz=122,
    dx=10.0,
    dz=10.0,
    seed=7,
)

model.plot(title="Heterogeneous Velocity Model")
```

The `seed` parameter controls the random perturbations, allowing the same synthetic realization to be reproduced.

## Modifying existing models

The `VelocityModel` class provides methods for modifying velocity fields in place. These methods return the model instance, allowing method chaining.

### Add a Gaussian anomaly

Use `add_anomaly()` to introduce a localized positive or negative velocity perturbation.

```python
model = VelocityModel.gradient(nx=201, nz=101)

model.add_anomaly(
    cx=100,
    cz=50,
    sigma_x=15,
    sigma_z=10,
    dv=-500.0,
)
```

Parameters:

- `cx`, `cz`: Anomaly center in grid cells.
- `sigma_x`, `sigma_z`: Gaussian standard deviations in grid cells.
- `dv`: Velocity perturbation at the anomaly center, in m/s.
- `vmin`, `vmax`: Minimum and maximum allowed velocities.

### Apply a fault offset

```python
model.add_fault(
    ix_fault=100,
    throw=8,
)
```

This applies a vertical displacement to the velocity field across the specified fault index.

### Clip velocity values

Constrain the velocity field to a specified range.

```python
model.clip(
    vmin=500.0,
    vmax=6000.0,
)
```

### Smooth the velocity field

Apply a moving-average filter to reduce small-scale variations.

```python
model.smooth(k=5)
```

The smoothing kernel size `k` must be an odd integer greater than or equal to 3. The array is edge-padded during filtering.

### Chain operations

Since the modification methods return the model instance, they can be combined.

```python
model = (
    VelocityModel.dipping_layers(
        nx=201,
        nz=101,
        dx=10.0,
        dz=10.0,
    )
    .add_anomaly(
        cx=100,
        cz=50,
        sigma_x=12,
        sigma_z=8,
        dv=800.0,
    )
    .clip(vmin=500.0, vmax=6000.0)
    .smooth(k=3)
)

model.plot(title="Processed Velocity Model")
```

These operations modify the existing model rather than creating independent copies.

Use `copy()` when an independent model is needed.

## Accessing model properties

A `VelocityModel` instance exposes its underlying velocity array and grid metadata.

```python
model = VelocityModel.gradient(
    nx=201,
    nz=101,
    dx=10.0,
    dz=10.0,
)

print("Shape:", model.data.shape)
print("Horizontal cells:", model.nx)
print("Vertical cells:", model.nz)
print("Horizontal spacing:", model.dx)
print("Vertical spacing:", model.dz)
print("Data type:", model.dtype)
print("Minimum velocity:", model.vmin)
print("Maximum velocity:", model.vmax)
```

The `stats()` method returns a dictionary containing summary statistics.

```python
statistics = model.stats()

for name, value in statistics.items():
    print(f"{name}: {value}")
```

The returned dictionary contains:

| Key | Description |
|---|---|
| `shape` | Velocity array dimensions `(nx, nz)` |
| `dx` | Horizontal grid spacing in meters |
| `dz` | Vertical grid spacing in meters |
| `dtype` | NumPy data type |
| `vmin` | Minimum velocity in m/s |
| `vmax` | Maximum velocity in m/s |
| `vmean` | Mean velocity in m/s |
| `vstd` | Standard deviation of velocity in m/s |

## Exporting and loading models

Velocity fields can be exported for use with external seismic modeling and inversion software.

### Export a model

```python
from seiscode_models import VelocityModel

model = VelocityModel.gradient(
    nx=201,
    nz=101,
    dx=10.0,
    dz=10.0,
)

# NumPy format: preserves array shape and data type.
model.export("velocity_model.npy", format="npy")

# Raw binary format.
model.export("velocity_model.bin", format="bin")

# Raw binary in Fortran memory order.
model.export(
    "velocity_model_fortran.bin",
    format="bin",
    order="F",
)
```

Supported export formats:

| Format | Description |
|---|---|
| `npy` | NumPy binary format, preserving array shape and data type |
| `bin` | Raw binary velocity data |
| `raw` | Alias for raw binary export |

For raw binary files, the `order` parameter controls the memory layout:

- `C`: C-contiguous, row-major ordering.
- `F`: Fortran-contiguous, column-major ordering.

Choose the layout expected by your numerical solver. Raw binary files do not contain array dimensions or grid-spacing metadata, so these must be tracked separately.

### Load a model

The `VelocityModel.load()` class method can load a previously exported model.

For NumPy files:

```python
model = VelocityModel.load(
    "velocity_model.npy",
    dx=10.0,
    dz=10.0,
)
```

For raw binary files, provide the expected array shape and data type:

```python
model = VelocityModel.load(
    "velocity_model.bin",
    shape=(201, 101),
    dtype="float32",
    dx=10.0,
    dz=10.0,
)
```

The shape must match the dimensions used when exporting the model, and the data type must match the binary representation.

## Working directly with NumPy

`VelocityModel` wraps a NumPy array, accessible through the `data` property.

```python
import numpy as np
from seiscode_models import VelocityModel

model = VelocityModel.gradient(nx=201, nz=101)

velocity = model.data

print(type(velocity))
print(velocity.shape)
print(velocity.dtype)

# Example: compute the mean velocity along the vertical axis.
mean_velocity_by_column = np.mean(velocity, axis=1)
```

The array uses the convention `(nx, nz)`, with the horizontal coordinate first and the vertical coordinate second.

This convention should be considered when passing the velocity field to external numerical solvers or plotting utilities.

## Public API reference

The package exposes its principal classes and functions directly from `seiscode_models`.

### Main class

| API | Description |
|---|---|
| `VelocityModel` | Represents a 2D subsurface velocity model |

### Model constructors

| API | Description |
|---|---|
| `VelocityModel.gradient()` | Linear velocity gradient with depth |
| `VelocityModel.dipping_layers()` | Inclined geological layers |
| `VelocityModel.faulted_layers()` | Stratified model with a fault offset |
| `VelocityModel.salt_dome()` | Salt dome embedded in sediments |
| `VelocityModel.sinusoidal_interface()` | Two layers separated by a sinusoidal interface |
| `VelocityModel.checkerboard()` | Checkerboard pattern for resolution tests |
| `VelocityModel.low_velocity_layer()` | Buried low-velocity layer |
| `VelocityModel.heterogeneous_anomaly_model()` | Heterogeneous model with synthetic anomalies |

### Model operations

| API | Description |
|---|---|
| `model.add_anomaly()` | Add a Gaussian velocity anomaly |
| `model.add_fault()` | Apply a fault offset |
| `model.clip()` | Clip velocities to specified bounds |
| `model.smooth()` | Smooth the velocity field |
| `model.copy()` | Create an independent copy of the model |
| `model.plot()` | Visualize the velocity field |
| `model.export()` | Export the velocity field to a file |
| `VelocityModel.load()` | Load a model from a supported file |
| `model.stats()` | Return summary statistics |

### Additional public functions

The following functions are also available from the package's top-level namespace:

| Function | Description |
|---|---|
| `gradient_background()` | Generate a linear background velocity field |
| `add_gaussian_anomaly()` | Add a Gaussian anomaly to a NumPy velocity array |
| `add_fault_offset()` | Apply a fault displacement to a NumPy velocity array |
| `dipping_layers()` | Generate dipping layers |
| `sinusoidal_interface()` | Generate a sinusoidal interface |
| `faulted_layers()` | Generate faulted geological layers |
| `checkerboard()` | Generate a checkerboard model |
| `salt_dome()` | Generate a salt dome |
| `low_velocity_layer()` | Generate a buried low-velocity layer |
| `heterogeneous_anomaly_model()` | Generate a heterogeneous synthetic model |
| `parse_dtype()` | Normalize supported data type specifications |
| `validate_model()` | Validate a velocity model |
| `export_model()` | Export a NumPy velocity array |
| `load_model()` | Load velocity data from a file |
| `quick_view()` | Visualize a NumPy velocity array |

For example, the lower-level functions can be used without constructing a `VelocityModel` instance:

```python
from seiscode_models import gradient_background

velocity = gradient_background(
    nx=201,
    nz=101,
    v0=1500.0,
    v1=4500.0,
)
```

These functions are useful when working directly with NumPy arrays or integrating model generation into an existing scientific workflow.

## Running tests

To install the development dependencies:

```bash
pip install -e ".[dev]"
```

Run the test suite with pytest:

```bash
pytest
```

To run the tests with coverage reporting:

```bash
pytest --cov=seiscode_models --cov-report=term-missing
```

## Requirements

- Python 3.8 or later
- NumPy 1.21 or later
- Matplotlib 3.5 or later

## Contributing

Contributions, bug reports, and suggestions are welcome.

When contributing:

1. Keep examples consistent with the actual public API.
2. Add or update tests for behavioral changes.
3. Run the test suite before submitting a pull request.
4. Follow the project's existing code style.

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use SeisCode Models in academic research, please cite the software using the project's [Zenodo record](https://doi.org/10.5281/zenodo.23349342).

For the recommended citation format and version-specific metadata, consult the Zenodo record directly.