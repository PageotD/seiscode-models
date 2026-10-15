# seiscode-models

[![CI](https://github.com/PageotD/seiscode-models/actions/workflows/ci.yml/badge.svg)](https://github.com/PageotD/seiscode-models/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/PageotD/seiscode-models/branch/main/graph/badge.svg)](https://codecov.io/gh/PageotD/seiscode-models)
[![PyPI - Version](https://img.shields.io/pypi/v/seiscode-models)](https://pypi.org/project/seiscode-models/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/seiscode-models)](https://pypi.org/project/seiscode-models/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23349342.svg)](https://doi.org/10.5281/zenodo.23349342)

A robust Python package for seismic modeling, simulation configurations, and waveform processing utilities. Designed with a clean API, strict type hinting, and seamless cloud-native CI/CD workflows.

---

## Features

- **Seismic Modeling Core:** Extensible Python framework for forward modeling and synthetic data generation.
- **High Performance:** Optimized numerical routines and data structures.
- **Modern Python Standards:** Fully type-hinted, tested across multiple Python versions (3.8–3.11), and linted with Ruff.
- **Automated CI/CD:** Continuous integration, automated semantic versioning, and publishing pipeline via GitHub Actions.

---

## Installation

You can install the latest stable version of `seiscode-models` directly from PyPI:

```bash
pip install seiscode-models

```

Or install it for local development from source:

```bash
git clone [https://github.com/PageotD/seiscode-models.git](https://github.com/PageotD/seiscode-models.git)
cd seiscode-models
pip install -e ".[dev]"

```

---

## Quick Start

Here is a quick example showing how to initialize a basic modeling configuration and run a simulation routine:

```python
from seiscode_models.config import ModelConfig
from seiscode_models.engine import SeismicEngine

# 1. Define the simulation configuration
config = ModelConfig(
    velocity_model="homogeneous",
    p_wave_velocity=3500.0,  # m/s
    s_wave_velocity=2000.0,  # m/s
    density=2200.0,          # kg/m^3
    frequency=15.0           # Hz
)

# 2. Initialize the engine and execute simulation
engine = SeismicEngine(config=config)
result = engine.run_forward_modeling()

print(f"Simulation completed successfully. Peak amplitude: {result.max_amplitude:.4f}")

```

For more advanced use cases, check out the [examples directory](https://www.google.com/search?q=./examples/) or the [CHANGELOG.md](https://www.google.com/search?q=./CHANGELOG.md).

---

## Running Tests

To run the test suite locally with `pytest`:

```bash
pytest tests/ -v --cov=seiscode_models

```

---

## Citation

If you use `seiscode-models` in your research or academic work, please cite it using the following Zenodo DOI:

> Pageot, D. (2026). *seiscode-models: A Python package for seismic modeling* (Version x.y.z). Zenodo. https://doi.org/10.5281/zenodo.23349342

---

## License

This project is open-source under the [MIT License](https://www.google.com/search?q=LICENSE).