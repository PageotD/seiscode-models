# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to Semantic Versioning.

## [x.y.z] YYYY-MM-DD

### Added
- `core`: Add `load` and `copy` methods
- `core`: Add `vmin`, `vmax`, and `stats` methods
- `core`: Add `clip` and `smooth` methods

### Changed
- `models.geological.sinusoidal_interface`: Vectorisation of the function
- `models.geological.dipping_layers`: Add thickness of the first layer as parameter
- `models.geological.faulted_layers`: Bounds proportional to the number of layers
- `utils.io.export_model`: Add .npy if missing and manage C and F ordering
- move ruff configuration in `pyproject.toml`

### Fixed
- Version comes from pyproject.toml only

## [0.2.0] 2026-10-12

### Added
- Object-oriented interface for 2D subsurface velocity models (Phase 3).
  
## [0.1.0] 2026-10-12

### Added
- Initial project setup, folder structure, and CI/CD pipelines (Phase 0).
- Support for precision selection (`f32` / `f64`) in model generation (Phase 0).
- Core velocity primitives and validation with f32/f64 support (Phase 1).
- Model export functions for seismic solvers (Pahse 2).
