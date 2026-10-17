"""Tests for geological models and precision behavior."""

import numpy as np
import pytest

import seiscode_models as sm


class TestGeologicalModels:
    @pytest.mark.parametrize(
        "model_func",
        [
            sm.dipping_layers,
            sm.sinusoidal_interface,
            sm.faulted_layers,
            sm.checkerboard,
            sm.salt_dome,
            sm.low_velocity_layer,
            sm.heterogeneous_anomaly_model,
        ],
    )
    def test_model_generation_and_dtype(self, model_func):
        # Test f32
        v_f32 = model_func(nx=50, nz=30, dtype="f32")
        assert v_f32.shape == (50, 30)
        assert v_f32.dtype == np.float32
        sm.validate_model(v_f32)

        # Test f64
        v_f64 = model_func(nx=50, nz=30, dtype="f64")
        assert v_f64.shape == (50, 30)
        assert v_f64.dtype == np.float64
        sm.validate_model(v_f64)

    def test_physical_parameters(self):
        # Test low_velocity_layer with z0 and thickness in meters
        v_lvl = sm.low_velocity_layer(nx=40, nz=40, dx=5.0, dz=2.0, z0=20.0, thickness=10.0, v_slow=1000.0)
        # z0=20m -> row 10, thickness=10m -> rows 10 to 15
        assert v_lvl[0, 10] == 1000.0
        assert v_lvl[0, 14] == 1000.0
        assert v_lvl[0, 5] != 1000.0

        # Test checkerboard with cell_x and cell_z
        v_cb = sm.checkerboard(nx=40, nz=40, dx=2.0, dz=2.0, cell_x=20.0, cell_z=10.0)
        assert v_cb.shape == (40, 40)
        sm.validate_model(v_cb, dx=2.0, dz=2.0)

