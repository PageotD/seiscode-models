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
