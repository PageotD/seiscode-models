"""Tests for the object-oriented VelocityModel interface."""

import numpy as np
import pytest

import seiscode_models as sm
from seiscode_models.core import VelocityModel


class TestVelocityModelOOP:
    def test_factory_and_properties(self):
        model = sm.VelocityModel.gradient(nx=100, nz=50, dx=2.0, dz=2.0, dtype="f32")
        assert model.nx == 100
        assert model.nz == 50
        assert model.dx == 2.0
        assert model.dz == 2.0
        assert model.dtype == np.float32
        assert model.data.shape == (100, 50)

    def test_method_chaining(self):
        model = (
            sm.VelocityModel.gradient(nx=80, nz=40, dtype="f64")
            .add_anomaly(cx=40, cz=20, sigma_x=5.0, sigma_z=5.0, dv=500.0)
            .add_fault(ix_fault=50, throw=5)
        )
        assert isinstance(model, sm.VelocityModel)
        assert model.dtype == np.float64
        assert model.data.shape == (80, 40)

    def test_export_from_model(self, tmp_path):
        model = sm.VelocityModel.salt_dome(30, 30)
        file_path = tmp_path / "oop_model.bin"
        model.export(file_path, format="bin")
        assert file_path.exists()

    def test_version_matches(self):
        assert isinstance(sm.__version__, str)
        assert sm.__version__.count(".") == 2

    def test_copy_is_independent(self):
        m = sm.VelocityModel.gradient()
        c = m.copy()
        c.add_anomaly(100, 25, 10, 5, 500.0)
        assert m != c
        assert np.array_equal(m.data, sm.VelocityModel.gradient().data)

    def test_stats(self):
        m = sm.VelocityModel.gradient()
        s = m.stats()
        assert s["shape"] == (m.nx, m.nz)
        assert s["vmin"] <= s["vmean"] <= s["vmax"]
        assert s["dtype"] == "float32"

    def test_smooth_preserves_shape_dtype(self):
        m = VelocityModel.checkerboard()
        m2 = m.copy().smooth(k=5)
        assert m2.data.shape == m.data.shape
        assert m2.dtype == m.dtype

    def test_smooth_invalid_kernel(self):
        m = sm.VelocityModel.gradient()
        with pytest.raises(ValueError):
            m.smooth(k=4)

    def test_clip(self):
        m = sm.VelocityModel.gradient(v0=100.0, v1=9000.0)

        m.clip(vmin=500.0, vmax=8000.0)

        assert m.vmin >= 500.0
        assert m.vmax <= 8000.0

    def test_export_load_roundtrip(self, tmp_path):
        m = sm.VelocityModel.gradient()
        f = tmp_path / "model.bin"
        m.export(f, format="bin", order="F")
        m2 = sm.VelocityModel.load(f, shape=(m.nx, m.nz), dx=1.0, dz=1.0)
        assert np.array_equal(m.data, m2.data)

    def test_repr(self):
        assert "VelocityModel" in repr(sm.VelocityModel.gradient())
