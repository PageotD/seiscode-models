"""Tests for model I/O export and import utilities."""

import numpy as np
import pytest

import seiscode_models as sm


class TestModelIO:
    def test_export_import_bin(self, tmp_path):
        v = sm.gradient_background(40, 20, dtype="f32")
        file_path = tmp_path / "model.bin"

        sm.export_model(v, file_path, format="bin")
        assert file_path.exists()

        v_loaded = sm.load_model(file_path, shape=(40, 20), dtype=np.float32)
        assert v_loaded.shape == (40, 20)
        assert v_loaded.dtype == np.float32
        assert np.allclose(v, v_loaded)

    def test_fortran_order_distinction(self, tmp_path):
        v = np.arange(12, dtype=np.float32).reshape(3, 4)
        c_path = tmp_path / "c_order.bin"
        f_path = tmp_path / "f_order.bin"

        sm.export_model(v, c_path, format="bin", order="C")
        sm.export_model(v, f_path, format="bin", order="F")

        c_bytes = c_path.read_bytes()
        f_bytes = f_path.read_bytes()
        # Binary files must be different on non-symmetric multi-element arrays
        assert c_bytes != f_bytes

        # Reading back with corresponding order recovers identical arrays
        v_c = sm.load_model(c_path, shape=(3, 4), dtype=np.float32, order="C")
        v_f = sm.load_model(f_path, shape=(3, 4), dtype=np.float32, order="F")
        assert np.array_equal(v, v_c)
        assert np.array_equal(v, v_f)

    def test_invalid_order(self, tmp_path):
        v = sm.gradient_background(10, 10)
        with pytest.raises(ValueError, match="Unsupported memory order"):
            sm.export_model(v, tmp_path / "model.bin", order="Z")
        with pytest.raises(ValueError, match="Unsupported memory order"):
            sm.load_model(tmp_path / "model.bin", shape=(10, 10), order="Z")

    def test_export_import_npy(self, tmp_path):
        v = sm.salt_dome(30, 30, dtype="f64")
        file_path = tmp_path / "model.npy"

        sm.export_model(v, file_path, format="npy")
        assert file_path.exists()

        v_loaded = sm.load_model(file_path, shape=(30, 30))
        assert v_loaded.dtype == np.float64
        assert np.allclose(v, v_loaded)

    def test_invalid_format(self, tmp_path):
        v = sm.gradient_background(10, 10)
        with pytest.raises(ValueError, match="Unsupported export format"):
            sm.export_model(v, tmp_path / "model.xyz", format="xyz")
