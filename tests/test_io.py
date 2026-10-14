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
