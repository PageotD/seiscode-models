"""Tests for precision dtype utilities."""
import numpy as np
import pytest

from seiscode_models.utils.dtypes import parse_dtype


class TestParseDtype:
    def test_default_dtype(self):
        assert parse_dtype() == np.dtype(np.float32)

    def test_string_inputs(self):
        assert parse_dtype("f32") == np.dtype(np.float32)
        assert parse_dtype("float32") == np.dtype(np.float32)
        assert parse_dtype("f64") == np.dtype(np.float64)
        assert parse_dtype("float64") == np.dtype(np.float64)

    def test_numpy_dtype_inputs(self):
        assert parse_dtype(np.float32) == np.dtype(np.float32)
        assert parse_dtype(np.float64) == np.dtype(np.float64)

    def test_invalid_inputs(self):
        with pytest.raises(ValueError, match="Unsupported string dtype"):
            parse_dtype("int32")
        with pytest.raises(ValueError, match="Unsupported dtype"):
            parse_dtype(np.int32)
