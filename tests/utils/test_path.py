import sys
import os
import pytest
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.utils.path import verify_saving_path


def test_invalid_type_raises_type_error():
    with pytest.raises(TypeError, match="Expected str or pathlib.Path"):
        verify_saving_path(12345)