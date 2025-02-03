"""Tests for sequence helper functions."""

import pytest
from sequences.utils import round_to_raster


@pytest.mark.parametrize(
    'value, raster_time, expected',
    [
        (+1.0e-6, 1e-6, 1e-6),
        (+1.1e-6, 1e-6, 2e-6),
        (-1.1e-6, 1e-6, -1e-6),
        (+1.0e-9, 1e-6, 1e-6),
        (0.0, 1e-6, 0.0),  # Zero case
        (2.0e-6, 1e-6, 2.0e-6),  # Already a multiple
        (-2.0e-6, 1e-6, -2.0e-6),  # Negative multiple
        (-1.9e-6, 1e-6, -1e-6),  # Negative rounding up
        (1e6, 1e-6, 1e6),  # Large value
        (1e-12, 1e-6, 1e-6),  # Very small value
        (2.5e-6, 1e-6, 3e-6),  # Non-integer multiple
        (1.0000000001e-6, 1e-6, 2e-6),  # Floating-point precision up
        (0.9999999999e-6, 1e-6, 1e-6),  # Floating-point precision down
    ],
)
def test_round_to_raster(value: float, raster_time: float, expected: float):
    """Test the rounding to raster function."""
    assert round_to_raster(value, raster_time) == expected
