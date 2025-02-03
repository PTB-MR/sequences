"""Helper functions for the creation of sequences."""

import numpy as np


def round_to_raster(value: float, raster_time: float) -> float:
    """Round a value to the closest larger multiple of a raster time.

    Parameters
    ----------
    value
        Value to be rounded.
    raster_time
        Raster time, e.g. gradient, rf or ADC raster time.

    Returns
    -------
    rounded_value
        Rounded value.
    """
    return raster_time * np.ceil(value / raster_time)
