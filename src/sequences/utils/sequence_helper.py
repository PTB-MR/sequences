"""Helper functions for the creation of sequences."""

from typing import Literal

import numpy as np


def round_to_raster(value: float, raster_time: float, method: Literal['floor', 'round', 'ceil'] = 'round') -> float:
    """Round a value to the given raster time using the defined method.

    Parameters
    ----------
    value
        Value to be rounded.
    raster_time
        Raster time, e.g. gradient, rf or ADC raster time.
    method
        Rounding method. Options: "floor", "round", "ceil".

    Returns
    -------
    rounded_value
        Rounded value.
    """
    if method == 'floor':
        return raster_time * np.floor(value / raster_time)
    elif method == 'round':
        return raster_time * np.round(value / raster_time)
    elif method == 'ceil':
        return raster_time * np.ceil(value / raster_time)
    else:
        raise ValueError(f'Unknown rounding method: {method}. Expected: "floor", "round" or "ceil".')
