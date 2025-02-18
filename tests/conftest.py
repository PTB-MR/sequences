"""PyTest fixtures for the MRseq package."""

from copy import deepcopy

import pytest
from mrseq.utils import sys_defaults


@pytest.fixture(scope='function')
def system_defaults():
    """System defaults for sequence generation."""
    return deepcopy(sys_defaults)
