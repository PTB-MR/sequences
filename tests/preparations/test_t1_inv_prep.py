"""Tests for the adiabatic T1 preparation block."""

import pypulseq as pp
import pytest
from sequences.preparations.t1_inv_prep import add_t1_inv_prep


def test_add_t1_inv_prep_system_defaults_if_none(system_defaults):
    """Test if system defaults are used if no system limits are provided."""
    _, block_duration1, _ = add_t1_inv_prep(system=system_defaults)
    _, block_duration2, _ = add_t1_inv_prep(system=None)

    assert block_duration1 == block_duration2


@pytest.mark.parametrize(
    ('rf_duration', 'add_spoiler', 'spoiler_ramp_time', 'spoiler_flat_time'),
    [
        (10.24e-3, True, 6e-4, 8.4e-3),
        (20.00e-3, True, 6e-4, 8.4e-3),
        (10.24e-3, False, 6e-4, 8.4e-3),
        (10.24e-3, True, 1e-3, 10e-3),
    ],
    ids=['defaults', 'longer_pulse', 'no_spoiler', 'longer_spoiler'],
)
def test_add_t1_inv_prep_duration(system_defaults, rf_duration, add_spoiler, spoiler_ramp_time, spoiler_flat_time):
    """Ensure the default parameters are set correctly."""
    seq = pp.Sequence(system=system_defaults)

    seq, block_duration, _ = add_t1_inv_prep(
        seq=seq,
        system=system_defaults,
        rf_duration=rf_duration,
        add_spoiler=add_spoiler,
        spoiler_ramp_time=spoiler_ramp_time,
        spoiler_flat_time=spoiler_flat_time,
    )

    manual_time_calc = (
        system_defaults.rf_dead_time  # dead time before 180° inversion pulse
        + rf_duration  # half duration of 180° inversion pulse
        + system_defaults.rf_ringdown_time
    )
    if add_spoiler:
        manual_time_calc += spoiler_ramp_time + spoiler_flat_time + spoiler_ramp_time

    assert sum(seq.block_durations.values()) == block_duration
    assert block_duration == pytest.approx(manual_time_calc)
