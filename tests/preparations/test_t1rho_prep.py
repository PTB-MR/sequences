"""Tests for the T1 rho preparation block."""

import pypulseq as pp
import pytest
from sequences.preparations.t1rho_prep import add_t1rho_prep


def test_add_t1rho_prep_raise_error_no_rf_dead_time(system_defaults):
    """Test if a ValueError is raised if rf_dead_time is not set."""
    system_defaults.rf_dead_time = None
    with pytest.raises(ValueError, match='rf_dead_time must be provided'):
        add_t1rho_prep(system=system_defaults)


def test_add_t1rho_prep_system_defaults_if_none(system_defaults):
    """Test if system defaults are used if no system limits are provided."""
    _, block_duration1 = add_t1rho_prep(system=system_defaults)
    _, block_duration2 = add_t1rho_prep(system=None)

    assert block_duration1 == block_duration2


@pytest.mark.parametrize(
    (
        'duration_90',
        'spin_lock_time',
        'spin_lock_amplitude',
        'add_spoiler',
        'spoiler_ramp_time',
        'spoiler_flat_time',
    ),
    [
        (2e-3, 20e-3, 5e-6, True, 6e-4, 6e-3),
        (2e-3, 80e-3, 5e-6, True, 6e-4, 6e-3),
        (2e-3, 20e-3, 3e-6, True, 6e-4, 6e-3),
        (2e-3, 80e-3, 5e-6, False, 0, 0),
        (2e-3, 80e-3, 5e-6, True, 6e-4, 12e-3),
    ],
    ids=['defaults', 'longer_spin_lock_time', 'smaller_spin_lock_amplitude', 'no_spoiler', 'longer_spoiler'],
)
def test_add_t1rho_prep_duration(
    system_defaults, duration_90, spin_lock_time, spin_lock_amplitude, add_spoiler, spoiler_ramp_time, spoiler_flat_time
):
    """Ensure the default parameters are set correctly."""
    seq = pp.Sequence(system=system_defaults)

    seq, block_duration = add_t1rho_prep(
        seq=seq,
        system=system_defaults,
        duration_90=duration_90,
        spin_lock_time=spin_lock_time,
        spin_lock_amplitude=spin_lock_amplitude,
        add_spoiler=add_spoiler,
        spoiler_ramp_time=spoiler_ramp_time,
        spoiler_flat_time=spoiler_flat_time,
    )

    manual_time_calc = (
        system_defaults.rf_dead_time  # dead time before 90° pulse
        + duration_90
        + system_defaults.rf_dead_time  # dead time before spin-lock pulse
        + spin_lock_time
        + system_defaults.rf_dead_time  # dead time before 90° pulse
        + duration_90
    )
    if add_spoiler:
        manual_time_calc += 2 * spoiler_ramp_time + spoiler_flat_time

    assert sum(seq.block_durations.values()) == block_duration
    assert block_duration == pytest.approx(manual_time_calc)
