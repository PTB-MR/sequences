"""Tests for the MLEV-4 type T2 preparation block."""

from copy import deepcopy

import pypulseq as pp
import pytest
from sequences.preparations.t2prep import add_composite_refocusing_block
from sequences.preparations.t2prep import add_t2prep


def test_add_composite_refocusing_block_raise_error_no_rf_dead_time(system_defaults):
    """Test if a ValueError is raised if rf_dead_time is not set."""
    system_defaults.rf_dead_time = None

    seq = pp.Sequence(system=system_defaults)
    with pytest.raises(ValueError, match='rf_dead_time must be provided'):
        add_composite_refocusing_block(seq=seq, system=system_defaults, duration_180=2e-3)


@pytest.mark.parametrize(
    ('duration_180', 'rf_dead_time'),
    [(2e-3, 100e-6), (2e-3, 200e-6), (4e-3, 100e-6), (6e-3, 200e-6)],
)
def test_add_composite_refocusing_block_duration(system_defaults, duration_180, rf_dead_time):
    """Ensure the default parameters are set correctly."""
    system_defaults.rf_dead_time = rf_dead_time
    seq = pp.Sequence(system=system_defaults)

    seq, total_dur, _ = add_composite_refocusing_block(seq=seq, system=system_defaults, duration_180=duration_180)

    assert total_dur == sum(seq.block_durations.values())
    assert total_dur == pytest.approx(2 * duration_180 + 3 * rf_dead_time)


@pytest.mark.parametrize('rf_ringdown_time', [0, 30e-6, 100e-6, 1])
def test_add_composite_refocusing_block_no_ringdown_dependency(system_defaults, rf_ringdown_time):
    """Test if the block duration is not dependent on the ringdown time."""
    system1 = system_defaults
    system2 = deepcopy(system1)
    system2.rf_ringdown_time = rf_ringdown_time

    seq = pp.Sequence(system=system_defaults)

    _, total_dur1, _ = add_composite_refocusing_block(seq=seq, system=system1, duration_180=2e-3)
    _, total_dur2, _ = add_composite_refocusing_block(seq=seq, system=system2, duration_180=2e-3)

    assert total_dur1 == total_dur2


def test_add_t2prep_raise_error_no_rf_dead_time(system_defaults):
    """Test if a ValueError is raised if rf_dead_time is not set."""
    system_defaults.rf_dead_time = None
    with pytest.raises(ValueError, match='rf_dead_time must be provided'):
        add_t2prep(system=system_defaults)


@pytest.mark.parametrize(('echo_time', 'duration_180'), [(0.01, 1e-3), (0.011, 1e-3), (0.015, 1e-3), (0.04, 4e-3)])
def test_add_t2prep_fail_on_short_echo_time(system_defaults, echo_time, duration_180):
    """Test if function raises an error when desired echo time is too short for given pulse duration."""
    seq = pp.Sequence(system=system_defaults)
    with pytest.raises(ValueError, match='Desired echo time'):
        add_t2prep(seq=seq, system=system_defaults, echo_time=echo_time, duration_180=duration_180)


def test_add_t2prep_system_defaults_if_none(system_defaults):
    """Test if system defaults are used if no system limits are provided."""
    _, block_duration1 = add_t2prep(system=system_defaults)
    _, block_duration2 = add_t2prep(system=None)

    assert block_duration1 == block_duration2


@pytest.mark.parametrize(
    ('echo_time', 'duration_180', 'add_spoiler', 'spoiler_ramp_time', 'spoiler_flat_time'),
    [
        (0.1, 1e-3, True, 6e-4, 6e-3),
        (0.2, 1e-3, True, 6e-4, 6e-3),
        (0.1, 4e-3, True, 6e-4, 6e-3),
        (0.1, 1e-3, False, 6e-4, 6e-3),
        (0.1, 1e-3, True, 1e-3, 10e-3),
    ],
    ids=['defaults', 'longer_te', 'longer_pulses', 'no_spoiler', 'longer_spoiler'],
)
def test_add_t2prep_duration(
    system_defaults, echo_time, duration_180, add_spoiler, spoiler_ramp_time, spoiler_flat_time
):
    """Ensure the default parameters are set correctly."""
    seq = pp.Sequence(system=system_defaults)

    seq, block_duration = add_t2prep(
        seq=seq,
        system=system_defaults,
        echo_time=echo_time,
        duration_180=duration_180,
        add_spoiler=add_spoiler,
        spoiler_ramp_time=spoiler_ramp_time,
        spoiler_flat_time=spoiler_flat_time,
    )

    manual_time_calc = (
        system_defaults.rf_dead_time
        + duration_180 / 4  # half duration of 90° excitation pulse
        + echo_time  # echo time
        + duration_180 / 2 * 3 / 2  # half duration of 270° pulse
        + system_defaults.rf_dead_time  # dead time before 360° pulse
        + duration_180 * 2  # duration of 360° pulse
    )
    if add_spoiler:
        manual_time_calc += 2 * spoiler_ramp_time + spoiler_flat_time

    assert sum(seq.block_durations.values()) == block_duration
    assert block_duration == pytest.approx(manual_time_calc)
