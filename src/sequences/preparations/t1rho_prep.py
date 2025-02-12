"""Spin-lock T1 rho preparation block."""

import pypulseq as pp
import numpy as np
from sequences.utils import sys_defaults
from sequences.utils.constants import GYROMAGNETIC_RATIO_PROTON


def add_t1rho_prep(
    seq: pp.Sequence | None = None,
    system: pp.Opts | None = None,
    duration_90: float = 10.24e-3,
    spin_lock_time: float = 10e-3,
    spin_lock_amplitude: float = 5.0e-6,
    add_spoiler: bool = True,
    spoiler_ramp_time: float = 6e-4,
    spoiler_flat_time: float = 8.4e-3,
) -> tuple[pp.Sequence, float, float]:
    """Add spin-lock T1 rho preparation block to a sequence.
    
    The spin-lock block consists of a 90° pulse, a spin-lock pulse with a certain duration (spin lock time) and a 90°
    pulse. Optionally, spoiler gradients can be added after the spin-lock pulse.

    Parameters
    ----------
    seq
        PyPulseq Sequence object.
    system
        PyPulseq system limit object.
    duration_90
        Duration of the 90° pulses (in seconds).
    spin_lock_time
        Duration of the spin-lock pulse (in seconds).
    spin_lock_amplitude
        Ampli   tude of the spin-lock pulse (in T).
    add_spoiler
        Toggles addition of spoiler gradients after the spin lock pulse.
    spoiler_ramp_time
        Duration of gradient spoiler ramps (in seconds).
    spoiler_flat_time
        Duration of gradient spoiler plateau (in seconds).

    Returns
    -------
    seq
        PyPulseq Sequence object.
    block_duration
        Total duration of the T1 rho preparation block (in seconds).
    time_of_spoiler
        Time duration of spoiler gradient (in seconds).
    """
    # set system to default if not provided
    if system is None:
        system = sys_defaults

    if seq is None:
        seq = pp.Sequence(system=system)

    # get current duration of sequence before adding T1 preparation block
    time_start = sum(seq.block_durations.values())
    
    # add 90° pulse
    rf_pre = pp.make_sinc_pulse(np.pi / 2, duration=duration_90, phase_offset=np.pi / 2, system=system)
    seq.add_block(rf_pre)
    
    # spin-lock pulse
    # calculate flip angle of spin-lock block pulse, because make_block_pulse does not support b1 amp argument
    flip_angle_sl_block = 2 * np.pi * GYROMAGNETIC_RATIO_PROTON * spin_lock_time * spin_lock_amplitude
    rf_spin_lock = pp.make_block_pulse(flip_angle_sl_block, duration=spin_lock_time, system=system)
    seq.add_block(rf_spin_lock)
    
    # add 90° pulse
    rf_post = pp.make_sinc_pulse(np.pi / 2, duration=duration_90, phase_offset=-np.pi / 2, system=system)
    seq.add_block(rf_post)
    
    # add spoiler gradient if requested
    if add_spoiler:
        gz_spoiler = pp.make_trapezoid(
            channel='z',
            amplitude=0.4 * system.max_grad,
            flat_time=spoiler_flat_time,
            rise_time=spoiler_ramp_time,
        )
        seq.add_block(gz_spoiler)

    # calculate total duration of T1rho-prep block
    block_duration = sum(seq.block_durations.values()) - time_start

    # calculate of spoiler gradient
    time_of_spoiler = gz_spoiler.duration() if add_spoiler else 0.0

    return (seq, block_duration, time_of_spoiler)
