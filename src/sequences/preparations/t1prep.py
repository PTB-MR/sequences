"""Adiabatic T1 preparation block."""

import pypulseq as pp

from sequences.utils import sys_defaults


def add_t1prep(
    seq: pp.Sequence | None = None,
    system: pp.Opts | None = None,
    inversion_time: float = 21e-3,
    rf_duration: float = 10.24e-3,
    add_spoiler: bool = True,
    spoiler_ramp_time: float = 6e-4,
    spoiler_flat_time: float = 8.4e-3,
) -> tuple[pp.Sequence, float]:
    """Add an adiabatic T1 preparation block to a sequence.

    The adiabatic inversion pulse is a hyperbolic secant pulse with default values similar to the one used by vendors.

    Parameters
    ----------
    seq
        PyPulseq Sequence object.
    system
        PyPulseq system limit object.
    inversion_time
        Desired inversion time (in seconds).
    rf_duration
        Duration of the adiabatic inversion pulse (in seconds).
    add_spoiler
        Toggles addition of spoiler gradients after the inversion pulse.
        The spoiler does not increase the total duration of the block, but limits the minimum inversion time.
    spoiler_ramp_time
        Duration of gradient spoiler ramps (in seconds).
    spoiler_flat_time
        Duration of gradient spoiler plateau (in seconds).

    Returns
    -------
    seq
        PyPulseq Sequence object.
    block_duration
        Total duration of the T1 preparation block (in seconds).

    Raises
    ------
    ValueError
        If the inversion time is too short for the given RF and spoiler durations.
    """
    # set system to default if not provided
    if system is None:
        system = sys_defaults

    if seq is None:
        seq = pp.Sequence(system=system)

    # get current duration of sequence before adding T1 preparation block
    time_start = sum(seq.block_durations.values())

    # calculate inversion time delay
    total_spoil_time = (2 * spoiler_ramp_time + spoiler_flat_time) if add_spoiler else 0
    tau = inversion_time - rf_duration / 2 - system.rf_ringdown_time - total_spoil_time

    # check if delay is valid
    if tau < 0:
        raise ValueError('Inversion time too short for given RF and spoiler durations.')

    # Add adiabatic inversion pulse
    rf = pp.make_adiabatic_pulse(
        pulse_type='hypsec',
        adiabaticity=6,
        beta=800,
        mu=4.9,
        delay=system.rf_dead_time,
        duration=rf_duration,
        system=system,
        use='inversion',
    )
    seq.add_block(rf)

    # Add spoiler gradient if requested
    if add_spoiler:
        gz_spoil = pp.make_trapezoid(
            channel='z',
            amplitude=0.4 * system.max_grad,
            flat_time=spoiler_flat_time,
            rise_time=spoiler_ramp_time,
            system=system,
        )
        seq.add_block(gz_spoil)

    if tau > 0:
        seq.add_block(pp.make_delay(tau))

    # calculate total duration of T1prep block
    block_duration = sum(seq.block_durations.values()) - time_start

    return (seq, block_duration)
