"""Gold standard GRE-based inversion recovery sequence with one inversion pulse before every readout."""

from pathlib import Path

import numpy as np
import pypulseq as pp

from sequences.preparations import add_t1prep
from sequences.utils import sys_defaults


def main(
    system: pp.Opts | None = None,
    inversion_times: np.ndarray | None = None,
    te: float | None = None,
    tr: float = 8,
    fov_xy: float = 128e-3,
    n_readout: int = 128,
    n_phase_encoding: int = 128,
    slice_thickness: float = 8e-3,
    show_plots: bool = True,
    test_report: bool = True,
    timing_check: bool = True,
) -> pp.Sequence:
    """Generate a GRE-based inversion recovery sequence with one inversion pulse before every readout.

    Parameters
    ----------
    system
        PyPulseq system limits object.
    inversion_times
        Array of inversion times (in seconds).
        Default values [25, 50, 300, 600, 1200, 2400, 4800] ms are used if set to None.
    te
        Desired echo time (TE) (in seconds). Minimum echo time is used if set to None.
    tr
        Desired repetition time (TR) (in seconds).
    fov_xy
        Field of view in x and y direction (in meters).
    n_readout
        Number of frequency encoding steps.
    n_phase_encoding
        Number of phase encoding steps.
    slice_thickness
        Slice thickness of the 2D slice (in meters).
    show_plots
        Toggles sequence plot.
    test_report
        Toggles advanced test report.
    timing_check
        Toggles timing check of the sequence.
    """
    if system is None:
        system = sys_defaults

    if inversion_times is None:
        inversion_times = np.array([25, 50, 300, 600, 1200, 2400, 4800]) / 1e3

    # create PyPulseq Sequence object and set system limits
    seq = pp.Sequence(system=system)

    # define T1prep settings
    rf_inv_duration = 10.24e-3  # duration of adiabatic inversion pulse [s]
    rf_inv_spoil_risetime = 0.6e-3  # rise time of spoiler after inversion pulse [s]
    rf_inv_spoil_flattime = 8.4e-3  # flat time of spoiler after inversion pulse [s]

    # define ADC and gradient timing
    adc_dwell = system.grad_raster_time
    gx_pre_duration = 1.0e-3  # duration of readout pre-winder gradient [s]
    gx_flat_time = n_readout * adc_dwell  # flat time of readout gradient [s]

    # define settings of rf excitation pulse
    rf_duration = 1.28e-3  # duration of the rf excitation pulse [s]
    rf_flip_angle = 12  # flip angle of rf excitation pulse [°]
    rf_bwt = 4  # bandwidth-time product of rf excitation pulse [Hz*s]
    rf_apodization = 0.5  # apodization factor of rf excitation pulse

    # create slice selective excitation pulse and gradients
    rf, gz, gzr = pp.make_sinc_pulse(  # type: ignore
        flip_angle=rf_flip_angle / 180 * np.pi,
        duration=rf_duration,
        slice_thickness=slice_thickness,
        apodization=rf_apodization,
        time_bw_product=rf_bwt,
        delay=system.rf_dead_time,
        system=system,
        return_gz=True,
    )

    # create readout gradient and ADC
    delta_k = 1 / fov_xy
    gx = pp.make_trapezoid(channel='x', flat_area=n_readout * delta_k, flat_time=gx_flat_time, system=system)
    adc = pp.make_adc(num_samples=n_readout, duration=gx.flat_time, delay=gx.rise_time, system=system)

    # create frequency encoding pre- and re-winder gradient
    gx_pre = pp.make_trapezoid(channel='x', area=-gx.area / 2 - delta_k / 2, duration=gx_pre_duration, system=system)
    gx_post = pp.make_trapezoid(channel='x', area=-gx.area / 2 + delta_k / 2, duration=gx_pre_duration, system=system)

    # calculate gradient areas for (linear) phase encoding direction
    phase_areas = (np.arange(n_phase_encoding) - n_phase_encoding / 2) * delta_k
    k0_center_id = np.where((np.arange(n_readout) - n_readout / 2) * delta_k == 0)[0][0]

    # create spoiler gradients
    gz_spoil = pp.make_trapezoid(channel='z', area=4 / slice_thickness, system=system)

    # calculate minimum echo time
    min_te = (
        rf.shape_dur / 2  # time from center to end of RF pulse
        + rf.ringdown_time  # RF ringdown time
        + pp.calc_duration(gzr)  # slice selection rewinder gradient
        + pp.calc_duration(gx_pre)  # readout pre-winder gradient
        + gx.delay  # potential delay of readout gradient
        + gx.rise_time  # rise time of readout gradient
        + k0_center_id * adc.dwell  # time from begin of ADC to time point of k-space center sample
    )

    # calculate delay to achieve desired echo time
    if te is None:
        te_delay = 0
    elif te > min_te:
        te_delay = np.ceil((te - min_te) / system.grad_raster_time) * system.grad_raster_time
    else:
        raise ValueError(f'TE must be larger than {min_te * 1000:.2f} ms. Current value is {te * 1000:.2f} ms.')

    for ti_idx, ti in enumerate(inversion_times):
        # set contrast ('ECO') label for current inversion time
        contrast_label = pp.make_label(type='SET', label='ECO', value=int(ti_idx))

        # loop over phase encoding steps
        for pe_idx in np.arange(n_phase_encoding):
            # set phase encoding ('LIN') label
            pe_label = pp.make_label(type='SET', label='LIN', value=int(pe_idx))

            # save start time of current TR block
            _start_time_tr_block = sum(seq.block_durations.values())

            seq, _ = add_t1prep(
                seq=seq,
                system=system,
                inversion_time=ti,
                rf_duration=rf_inv_duration,
                spoiler_ramp_time=rf_inv_spoil_risetime,
                spoiler_flat_time=rf_inv_spoil_flattime,
            )

            # add rf pulse followed by refocusing gradient
            seq.add_block(rf, gz)
            seq.add_block(gzr)

            # add echo time delay
            seq.add_block(pp.make_delay(te_delay))

            # calculate phase encoding gradient for current phase encoding step
            gy_pre = pp.make_trapezoid(channel='y', area=phase_areas[pe_idx], duration=gx_pre_duration, system=system)

            # add pre-winder gradients and labels
            seq.add_block(gx_pre, gy_pre, pe_label, contrast_label)

            # add readout gradient and ADC
            seq.add_block(gx, adc)

            # add x and y re-winder and spoiler gradient in z-direction
            gy_pre.amplitude = -gy_pre.amplitude
            seq.add_block(gx_post, gy_pre, gz_spoil)

            # calculate TR delay
            duration_tr_block = sum(seq.block_durations.values()) - _start_time_tr_block
            tr_delay = tr - duration_tr_block
            tr_delay = np.ceil(tr_delay / system.grad_raster_time) * system.grad_raster_time

            # save time for sequence plot
            if show_plots and ti_idx == 0 and pe_idx == 0:
                upper_time_limit = duration_tr_block

            if tr_delay < 0:
                raise ValueError('Desired TR too short for given sequence parameters.')

            seq.add_block(pp.make_delay(tr_delay))

    # check timing of the sequence
    if timing_check and not test_report:
        ok, error_report = seq.check_timing()
        if ok:
            print('\nTiming check passed successfully')
        else:
            print('\nTiming check failed! Error listing follows\n')
            print(error_report)

    # show advanced rest report
    if test_report:
        print('\nCreating advanced test report...')
        print(seq.test_report())

    # define sequence filename
    filename = (
        f'{Path(__file__).stem}_{int(fov_xy * 1000)}fov_{n_readout}nx_{n_phase_encoding}ny_{len(inversion_times)}TIs'
    )

    # write all required parameters in the seq-file header/definitions
    seq.set_definition('FOV', [fov_xy, fov_xy, slice_thickness])
    seq.set_definition('ReconMatrix', (n_readout, n_phase_encoding, 1))
    seq.set_definition('SliceThickness', slice_thickness)
    seq.set_definition('TE', te or min_te)
    seq.set_definition('TR', tr)
    seq.set_definition('TI', inversion_times)

    # save seq-file to disk
    output_path = Path.cwd() / 'output'
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving sequence file '{filename}.seq' into folder '{output_path}'.")
    seq.write(str(output_path / filename), create_signature=True)

    if show_plots:
        seq.plot(time_range=(0, upper_time_limit))

    return seq


if __name__ == '__main__':
    main()
