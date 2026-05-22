"""Tests for the apply_optimal_filter_fixed_shift offline helper."""

import numpy as np
import pytest

from straxion.plugins.hit_classification import (
    DxHitClassification,
    apply_optimal_filter_fixed_shift,
)
from straxion.utils import load_interpolation
from straxion.constants import DEFAULT_TEMPLATE_INTERP_PATH, NOISE_PSD_38kHz

FS = 38_000
DT = 1.0 / FS
OF_WL = 100
OF_WR = 300
WINDOW = OF_WL + OF_WR


@pytest.fixture(scope="module")
def template():
    """Load the default template interpolation shipped with straxion."""
    At_interp, t_max = load_interpolation(DEFAULT_TEMPLATE_INTERP_PATH)
    return At_interp, t_max


def _make_signal(At_interp, t_max_template, length, peak_sample, amplitude):
    """Synthesize a noiseless signal whose template peak sits at peak_sample."""
    t = np.arange(length) * DT
    return At_interp(t - peak_sample * DT + t_max_template) * amplitude


def test_recovers_known_amplitude_zero_shift(template):
    At_interp, t_max_template = template
    length = 1000
    peak_sample = 500
    amplitude = 1.5e-3

    St = _make_signal(At_interp, t_max_template, length, peak_sample, amplitude)

    ahat, chisq = apply_optimal_filter_fixed_shift(
        St_full=St,
        peak_index=peak_sample,
        tau=0,
        At_interp=At_interp,
        t_max_template=t_max_template,
    )

    assert ahat == pytest.approx(amplitude, rel=1e-6)
    # Noiseless input with matched template -> chi^2 ~ 0.
    assert chisq < 1e-10


def test_zero_signal_gives_zero_amplitude(template):
    At_interp, t_max_template = template
    St = np.zeros(800)

    ahat, _ = apply_optimal_filter_fixed_shift(
        St_full=St,
        peak_index=400,
        tau=0,
        At_interp=At_interp,
        t_max_template=t_max_template,
    )
    assert ahat == pytest.approx(0.0, abs=1e-12)


def test_short_slice_matches_long_slice(template):
    """Trimming St_full to a tight slice around the peak should not change result."""
    At_interp, t_max_template = template
    amplitude = 2.3e-3

    long_len = 1500
    long_peak = 800
    St_long = _make_signal(
        At_interp, t_max_template, long_len, peak_sample=long_peak, amplitude=amplitude
    )

    # Slice 600 samples around the peak (150 left, 450 right).
    slice_left = 150
    slice_right = 450
    St_short = St_long[long_peak - slice_left : long_peak + slice_right]
    short_peak = slice_left

    ahat_long, chi2_long = apply_optimal_filter_fixed_shift(
        St_full=St_long,
        peak_index=long_peak,
        tau=0,
        At_interp=At_interp,
        t_max_template=t_max_template,
    )
    ahat_short, chi2_short = apply_optimal_filter_fixed_shift(
        St_full=St_short,
        peak_index=short_peak,
        tau=0,
        At_interp=At_interp,
        t_max_template=t_max_template,
    )

    assert ahat_short == pytest.approx(ahat_long, rel=1e-8)
    assert chi2_short == pytest.approx(chi2_long, rel=1e-8, abs=1e-12)


def test_fixed_shift_matches_built_in_scan(template):
    """At tau = best_OF_shift, our fixed-shift result must match the scan's.

    The production scan in DxHitClassification.optimal_filter anchors its
    signal window to round(t_max_template / dt), so we synthesize a signal
    whose peak sits near that anchor with a small known offset.
    """
    At_interp, t_max_template = template
    amplitude = 1.0e-3

    # Production anchor and the corresponding peak_index for our wrapper.
    anchor = int(round(t_max_template / DT))
    true_shift = 3
    length = 800
    peak_sample = anchor + true_shift

    St = _make_signal(At_interp, t_max_template, length, peak_sample, amplitude)

    best_aOF, best_chi2, best_OF_shift, _, _ = DxHitClassification.optimal_filter(
        St,
        DT,
        np.asarray(NOISE_PSD_38kHz),
        At_interp,
        t_max_template,
        of_window_left=OF_WL,
        of_window_right=OF_WR,
        of_shift_range_min=-10,
        of_shift_range_max=10,
        of_shift_step=1,
    )

    # Our wrapper centers its window on peak_index, so we use the anchor
    # (matching production) and apply the same tau the scan picked.
    ahat_fixed, chi2_fixed = apply_optimal_filter_fixed_shift(
        St_full=St,
        peak_index=anchor,
        tau=int(best_OF_shift),
        At_interp=At_interp,
        t_max_template=t_max_template,
    )

    # The two implementations build the shifted template differently
    # (analytic interpolation here vs FFT phase shift in production), which
    # leaves a small sub-sample discretization residual.
    assert ahat_fixed == pytest.approx(best_aOF, rel=1e-4)
    assert ahat_fixed == pytest.approx(amplitude, rel=1e-8)
    assert chi2_fixed < 1e-8


def test_template_path_loads_when_at_interp_missing():
    St = np.zeros(WINDOW + 10)
    ahat, _ = apply_optimal_filter_fixed_shift(
        St_full=St,
        peak_index=OF_WL + 5,
        tau=0,
        template_path=DEFAULT_TEMPLATE_INTERP_PATH,
    )
    assert ahat == pytest.approx(0.0, abs=1e-12)


def test_raises_when_no_template_source():
    with pytest.raises(ValueError, match="At_interp or template_path"):
        apply_optimal_filter_fixed_shift(
            St_full=np.zeros(WINDOW + 10),
            peak_index=OF_WL + 5,
        )


def test_raises_when_at_interp_without_t_max(template):
    At_interp, _ = template
    with pytest.raises(ValueError, match="t_max_template"):
        apply_optimal_filter_fixed_shift(
            St_full=np.zeros(WINDOW + 10),
            peak_index=OF_WL + 5,
            At_interp=At_interp,
        )


def test_raises_on_wrong_psd_length(template):
    At_interp, t_max_template = template
    with pytest.raises(ValueError, match="noise_psd length"):
        apply_optimal_filter_fixed_shift(
            St_full=np.zeros(WINDOW + 10),
            peak_index=OF_WL + 5,
            At_interp=At_interp,
            t_max_template=t_max_template,
            noise_psd=np.ones(WINDOW - 1),
        )


@pytest.mark.parametrize(
    "length,peak_index",
    [
        (WINDOW + 10, OF_WL - 1),  # too close to left edge
        (WINDOW + 10, OF_WL + 11),  # too close to right edge (len - OF_WR = OF_WL + 10)
    ],
)
def test_raises_when_window_does_not_fit(template, length, peak_index):
    At_interp, t_max_template = template
    with pytest.raises(ValueError, match="peak_index"):
        apply_optimal_filter_fixed_shift(
            St_full=np.zeros(length),
            peak_index=peak_index,
            At_interp=At_interp,
            t_max_template=t_max_template,
        )


def test_exported_from_package_namespace():
    import straxion

    assert hasattr(straxion, "apply_optimal_filter_fixed_shift")
    assert straxion.apply_optimal_filter_fixed_shift is apply_optimal_filter_fixed_shift
