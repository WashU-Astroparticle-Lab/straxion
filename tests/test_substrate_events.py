import numpy as np
import pytest
import straxion
from straxion.plugins.substrate_events import SubstrateEvents, N_CHANNELS
from straxion.colors import get_channel_position


HIT_DTYPE = [
    ("time", np.int64),
    ("endtime", np.int64),
    ("channel", np.int16),
    ("amplitude", np.float32),
    ("best_aOF", np.float32),
    ("best_chi2", np.float32),
    ("kappa", np.float32),
    ("n_spikes_coinciding", int),
    ("is_symmetric_spike", bool),
    ("is_invalid_kappa", bool),
    ("is_truncated_hit", bool),
    ("is_photon_candidate", bool),
]


def _make_hit(time, channel, amplitude=1e-5, best_aOF=1.0, **overrides):
    """One substrate-like hit passing all selection cuts unless overridden."""
    hit = dict(
        time=time,
        endtime=time + 1000,
        channel=channel,
        amplitude=amplitude,
        best_aOF=best_aOF,
        best_chi2=2.0,
        kappa=40.0,  # 40 samples at 38 kHz ~ 1.05 ms, inside (0.5, 3.0) ms
        n_spikes_coinciding=10,
        is_symmetric_spike=False,
        is_invalid_kappa=False,
        is_truncated_hit=False,
        is_photon_candidate=False,
    )
    hit.update(overrides)
    return tuple(hit[name] for name, _ in HIT_DTYPE)


def _get_plugin(**config_overrides):
    st = straxion.qualiphide_thz_offline()
    if config_overrides:
        st.set_config(config_overrides)
    plugin = st.get_single_plugin("test_run", "substrate_events")
    plugin.setup()
    return plugin


def test_substrate_events_plugin_registration():
    """Test that SubstrateEvents is registered in the offline context."""
    st = straxion.qualiphide_thz_offline()
    assert "substrate_events" in st._plugin_class_registry
    assert st._plugin_class_registry["substrate_events"] == SubstrateEvents


def test_substrate_events_dtype_inference():
    """Test that SubstrateEvents infers the expected fields."""
    plugin = _get_plugin()
    dtype = np.dtype(plugin.infer_dtype())
    expected_fields = [
        "time",
        "endtime",
        "max_amplitude_time",
        "n_hits",
        "pattern_amplitude",
        "pattern_best_aOF",
        "pattern_energy_mev",
        "pattern_energy_of_mev",
        "summed_energy_mev",
        "summed_energy_of_mev",
        "max_energy_channel",
        "max_energy_channel_of",
        "max_energy_position",
        "max_energy_position_of",
        "all_distances_mm",
    ]
    for field in expected_fields:
        assert field in dtype.names, f"Missing field: {field}"
    assert dtype["pattern_amplitude"].shape == (N_CHANNELS,)
    assert dtype["all_distances_mm"].shape == (N_CHANNELS,)
    assert dtype["max_energy_position"].shape == (2,)


def test_find_clusters():
    """Test the clustering helper on a simple sorted time series."""
    times = np.array([0, 1, 5, 6, 7, 20, 21])
    clusters = SubstrateEvents.find_clusters(times, window_ns=2, min_coincidence=2)
    assert [list(c) for c in clusters] == [[0, 1], [2, 3, 4], [5, 6]]

    clusters = SubstrateEvents.find_clusters(times, window_ns=2, min_coincidence=3)
    assert [list(c) for c in clusters] == [[2, 3, 4]]

    assert SubstrateEvents.find_clusters(np.array([]), 2, 2) == []

    with pytest.raises(ValueError, match="sorted"):
        SubstrateEvents.find_clusters(np.array([3, 1, 2]), 2, 2)


def test_substrate_hit_selection():
    """Test that each quality cut removes the intended hits."""
    plugin = _get_plugin()
    hits = np.array(
        [
            _make_hit(0, 0),  # passes
            _make_hit(10, 1, n_spikes_coinciding=2),  # too few spike coincidences
            _make_hit(20, 2, kappa=5.0),  # kappa too small (~0.13 ms)
            _make_hit(30, 3, kappa=200.0),  # kappa too large (~5.3 ms)
            _make_hit(40, 4, best_chi2=0.5),  # chi2 too small (photon-like)
            _make_hit(50, 5, is_symmetric_spike=True),
            _make_hit(60, 6, is_invalid_kappa=True),
            _make_hit(70, 7, is_photon_candidate=True),
            _make_hit(80, 8, is_truncated_hit=True),  # kept by default
        ],
        dtype=HIT_DTYPE,
    )
    mask = SubstrateEvents.select_substrate_hits(hits, plugin.config)
    np.testing.assert_array_equal(
        mask, [True, False, False, False, False, False, False, False, True]
    )

    plugin_strict = _get_plugin(substrate_exclude_truncated_hits=True)
    mask = SubstrateEvents.select_substrate_hits(hits, plugin_strict.config)
    assert not mask[8]


def test_substrate_events_compute():
    """Test event building end to end on synthetic hits."""
    plugin = _get_plugin()
    window_ns = int(plugin.coincidence_window_ns)

    # Cluster 1: 6 hits on channels 0-5 within the coincidence window,
    # channel 3 has the largest amplitude.
    cluster1 = [
        _make_hit(1_000_000 + i * window_ns // 2, channel=i, amplitude=(1 + i % 4) * 1e-5)
        for i in range(6)
    ]
    # Isolated pair far away: below min_hit_coincidence, must not become an event.
    pair = [_make_hit(500_000_000 + i * 100, channel=i) for i in range(2)]
    # Cluster 2: exactly min_hit_coincidence hits.
    cluster2 = [_make_hit(1_000_000_000 + i * 100, channel=10 + i) for i in range(5)]
    # A photon-like hit inside cluster 2's time range must be ignored.
    photon = [_make_hit(1_000_000_050, channel=20, is_photon_candidate=True)]

    hits = np.array(cluster1 + pair + cluster2 + photon, dtype=HIT_DTYPE)
    events = plugin.compute(hits=hits)

    assert len(events) == 2
    event1, event2 = events

    # Timing.
    assert event1["time"] == 1_000_000
    assert event1["endtime"] == cluster1[-1][0] + 1000
    assert event1["n_hits"] == 6
    assert event2["n_hits"] == 5

    # Channel 3 has amplitude 4e-5, the cluster maximum.
    assert event1["max_energy_channel"] == 3
    assert event1["max_amplitude_time"] == cluster1[3][0]
    np.testing.assert_allclose(event1["pattern_amplitude"][3], 4e-5, rtol=1e-6)
    # Channels that did not fire stay at zero.
    assert event1["pattern_amplitude"][6:].sum() == 0

    # Energy calibration: amplitude / single_photon_dx_amplitude * single_photon_energy_mev.
    expected_energy = (
        4e-5
        / plugin.config["single_photon_dx_amplitude"]
        * (plugin.config["single_photon_energy_mev"])
    )
    np.testing.assert_allclose(event1["pattern_energy_mev"][3], expected_energy, rtol=1e-5)
    np.testing.assert_allclose(
        event1["summed_energy_mev"], event1["pattern_energy_mev"].sum(), rtol=1e-6
    )
    # OF-based pattern: best_aOF = 1.0 on every fired channel.
    np.testing.assert_allclose(
        event1["pattern_energy_of_mev"][:6], plugin.config["of_amplitude_to_mev"], rtol=1e-5
    )

    # Location: position of the max-energy channel, and distances from it.
    np.testing.assert_allclose(event1["max_energy_position"], get_channel_position(3), rtol=1e-6)
    expected_distances = np.sqrt(
        np.sum((get_channel_position(np.arange(N_CHANNELS)) - get_channel_position(3)) ** 2, axis=1)
    )
    np.testing.assert_allclose(event1["all_distances_mm"], expected_distances, rtol=1e-5)
    assert event1["all_distances_mm"][3] == 0.0


def test_substrate_events_duplicate_channel():
    """If a channel fires twice in a cluster, its largest-amplitude hit wins."""
    plugin = _get_plugin(substrate_min_hit_coincidence=2)
    hits = np.array(
        [
            _make_hit(1_000, channel=0, amplitude=5e-5, best_aOF=5.0),
            _make_hit(1_100, channel=0, amplitude=1e-5, best_aOF=1.0),
            _make_hit(1_200, channel=1, amplitude=2e-5, best_aOF=2.0),
        ],
        dtype=HIT_DTYPE,
    )
    events = plugin.compute(hits=hits)
    assert len(events) == 1
    np.testing.assert_allclose(events[0]["pattern_amplitude"][0], 5e-5, rtol=1e-6)
    np.testing.assert_allclose(events[0]["pattern_best_aOF"][0], 5.0, rtol=1e-6)
    assert events[0]["n_hits"] == 3


def test_substrate_events_empty_input():
    """No hits in, no events out."""
    plugin = _get_plugin()
    events = plugin.compute(hits=np.zeros(0, dtype=HIT_DTYPE))
    assert len(events) == 0
