"""Acceptance boundaries for the nominal-background photon selection."""

import numpy as np
import pytest
import strax
import straxion
from straxion.plugins.hit_classification import DxHitClassification
from straxion.plugins.hits import DxHits


def classifier(**config):
    plugin = DxHitClassification()
    plugin.config = {name: option.default for name, option in plugin.takes_config.items()}
    plugin.config.update(config)
    plugin.max_spike_coincidence = plugin.config["max_spike_coincidence"]
    return plugin


def rows(n=1):
    data = np.zeros(n, dtype=DxHitClassification().infer_dtype())
    data["kappa"] = 38  # 1 ms at the default fs
    data["best_chi2"] = 1
    data["n_spikes_coinciding"] = 1
    data["hit_coincidence"] = 1
    data["legacy_spike_coincidence_valid"] = True
    data["peak_ma_valid"] = True
    return data


@pytest.mark.parametrize(
    "field,value,expected",
    [
        ("n_spikes_coinciding", 0, True),
        ("n_spikes_coinciding", 3, True),
        ("n_spikes_coinciding", 4, True),
        ("n_spikes_coinciding", 5, False),
        ("kappa", 0.4 * 38, False),
        # This float32 sample value converts to exactly float32(0.7) ms.
        ("kappa", np.nextafter(np.float32(0.7 * 38), np.float32(-np.inf)), False),
        ("kappa", 0.701 * 38, True),
        ("kappa", 2.499 * 38, True),
        ("kappa", 2.5 * 38, False),
        ("kappa", np.nan, False),
        ("kappa", np.inf, False),
        ("best_chi2", 0.8, False),
        ("best_chi2", 0.801, True),
        ("best_chi2", np.nan, False),
        ("is_symmetric_spike", True, False),
        ("is_truncated_hit", True, False),
    ],
)
def test_nominal_background_decision_boundaries(field, value, expected):
    data = rows()
    data[field] = value
    classifier().compute_photon_selection(data, data)
    assert bool(data["is_photon_candidate"][0]) is expected


def test_kappa_units_use_configured_sampling_frequency():
    data = rows()
    data["kappa"] = 19  # 1 ms at 19 kHz, 0.5 ms at 38 kHz
    classifier(fs=19000).compute_photon_selection(data, data)
    assert data["is_photon_candidate"][0]
    classifier().compute_photon_selection(data, data)
    assert not data["is_photon_candidate"][0]


def test_missing_legacy_count_fails_instead_of_passing_negative_sentinel():
    for field, value in [("legacy_spike_coincidence_valid", False), ("n_spikes_coinciding", -1)]:
        data = rows()
        data[field] = value
        with pytest.raises(ValueError, match="compute_legacy_spike_coincidence"):
            classifier().compute_photon_selection(data, data)


def test_explicit_peak_mode_keeps_shape_cuts_and_rejects_unknown_timing():
    data = rows(5)
    data["legacy_spike_coincidence_valid"] = False
    data["n_spikes_coinciding"] = -1
    data["hit_coincidence"][1] = 2
    data["peak_ma_valid"][2] = False
    data["kappa"][3] = 0.4 * 38
    data["best_chi2"][4] = 0.7
    classifier(classification_coincidence_mode="hit_peaks").compute_photon_selection(data, data)
    np.testing.assert_array_equal(data["is_photon_candidate"], [True, False, False, False, False])


def test_configured_cuts_and_empty_input():
    plugin = classifier(
        max_spike_coincidence=2,
        photon_kappa_min_ms=0.9,
        photon_kappa_max_ms=1.1,
        photon_min_best_chi2=1.5,
    )
    data = rows(4)
    data["best_chi2"] = 2
    data["n_spikes_coinciding"][1] = 3
    data["kappa"][2] = 1.2 * 38
    data["best_chi2"][3] = 1.4
    plugin.compute_photon_selection(data, data)
    np.testing.assert_array_equal(data["is_photon_candidate"], [True, False, False, False])
    plugin.compute_photon_selection(data[:0], data[:0])


@pytest.mark.parametrize(
    "config",
    [{"photon_kappa_min_ms": 2.5}, {"photon_kappa_max_ms": np.nan}, {"photon_min_best_chi2": -1}],
)
def test_invalid_selection_configuration(config):
    with pytest.raises(ValueError):
        classifier(**config).setup()


@pytest.mark.parametrize(
    "config",
    [
        {"photon_kappa_min_ms": 0.8},
        {"photon_kappa_max_ms": 2.4},
        {"photon_min_best_chi2": 0.9},
        {"max_spike_coincidence": 3},
        {"classification_coincidence_mode": "hit_peaks"},
        {"compute_legacy_spike_coincidence": False},
    ],
)
def test_selection_changes_invalidate_strax_lineage(config):
    base = straxion.qualiphide_thz_offline(calibration_mode="per_scan")
    context = strax.Context(config=base.config, register=list(base._plugin_class_registry.values()))
    original = context.key_for("run_A", "hit_classification").lineage_hash
    context.set_config(config)
    assert context.key_for("run_A", "hit_classification").lineage_hash != original


def test_default_context_produces_required_legacy_counts():
    assert DxHits.takes_config["compute_legacy_spike_coincidence"].default is True
    assert classifier().config["classification_coincidence_mode"] == "legacy_record"
