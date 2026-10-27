"""Synthetic contract/regression tests; these are not a real-data validation."""

import numpy as np
import pytest
import strax
import straxion
from straxion.hit_metadata import TIMING_DTYPE, ESTIMATOR_DTYPE
from straxion.substrate_reconstruction import prompt_groups, apply_veto, hit_coincidences


def context(**config):
    st = straxion.qualiphide_thz_offline()
    st.set_config(config)
    return st


def records():
    n = 4500
    dtype = [
        ("time", "i8"),
        ("endtime", "i8"),
        ("length", "i8"),
        ("dt", "i8"),
        ("channel", "i2"),
    ] + [
        (f, "f4", n) for f in ("data_dx", "data_dx_moving_average", "data_dx_convolved", "data_dr")
    ]
    r = np.zeros(2, dtype=dtype)
    r["time"] = 1_700_000_000_000_000_123
    r["endtime"] = r["time"] + n * 1_000_000_000 // 38000
    r["length"] = n
    r["dt"] = 26315
    r["channel"] = [0, 1]
    rng = np.random.default_rng(719)
    for field in ("data_dx", "data_dx_moving_average", "data_dx_convolved"):
        r[field] = rng.normal(0, 0.001, (2, n))
    for ch in range(2):
        for peak in (3, 800, 850, 2800, 4480):
            r[ch]["data_dx_convolved"][max(0, peak - 3) : peak + 4] = 1
            r[ch]["data_dx_convolved"][peak] = 2
            r[ch]["data_dx"][min(n - 1, peak + 2)] = 3
            r[ch]["data_dx_moving_average"][min(n - 1, peak + 1)] = 4
    # Real zeros inside valid support must not be mistaken for padding.
    r[0]["data_dx"][2799] = 0
    return r


def produce():
    st = context(
        hit_threshold_dx=0.5,
        hit_thresholds_sigma=None,
        hit_min_width=1 / 38000,
        compute_legacy_spike_coincidence=True,
        classification_coincidence_mode="legacy_record",
    )
    producer = st.get_single_plugin("run_A", "hits")
    r = records()
    h = producer.compute(r)
    noise = st.get_single_plugin("run_A", "noises").compute(r, h)
    classifier = st.get_single_plugin("run_A", "hit_classification")
    return st, r, h, noise, classifier, h


def test_native_timing_support_and_identity():
    st, r, c, _, _, h = produce()
    assert len(c) == 10
    for hit in c:
        assert hit["alignment_valid"]
        origin = int(hit["record_time"])
        for basis in ("raw", "ma", "convolved"):
            assert (
                hit[f"peak_{basis}_time"]
                == origin + int(hit[f"peak_{basis}_record_i"]) * 10**9 // 38000
            )
            assert hit[f"peak_{basis}_valid"]
        assert hit["support_stop_waveform_i"] - hit["support_start_waveform_i"] == hit["length"]
        assert hit["clean_left_alignment"] + hit["clean_right_alignment"] + 1 == hit["length"]
    edge = c[(c["candidate_record_i"] == 0) & (c["channel"] == 0)][0]
    assert edge["alignment_record_i"] == 3
    assert edge["peak_ma_record_i"] == 4
    assert edge["peak_raw_record_i"] == 5
    assert edge["peak_ma_time"] != edge["time"] + 200 * 10**9 // 38000
    intact = c[(c["alignment_record_i"] == 2800) & (c["channel"] == 0)][0]
    assert intact["intact25_alignment"]
    assert intact["legacy_nonzero_left_alignment"] == 0
    assert intact["clean_left_alignment"] == 200
    assert np.any(c["clean_right_alignment"] < 599)
    assert len(np.unique(c["hit_id"])) == len(c)
    again = st.get_single_plugin("run_A", "hits").compute(r)
    np.testing.assert_array_equal(c["hit_id"], again["hit_id"])
    other = st.get_single_plugin("run_B", "hits").compute(r)
    assert not set(c["hit_id"]) & set(other["hit_id"])
    np.testing.assert_array_equal(c["hit_id"], h["hit_id"])


def legacy_classify(plugin, hits, rec, noises):
    """Original compute sequence at c9aaa38, using the unchanged estimators."""
    out = np.zeros(len(hits), dtype=plugin.infer_dtype())
    for field in (
        "time",
        "endtime",
        "channel",
        "width",
        "amplitude",
        "amplitude_moving_average",
        "amplitude_convolved",
        "amplitude_dr",
        "hit_threshold",
    ):
        out[field] = hits[field]
    thresholds = plugin.determine_spike_threshold(rec)
    psds = plugin.compute_per_channel_noise_psd(noises, len(rec))
    plugin.compute_rise_edge_slope(hits, out)
    plugin.find_spike_coincidence(out, hits, rec, thresholds)
    plugin.is_symmetric_spike_hit(hits, out)
    plugin.is_truncated_hit(hits, out)
    plugin.compute_optimal_filter_parameters(out, hits, psds)
    plugin.compute_tau_fits(out, hits)
    out["is_coincident_with_spikes"] = out["n_spikes_coinciding"] > plugin.max_spike_coincidence
    out["is_invalid_kappa"] = ~np.isfinite(out["kappa"])
    out["is_photon_candidate"] = ~(
        out["is_coincident_with_spikes"]
        | out["is_symmetric_spike"]
        | out["is_truncated_hit"]
        | out["is_invalid_kappa"]
    )
    return out


@pytest.mark.parametrize("fallback", [False, True])
def test_classification_estimators_unchanged(fallback):
    st, r, c, noise, upstream, h = produce()
    if fallback:
        noise = noise[:0]
    downstream = st.get_single_plugin("run_A", "hit_classification")
    assert downstream.depends_on == ("hits", "noises")
    expected_plugin = st.get_single_plugin("run_A", "hit_classification")
    expected_plugin.setup_optimal_filter()
    old = legacy_classify(expected_plugin, c, r, noise)
    new = downstream.compute(h, noise)
    added = {d[0] for d in TIMING_DTYPE + ESTIMATOR_DTYPE} | {"is_coincident_with_hits"}
    for field in new.dtype.names:
        if field not in added:
            np.testing.assert_array_equal(new[field], old[field], err_msg=field)
    assert np.all(new["noise_psd_fallback"] == fallback)
    # Preserve legacy negative-slice behavior at the record edge.
    assert np.all(new["n_spikes_coinciding"][c["candidate_record_i"] == 0] == 0)
    assert np.all(new["hit_coincidence"][c["candidate_record_i"] == 0] == 2)


def table(times, channels=None):
    n = len(times)
    h = np.zeros(
        n,
        dtype=[("time", "i8"), ("endtime", "i8"), ("channel", "i2"), ("amplitude", "f4")]
        + TIMING_DTYPE,
    )
    h["time"] = np.arange(n) * 10
    h["endtime"] = h["time"] + 100000
    h["channel"] = np.arange(n) % 3 if channels is None else channels
    h["amplitude"] = np.arange(n) + 1
    h["hit_id"] = [str(i).encode() for i in range(n)]
    h["run_id"] = b"A"
    for basis in ("raw", "ma", "convolved"):
        h[f"peak_{basis}_time"] = times
        h[f"peak_{basis}_valid"] = True
    return h


def test_ma_coincidence_hits_channels_links_boundaries():
    h = table([100, 105, 110, 111], [0, 0, 1, 2])
    counts, links = hit_coincidences(h, 10, with_links=True)
    assert tuple(counts[0]) == (3, 2)  # self, same-channel duplicate, exact boundary
    assert set(links["member_hit_id"][links["reference_hit_id"] == b"0"]) == {b"0", b"1", b"2"}
    h["peak_ma_valid"][2] = False
    counts, _ = hit_coincidences(h, 10)
    assert tuple(counts[0]) == (2, 1)
    assert tuple(counts[2]) == (0, 0)
    h["record_time"][1] = 1
    counts, _ = hit_coincidences(h, 10)
    assert tuple(counts[0]) == (1, 1)


def test_groups_chaining_order_identity_and_veto():
    # Chain spans 400 ns despite a 100 ns gap; repeated channels count as hits.
    h = table([400, 0, 200, 100, 300, 1000, 1100, 1500, 1600, 1800, 1900])
    groups, members = prompt_groups(h, np.ones(len(h), bool), basis="ma", gap_ns=100, min_hits=2)
    assert groups[0]["n_hits"] == 5
    assert groups[0]["n_channels"] == 3
    assert groups[0]["n_repeated_channels"] == 2
    assert groups[0]["last_time"] - groups[0]["first_time"] == 400
    veto = apply_veto(groups, 600)
    # Exact veto boundary accepts; rejected third group cannot extend the veto.
    np.testing.assert_array_equal(veto["accepted"], [True, True, False, True])
    reordered = h[::-1]
    g2, m2 = prompt_groups(reordered, np.ones(len(h), bool), basis="ma", gap_ns=100, min_hits=2)
    np.testing.assert_array_equal(groups, g2)
    np.testing.assert_array_equal(members["hit_id"], m2["hit_id"])
    assert np.all(reordered["hit_id"][m2["hit_index"]] == m2["hit_id"])
    h["record_time"][4] = 42
    g3, _ = prompt_groups(h, np.ones(len(h), bool), basis="ma", gap_ns=100, min_hits=2)
    assert len(g3) > len(groups)


def test_subthreshold_groups_do_not_start_veto():
    h = table([0, 200, 201])
    g, _ = prompt_groups(h, np.ones(3, bool), basis="ma", gap_ns=1, min_hits=2)
    np.testing.assert_array_equal(apply_veto(g, 1000)["accepted"], [False, True])


def test_plugin_defaults_and_fragmented_chunks():
    st, _, _, noise, _, h = produce()
    classification = st.get_single_plugin("run_A", "hit_classification").compute(h, noise)
    # Force seeds; no truncation or energy requirement is introduced.
    classification["kappa"] = 38
    classification["best_chi2"] = 2
    classification["n_spikes_coinciding"] = 10
    classification["is_symmetric_spike"] = False
    classification["is_invalid_kappa"] = False
    classification["is_photon_candidate"] = False
    st.set_config({"substrate_min_hit_coincidence": 1, "substrate_min_peak_channels": 1})
    plugin = st.get_single_plugin("run_A", "substrate_events")
    assert plugin.config["substrate_time_basis"] == "ma"
    expected = plugin.compute(classification)
    start = int(classification["record_time"][0])
    end = int(classification["record_endtime"][0])
    full = strax.Chunk(
        start=start,
        end=end,
        run_id="run_A",
        data_type="hit_classification",
        data_kind="hits",
        dtype=classification.dtype,
        data=classification,
    )
    # Split legally between separated hit-window groups; include an empty middle chunk.
    split = start + 2000 * 10**9 // 38000
    first, tail = full.split(split, allow_early_split=True)
    empty = strax.Chunk(
        start=first.end,
        end=first.end,
        run_id="run_A",
        data_type="hit_classification",
        data_kind="hits",
        dtype=classification.dtype,
        data=classification[:0],
    )
    result = list(plugin.iter({"hit_classification": iter([first, empty, tail])}))
    np.testing.assert_array_equal(expected, result[0].data)
    assert result[0].start == start and result[0].end == end


def test_lineage_tracks_configuration():
    st = context()
    old = st.key_for("run_A", "hit_classification").lineage_hash
    st.set_config({"hit_coincidence_window_ns": 120000})
    assert st.key_for("run_A", "hit_classification").lineage_hash != old
    old = st.key_for("run_A", "hit_classification").lineage_hash
    st.set_config({"template_content_sha256": "changed-own-template"})
    assert st.key_for("run_A", "hit_classification").lineage_hash != old
    old = st.key_for("run_A", "records").lineage_hash
    st.set_config({"dx_map_sha256": "changed-avgmap"})
    assert st.key_for("run_A", "records").lineage_hash != old


class SyntheticRecords(strax.Plugin):
    depends_on = ()
    provides = "records"
    data_kind = "records"
    save_when = strax.SaveWhen.NEVER
    rechunk_on_save = False

    def infer_dtype(self):
        return records().dtype

    def is_ready(self, chunk_i):
        return chunk_i == 0

    def source_finished(self):
        return True

    def compute(self, chunk_i):
        r = records()
        return self.chunk(start=int(r["time"][0]), end=int(r["endtime"][0]), data=r)


def test_full_dependency_pipeline_and_export(tmp_path):
    from straxion.export_peak_contract import export_peak_contract
    import json

    st = context(hit_threshold_dx=0.5, hit_thresholds_sigma=None, hit_min_width=1 / 38000)
    st.storage = [strax.DataDirectory(str(tmp_path / "cache"))]
    st.register(SyntheticRecords)
    data = st.get_array("run_A", "hit_classification")
    assert len(data) == 10
    assert np.all(data["hit_coincidence"] >= 2)
    expected = st.get_single_plugin("run_A", "substrate_events").compute(data)
    actual = st.get_array("run_A", "substrate_events")
    np.testing.assert_array_equal(expected, actual)
    manifest = export_peak_contract(
        st, "run_A", tmp_path / "export", source_revision="synthetic-test", waveform_count=2
    )
    assert manifest["n_detected_hits"] == len(data)
    saved = np.load(tmp_path / "export" / "hit_classification.npy", allow_pickle=False)
    np.testing.assert_array_equal(data, saved)
    disk = json.loads((tmp_path / "export" / "manifest.json").read_text())
    assert disk["lineage"]["hits"]["hash"] == st.key_for("run_A", "hits").lineage_hash
    assert (tmp_path / "export" / "validation_waveforms.npy").exists()


def test_peak_alignment_distinct_and_boundary_flags():
    st = context(hit_threshold_dx=0.5, hit_thresholds_sigma=None, hit_min_width=3 / 38000)
    plugin = st.get_single_plugin("run_A", "hits")
    r = records()[:1].copy()
    for field in ("data_dx", "data_dx_moving_average", "data_dx_convolved"):
        r[field] = 0
    r["data_dx_convolved"][0, 1000:1010] = 1
    r["data_dx_convolved"][0, 1005] = 2
    # Narrow rejected candidate inside extraction support wins convolved maximum.
    r["data_dx_convolved"][0, 1100] = 3
    r["data_dx"][0, 805] = 4
    r["data_dx_moving_average"][0, 1604] = 5
    h = plugin.compute(r)[0]
    assert h["alignment_record_i"] == 1005
    assert h["peak_convolved_record_i"] == 1100
    assert h["peak_raw_at_left_boundary"]
    assert h["peak_ma_at_right_boundary"]


def test_long_chain_and_invalid_peak_rejection():
    h = table(np.arange(1000, dtype=np.int64) * 100)
    groups, _ = prompt_groups(h, np.ones(len(h), bool), basis="ma", gap_ns=100)
    assert len(groups) == 1 and groups[0]["n_hits"] == 1000
    h["peak_ma_valid"][0] = False
    with pytest.raises(ValueError, match="invalid peaks"):
        prompt_groups(h, np.ones(len(h), bool), basis="ma", gap_ns=100)


def test_legacy_ties_and_peak_ties():
    h = table([100, 100], [1, 0])
    h["time"] = 0
    h["amplitude"] = 1
    h["hit_id"] = [b"z", b"a"]
    legacy, _ = prompt_groups(h, np.ones(2, bool), basis="start", gap_ns=1, min_hits=1)
    peak, _ = prompt_groups(h, np.ones(2, bool), basis="ma", gap_ns=1, min_hits=1)
    assert legacy[0]["brightest_hit_id"] == b"z"
    assert peak[0]["brightest_hit_id"] == b"a"


def test_classifier_psd_independent_of_cached_chunking():
    st, _, _, noise, plugin, h = produce()
    start, end = int(h["record_time"][0]), int(h["record_endtime"][0])
    hc = strax.Chunk(
        start=start,
        end=end,
        run_id="run_A",
        data_type="hits",
        data_kind="hits",
        dtype=h.dtype,
        data=h,
    )
    nc = strax.Chunk(
        start=start,
        end=end,
        run_id="run_A",
        data_type="noises",
        data_kind="noises",
        dtype=noise.dtype,
        data=noise,
    )
    h1, h2 = hc.split(start + 2000 * 10**9 // 38000, allow_early_split=True)
    n1, n2 = nc.split(start + 1500 * 10**9 // 38000, allow_early_split=True)
    empty = strax.Chunk(
        start=n1.end,
        end=n1.end,
        run_id="run_A",
        data_type="noises",
        data_kind="noises",
        dtype=noise.dtype,
        data=noise[:0],
    )
    expected = plugin.compute(h, noise)
    outputs = list(plugin.iter({"hits": iter([h1, h2]), "noises": iter([n1, empty, n2])}))
    assert len(outputs) == 1
    np.testing.assert_array_equal(outputs[0].data, expected)
    assert np.any(expected["noise_window_count"] > 0)


def test_optional_legacy_count_and_default_peak_selection():
    st = context(hit_threshold_dx=0.5, hit_thresholds_sigma=None, hit_min_width=1 / 38000)
    r = records()
    hits = st.get_single_plugin("run_A", "hits").compute(r)
    assert np.all(hits["n_spikes_coinciding"] == -1)
    assert not np.any(hits["legacy_spike_coincidence_valid"])
    assert np.all(hits["hit_coincidence"] >= 2)
    noise = st.get_single_plugin("run_A", "noises").compute(r, hits)
    classified = st.get_single_plugin("run_A", "hit_classification").compute(hits, noise)
    assert np.all(classified["is_coincident_with_hits"])
    assert not np.any(classified["is_photon_candidate"])
    classified["kappa"] = 38
    classified["best_chi2"] = 2
    classified["is_symmetric_spike"] = False
    classified["is_invalid_kappa"] = False
    st.set_config({"substrate_min_peak_channels": 2})
    plugin = st.get_single_plugin("run_A", "substrate_events")
    assert np.all(plugin.select_substrate_hits(classified, plugin.config))
    with pytest.raises(ValueError, match="Legacy seeds"):
        plugin.select_substrate_hits(
            classified, dict(plugin.config, substrate_seed_coincidence="record_threshold")
        )


def test_export_retains_scalar_rows_with_unavailable_peak_view(tmp_path, monkeypatch):
    from straxion.export_peak_contract import export_peak_contract

    st, _, _, noise, plugin, h = produce()
    classified = plugin.compute(h, noise)
    classified["kappa"] = 38
    classified["best_chi2"] = 2
    classified["is_symmetric_spike"] = False
    classified["is_invalid_kappa"] = False
    classified["peak_raw_valid"][0] = False
    st.set_config({"substrate_min_peak_channels": 1})
    st.register(SyntheticRecords)
    monkeypatch.setattr(st, "get_array", lambda run, target: classified)
    manifest = export_peak_contract(st, "run_A", tmp_path / "export", source_revision="synthetic")
    assert "groups_raw" in manifest["unavailable_views"]
    assert "fixed_start_membership_raw" in manifest["unavailable_views"]
    assert len(np.load(tmp_path / "export" / "hit_classification.npy")) == len(h)


def test_classifier_rejects_partial_acquisition_coverage():
    st, _, _, noise, plugin, h = produce()
    start, end = int(h["record_time"][0]), int(h["record_endtime"][0])
    # First hit starts at origin; select later hits so the partial chunk is legal.
    selected = h[h["time"] > start + 1000000]
    selected_noise = noise[noise["time"] > start + 1000000]
    hc = strax.Chunk(
        start=start + 1000000,
        end=end,
        run_id="run_A",
        data_type="hits",
        data_kind="hits",
        dtype=h.dtype,
        data=selected,
    )
    nc = strax.Chunk(
        start=start + 1000000,
        end=end,
        run_id="run_A",
        data_type="noises",
        data_kind="noises",
        dtype=noise.dtype,
        data=selected_noise,
    )
    with pytest.raises(ValueError, match="complete acquisition coverage"):
        list(plugin.iter({"hits": iter([hc]), "noises": iter([nc])}))


def test_classifier_acquisition_psds_remain_isolated():
    _, _, _, noise, plugin, h = produce()
    second = h.copy()
    shift = int(h["record_endtime"][0] - h["record_time"][0]) + 1
    for field in (
        "time",
        "endtime",
        "record_time",
        "record_endtime",
        "alignment_time",
        "peak_raw_time",
        "peak_ma_time",
        "peak_convolved_time",
    ):
        second[field] += shift
    second["hit_id"] = [b"second_" + str(i).encode() for i in range(len(second))]
    second_noise = noise.copy()
    for field in ("time", "endtime", "record_time", "record_endtime"):
        second_noise[field] += shift
    second_noise["data_dx"] *= 3
    first_expected = plugin.compute(h, noise)
    second_expected = plugin.compute(second, second_noise)
    together = plugin.compute(np.r_[h, second], np.r_[noise, second_noise])
    np.testing.assert_array_equal(together[: len(h)], first_expected)
    np.testing.assert_array_equal(together[len(h) :], second_expected)
    assert np.any(first_expected["noise_psd_sha256"] != second_expected["noise_psd_sha256"])


def test_invalid_ma_cannot_pass_photon_selection(monkeypatch):
    st, _, h, noise, _, _ = produce()
    st.set_config({"classification_coincidence_mode": "hit_peaks"})
    plugin = st.get_single_plugin("run_A", "hit_classification")
    # Hold unrelated estimators benign so timing validity alone controls selection.
    for name in (
        "compute_rise_edge_slope",
        "is_symmetric_spike_hit",
        "is_truncated_hit",
        "compute_tau_fits",
    ):
        monkeypatch.setattr(plugin, name, lambda *args: None)
    monkeypatch.setattr(
        plugin,
        "compute_optimal_filter_parameters",
        lambda out, *args: out["kappa"].fill(38),
    )
    h["hit_coincidence"] = 1
    valid = plugin.compute(h, noise)
    assert np.all(valid["is_photon_candidate"])
    h["peak_ma_valid"][0] = False
    h["hit_coincidence"][0] = 0
    invalid = plugin.compute(h, noise)
    assert not invalid["is_photon_candidate"][0]
    assert not invalid["is_coincident_with_hits"][0]  # unknown, not measured coincidence
    assert np.all(invalid["is_photon_candidate"][1:])


@pytest.mark.parametrize("basis", ["raw", "ma", "convolved"])
def test_substrate_skips_invalid_timing_without_aborting(basis, caplog):
    st, _, h, noise, classifier, _ = produce()
    data = classifier.compute(h, noise)
    data["kappa"] = 38
    data["best_chi2"] = 2
    data["n_spikes_coinciding"] = 10
    data["is_coincident_with_spikes"] = True
    data["is_symmetric_spike"] = False
    data["is_invalid_kappa"] = False
    st.set_config(
        {
            "substrate_seed_coincidence": "record_threshold",
            "substrate_time_basis": basis,
            "substrate_min_hit_coincidence": 1,
        }
    )
    plugin = st.get_single_plugin("run_A", "substrate_events")
    expected = plugin.compute(data[1:])
    data[f"peak_{basis}_valid"][0] = False
    np.testing.assert_array_equal(plugin.compute(data), expected)
    assert f"Excluded 1 substrate seeds with invalid {basis}" in caplog.text
    assert len(data) == len(h)


def test_seed_mode_fallback_matches_plugin_default():
    st, _, h, noise, classifier, _ = produce()
    data = classifier.compute(h, noise)
    data["legacy_spike_coincidence_valid"] = False
    plugin = st.get_single_plugin("run_A", "substrate_events")
    config = dict(plugin.config)
    del config["substrate_seed_coincidence"]
    np.testing.assert_array_equal(
        plugin.select_substrate_hits(data, config),
        plugin.select_substrate_hits(data, plugin.config),
    )


def test_export_retains_rows_when_all_group_views_fail(tmp_path, monkeypatch):
    from straxion.export_peak_contract import export_peak_contract

    st, _, h, noise, classifier, _ = produce()
    data = classifier.compute(h, noise)
    data["kappa"] = 38
    data["best_chi2"] = 2
    data["is_symmetric_spike"] = False
    data["is_invalid_kappa"] = False
    data["is_coincident_with_spikes"] = True
    data["n_spikes_coinciding"] = 10
    data["amplitude"][0] = np.nan
    st.set_config({"substrate_min_peak_channels": 1})
    st.register(SyntheticRecords)
    monkeypatch.setattr(st, "get_array", lambda run, target: data)
    manifest = export_peak_contract(st, "run_A", tmp_path / "export", source_revision="synthetic")
    for name in (
        "groups_start",
        "legacy_groups_start",
        "fixed_start_membership_raw",
        "fixed_start_membership_ma",
        "fixed_start_membership_convolved",
        "substrate_events",
    ):
        assert name in manifest["unavailable_views"]
    saved = np.load(tmp_path / "export" / "hit_classification.npy", allow_pickle=False)
    assert len(saved) == len(data)
    assert np.isnan(saved["amplitude"][0])


def test_export_asset_hashes_are_required_and_never_overwritten(tmp_path):
    from straxion.processing_provenance import (
        verify_export_assets,
        template_sha256,
        file_sha256,
    )

    template = tmp_path / "default.pkl"
    template.write_bytes(b"template")
    folder = tmp_path / "templates"
    folder.mkdir()
    own = folder / "channel.pkl"
    own.write_bytes(b"own-template")
    averaged = tmp_path / "avg.npy"
    averaged.write_bytes(b"averaged-map")
    config = {
        "template_interp_path": str(template),
        "template_interp_folder": str(folder),
        "dx_map_dir": str(tmp_path),
        "dx_map_filename": averaged.name,
    }
    with pytest.raises(ValueError, match="requires an expected template"):
        verify_export_assets(config)
    config["template_content_sha256"] = template_sha256(template, folder)
    with pytest.raises(ValueError, match="requires an expected dx_map"):
        verify_export_assets(config)
    config["dx_map_sha256"] = file_sha256(averaged)
    expected = config.copy()
    verify_export_assets(config)
    own.write_bytes(b"changed-template")
    with pytest.raises(ValueError, match="Template content differs"):
        verify_export_assets(config)
    own.write_bytes(b"own-template")
    averaged.write_bytes(b"changed-map")
    with pytest.raises(ValueError, match="Averaged map differs"):
        verify_export_assets(config)
    assert config == expected


def test_template_default_is_pinned_and_verified(monkeypatch):
    import importlib
    from straxion.constants import (
        DEFAULT_TEMPLATE_INTERP_PATH,
        TEMPLATE_INTERP_FOLDER,
    )
    from straxion.processing_provenance import template_sha256, DEFAULT_TEMPLATE_CONTENT_SHA256

    assert template_sha256(DEFAULT_TEMPLATE_INTERP_PATH, TEMPLATE_INTERP_FOLDER) == (
        DEFAULT_TEMPLATE_CONTENT_SHA256
    )
    module = importlib.import_module("straxion.plugins.hit_classification")

    def forbid_hash(*args):
        raise AssertionError("Template contents read during configuration resolution")

    monkeypatch.setattr(module, "template_sha256", forbid_hash)
    first = context().key_for("run_A", "hit_classification")
    second = context().key_for("run_A", "hit_classification")
    assert first.lineage_hash == second.lineage_hash
    assert first.lineage["hit_classification"][2]["template_content_sha256"] == (
        DEFAULT_TEMPLATE_CONTENT_SHA256
    )
    # Setup still checks actual contents on every new classifier instance.
    monkeypatch.setattr(module, "template_sha256", lambda *args: "changed-assets")
    with pytest.raises(ValueError, match="template_content_sha256"):
        context().get_single_plugin("run_A", "hit_classification")


def test_export_effective_masks_reproduce_split_chain(tmp_path, monkeypatch):
    import json
    from straxion.export_peak_contract import export_peak_contract

    st, _, h, noise, classifier, _ = produce()
    data = classifier.compute(h, noise)
    data["kappa"] = 38
    data["best_chi2"] = 2
    data["is_symmetric_spike"] = False
    data["is_invalid_kappa"] = False
    data["is_coincident_with_spikes"] = True
    data["n_spikes_coinciding"] = 10
    times = data["record_time"][0] + 1_000_000 + np.arange(len(data)) * 1000
    for basis in ("raw", "ma", "convolved"):
        data[f"peak_{basis}_time"] = times
    data["time"] = times - 500
    data["endtime"] = times + 500
    data["peak_ma_valid"][4] = False
    counts, _ = hit_coincidences(data)
    data["hit_coincidence"] = counts["n_hits"]
    data["hit_coincidence_channels"] = counts["n_channels"]
    st.set_config(
        {
            "substrate_seed_coincidence": "record_threshold",
            "substrate_coincidence_window_ms": 0.001,
            "substrate_min_hit_coincidence": 1,
        }
    )
    st.register(SyntheticRecords)
    monkeypatch.setattr(st, "get_array", lambda run, target: data)
    destination = tmp_path / "export"
    export_peak_contract(st, "run_A", destination, source_revision="synthetic")
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["schema_version"] == 2
    assert "groups_ma" in manifest["unavailable_views"]
    saved = np.load(destination / "hit_classification.npy", allow_pickle=False)
    original = np.load(destination / "substrate_seed_mask.npy", allow_pickle=False)
    assert np.all(original)
    selection = manifest["event_seed_selection"]
    assert selection["active_basis"] == "ma"
    for basis, info in selection["by_basis"].items():
        mask = np.load(destination / info["mask_file"], allow_pickle=False)
        excluded = np.load(destination / info["excluded_hit_ids_file"], allow_pickle=False)
        expected = original.copy()
        if basis == "ma":
            expected[4] = False
        np.testing.assert_array_equal(mask, expected)
        np.testing.assert_array_equal(excluded, saved["hit_id"][original & ~expected])
        assert info["n_selected"] == np.count_nonzero(expected)
        assert info["n_excluded_invalid_timing"] == len(excluded)
    mask = np.load(destination / selection["by_basis"]["ma"]["mask_file"])
    groups, _ = prompt_groups(saved, mask, basis="ma", gap_ns=1000, min_hits=1)
    accepted = groups[apply_veto(groups, 0)["accepted"]]
    events = np.load(destination / "substrate_events.npy", allow_pickle=False)
    assert len(events) == 2  # Removing the middle seed splits the chain.
    np.testing.assert_array_equal(events["group_id"], accepted["group_id"])
    np.testing.assert_array_equal(events["n_hits"], accepted["n_hits"])
    np.testing.assert_array_equal(events["brightest_member_time"], accepted["brightest_time"])


def test_export_map_directory_uses_plugin_default(tmp_path, monkeypatch):
    from straxion.processing_provenance import verify_export_assets, template_sha256, file_sha256
    from straxion.constants import DEFAULT_TEMPLATE_INTERP_PATH, TEMPLATE_INTERP_FOLDER

    monkeypatch.chdir(tmp_path)
    path = tmp_path / "avg.npy"
    path.write_bytes(b"map")
    config = {
        "template_content_sha256": template_sha256(
            DEFAULT_TEMPLATE_INTERP_PATH, TEMPLATE_INTERP_FOLDER
        ),
        "dx_map_filename": path.name,
        "dx_map_sha256": file_sha256(path),
    }
    verify_export_assets(config)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        verify_export_assets(config)


@pytest.mark.parametrize("sigma", [False, True])
def test_legacy_producer_has_no_config_paradigm_warning(sigma):
    import warnings

    st = context(
        hit_threshold_dx=0.5,
        hit_thresholds_sigma=None,
        hit_min_width=1 / 38000,
        compute_legacy_spike_coincidence=True,
        spike_threshold_dx=None if sigma else 0.5,
        spike_thresholds_sigma=5.0 if sigma else None,
    )
    plugin = st.get_single_plugin("run_A", "hits")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        hits = plugin.compute(records())
    assert np.all(hits["legacy_spike_coincidence_valid"])
    assert not any("mixing config paradigms" in str(w.message) for w in caught)


@pytest.mark.parametrize("window", [0, 1, 10, 1000])
def test_fast_coincidence_matches_brute_force(window):
    rng = np.random.default_rng(147)
    hits = table(rng.integers(0, 100, 150), rng.choice([0, 2, 40], 150))
    hits["record_time"] = rng.integers(0, 3, len(hits))
    hits["run_id"][::7] = b"B"
    hits["peak_ma_valid"] = rng.random(len(hits)) > 0.2
    counts, links = hit_coincidences(hits, window, with_links=True)
    expected_links = set()
    for i, h in enumerate(hits):
        mask = (
            hits["peak_ma_valid"]
            & h["peak_ma_valid"]
            & (hits["record_time"] == h["record_time"])
            & (hits["run_id"] == h["run_id"])
            & (np.abs(hits["peak_ma_time"] - h["peak_ma_time"]) <= window)
        )
        assert counts["n_hits"][i] == np.count_nonzero(mask)
        assert counts["n_channels"][i] == len(np.unique(hits["channel"][mask]))
        expected_links.update((h["hit_id"], member) for member in hits["hit_id"][mask])
    assert set(map(tuple, links)) == expected_links
    without_links, empty = hit_coincidences(hits, window)
    np.testing.assert_array_equal(counts, without_links)
    assert len(empty) == 0


@pytest.mark.parametrize("window", [0, 10, 2**64])
def test_coincidence_int64_boundaries(window):
    limits = np.iinfo(np.int64)
    hits = table([limits.min, limits.min + 10, 0, limits.max - 10, limits.max])
    counts, _ = hit_coincidences(hits, window)
    for i, t in enumerate(hits["peak_ma_time"]):
        mask = np.array([abs(int(x) - int(t)) <= window for x in hits["peak_ma_time"]])
        assert counts["n_hits"][i] == np.count_nonzero(mask)
        assert counts["n_channels"][i] == len(np.unique(hits["channel"][mask]))
