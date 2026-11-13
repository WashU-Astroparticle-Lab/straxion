"""SR5 regression coverage using small, genuinely sharded CRS-like acquisitions."""

from pathlib import Path
import shutil
import json

from immutabledict import immutabledict
import numpy as np
import pytest

import straxion
from straxion.constants import TEMPLATE_INTERP_FOLDER
from straxion.hit_metadata import sample_offset_ns, timing_dtype
from straxion.sr5 import open_acquisition
from tests.test_offline_calibration import write_scan

ts = pytest.importorskip("tensorstore")
FS = 38146.97265625


def make_acquisition(tmp_path, n_channels=1, scan_channels=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "capture.zarr" / "ts_38kHz-100"
    path.mkdir(parents=True)
    group = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {"timestamp": "20261112,03:58:18"},
    }
    (path / "zarr.json").write_text(json.dumps(group))
    counts = np.stack(
        [
            np.random.default_rng(19 + ch).integers(-10000, 10000, size=(2, 10001), dtype=np.int32)
            for ch in range(n_channels)
        ],
        axis=1,
    )
    counts[0] += 50_000_000
    # Large, deliberately unphysical extra-tone signals must never enter science processing.
    counts[:, 41:] *= 30
    bytes_codec = {"name": "bytes", "configuration": {"endian": "little"}}
    z = ts.open(
        dict(
            driver="zarr3",
            kvstore=dict(driver="file", path=str(path / "z") + "/"),
            store_data_equal_to_fill_value=True,
            metadata=dict(
                shape=list(counts.shape),
                data_type="int32",
                chunk_grid=dict(name="regular", configuration=dict(chunk_shape=[2, 1, 2048])),
                codecs=[
                    dict(
                        name="sharding_indexed",
                        configuration=dict(
                            chunk_shape=[1, 1, 1024],
                            codecs=[bytes_codec, {"name": "zstd"}],
                            index_codecs=[bytes_codec, {"name": "crc32c"}],
                            index_location="end",
                        ),
                    )
                ],
            ),
        ),
        create=True,
    ).result()
    z.write(counts).result()
    for name, values in dict(
        dt=1 / FS,
        counts_to_s21=np.full(n_channels, 1e-8),
        fres=np.full(n_channels, 1e9 + 25),
        fres_requested=np.full(n_channels, 1e9),
    ).items():
        data = np.asarray(values)
        a = ts.open(
            dict(
                driver="zarr3",
                kvstore=dict(driver="file", path=str(path / name) + "/"),
                store_data_equal_to_fill_value=True,
                metadata=dict(shape=list(data.shape), data_type="float64"),
            ),
            create=True,
        ).result()
        a.write(data).result()
    write_scan(tmp_path, 90)
    files = {
        k: str(tmp_path / (f"iq_{k}_test-90.npy" if k != "fres" else "fres_test-90.npy"))
        for k in ("fine_f", "fine_z", "wide_f", "wide_z", "fres")
    }
    for filename in files.values():
        values = np.load(filename)
        np.save(filename, np.repeat(values, scan_channels or n_channels, axis=0))
    st = straxion.qualiphide_sr5_online(
        group_path=path,
        scan_files=files,
        template_interp_folder=TEMPLATE_INTERP_FOLDER,
        output_folder=tmp_path / "processed",
        config=dict(channel_map=immutabledict(kids=(0, min(n_channels, 41) - 1))),
    )
    return st, path, counts


@pytest.fixture
def acquisition(tmp_path):
    return make_acquisition(tmp_path)


def test_full_sharded_acquisition_and_fractional_time(acquisition):
    st, _, counts = acquisition
    raw = st.get_array("100", "raw_records", progress_bar=False)
    np.testing.assert_array_equal(raw["data_i"], (counts[0] * 1e-8).astype("f4"))
    np.testing.assert_array_equal(raw["data_q"], (counts[1] * 1e-8).astype("f4"))
    assert raw["length"].tolist() == [10001]
    # California 03:58 is UTC 11:58; do not derive timing from the group name.
    start = int(np.datetime64("2026-11-12T11:58:18", "ns").astype("i8"))
    assert raw["time"][0] == start
    assert raw["endtime"][0] == start + int(10001 * 1e9 / FS)
    assert st.get_single_plugin("100", "raw_records").config["fs"] == FS
    # Metadata resolution must also survive strax's per-call context copies.
    assert str(st.new_context().key_for("100", "hits")) == str(st.key_for("100", "hits"))


@pytest.mark.parametrize("filename", ["z/c/0/0/0", "counts_to_s21/c/0", "dt/c"])
def test_missing_payloads_are_errors_not_fill_values(acquisition, filename):
    st, _, _ = acquisition
    (Path(st.config["sr5_group_path"]) / filename).unlink()
    with pytest.raises(ValueError, match="Missing or empty Zarr payload"):
        st.key_for("100", "raw_records")


@pytest.mark.parametrize("key,value", [("fs", 38000), ("record_length", 10000)])
def test_reject_native_timing_override(acquisition, key, value):
    st, _, _ = acquisition
    with pytest.raises(ValueError, match="conflicts with the native"):
        st.new_context(config={key: value}).key_for("100", "hits")


def test_scan_association_and_actual_tones(acquisition):
    st, _, _ = acquisition
    records = st.get_single_plugin("100", "records")
    np.testing.assert_array_equal(records.fres, [1e9 + 25])
    np.save(st.config["sr5_scan_files"]["fres"], [1.01e9])
    with pytest.raises(ValueError, match="fres_requested"):
        st.get_single_plugin("100", "records")


def test_content_identity_not_local_paths(acquisition, tmp_path):
    st, group, _ = acquisition
    original = str(st.key_for("100", "records"))
    other = tmp_path / "moved"
    other.mkdir()
    moved = {}
    for role, filename in st.config["sr5_scan_files"].items():
        target = other / Path(filename).name
        shutil.copyfile(filename, target)
        moved[role] = str(target)
    relocated = st.new_context(config=dict(sr5_scan_files=moved))
    assert str(relocated.key_for("100", "records")) == original
    changed = np.load(moved["fine_z"]) * 1.01
    np.save(moved["fine_z"], changed)
    assert str(relocated.key_for("100", "records")) != original
    a = ts.open(
        dict(driver="zarr3", kvstore=dict(driver="file", path=str(group / "fres") + "/"))
    ).result()
    a.write([1e9 + 26]).result()
    assert str(st.key_for("100", "records")) != original


def set_timestamp(path, timestamp):
    meta = json.loads((path / "zarr.json").read_text())
    meta["attributes"]["timestamp"] = timestamp
    (path / "zarr.json").write_text(json.dumps(meta))


def test_metadata_change_after_lineage_is_rejected(acquisition):
    st, group, _ = acquisition
    reader = st.get_single_plugin("100", "raw_records")
    set_timestamp(group, "20261112,03:58:19")
    with pytest.raises(ValueError, match="metadata changed"):
        reader.compute()


@pytest.mark.parametrize("timestamp", ["20261101,01:30:00", "20260308,02:30:00"])
def test_ambiguous_or_missing_dst_hour_rejected(acquisition, timestamp):
    import pytz

    st, group, _ = acquisition
    set_timestamp(group, timestamp)
    with pytest.raises((pytz.AmbiguousTimeError, pytz.NonExistentTimeError)):
        open_acquisition(st.config["sr5_group_path"])


def test_native_hit_times_keep_nanoseconds_and_legacy_schema():
    indices = np.array([0, 1, 10001, 2021789])
    np.testing.assert_array_equal(
        sample_offset_ns(indices, FS), np.floor(indices * 26214.4).astype("i8")
    )
    np.testing.assert_array_equal(
        sample_offset_ns(indices, 38000), indices * 1_000_000_000 // 38000
    )
    assert np.dtype(timing_dtype(38000))["sampling_frequency_hz"] == np.dtype("i8")
    assert np.dtype(timing_dtype(FS))["sampling_frequency_hz"] == np.dtype("f8")


def test_processing_through_classification_and_substrate(acquisition):
    st, _, _ = acquisition
    st.set_config(
        dict(
            salt_rate=30,
            energy_resolution_mode="none",
            hit_threshold_dx=1e-7,
            hit_thresholds_sigma=None,
            hit_min_width=1 / FS,
            substrate_time_basis="ma",
            substrate_seed_coincidence="hit_peaks",
        )
    )
    st.make("100", "substrate_events", progress_bar=False)
    hits = st.get_array("100", "hit_classification", progress_bar=False)
    assert len(hits) > 0
    assert np.all(hits["sampling_frequency_hz"] == FS)
    np.testing.assert_array_equal(
        hits["peak_ma_time"], hits["record_time"] + sample_offset_ns(hits["peak_ma_record_i"], FS)
    )
    assert st.is_stored("100", "substrate_events")


@pytest.fixture(scope="module")
def science_products(tmp_path_factory):
    """A 41-channel reference, including PCA and reproducible nonzero salting."""
    st, _, _ = make_acquisition(tmp_path_factory.mktemp("sr5_science"), n_channels=41)
    st.set_config(
        dict(
            salt_rate=200,
            energy_resolution_mode="none",
            hit_threshold_dx=1e-7,
            hit_thresholds_sigma=None,
            hit_min_width=1 / FS,
            pca_n_components=1,
            substrate_time_basis="ma",
            substrate_seed_coincidence="hit_peaks",
            substrate_coincidence_window_ms=1.0,
            substrate_veto_ms=5.0,
        )
    )
    targets = (
        "raw_records",
        "records",
        "truth",
        "hits",
        "noises",
        "hit_classification",
        "substrate_events",
    )
    st.make("100", "substrate_events", save=targets, progress_bar=False)
    products = {t: st.get_array("100", t, progress_bar=False) for t in targets}
    assert len(products["truth"]) > 0
    assert len(products["hits"]) > 0
    return st.config, products


@pytest.mark.parametrize("n_channels,scan_channels", [(61, 41), (63, 63)])
def test_extra_tones_do_not_change_science(tmp_path, science_products, n_channels, scan_channels):
    config, expected = science_products
    st, _, counts = make_acquisition(tmp_path, n_channels, scan_channels)
    st.set_config({k: v for k, v in config.items() if not k.startswith("sr5_")})
    if scan_channels > 41:
        # Off-resonance tones need no usable resonator scans, even when rows are present.
        for filename in st.config["sr5_scan_files"].values():
            scan = np.load(filename)
            scan[41:] = np.nan
            np.save(filename, scan)
    st.make("100", "substrate_events", save=tuple(expected), progress_bar=False)
    for target, reference in expected.items():
        actual = st.get_array("100", target, progress_bar=False)
        if target == "raw_records":
            assert len(actual) == n_channels
            np.testing.assert_array_equal(actual["channel"], np.arange(n_channels))
            np.testing.assert_array_equal(actual["data_i"], (counts[0] * 1e-8).astype("f4"))
            np.testing.assert_array_equal(actual["data_q"], (counts[1] * 1e-8).astype("f4"))
            actual = actual[:41]
        elif "channel" in actual.dtype.names:
            assert np.all((actual["channel"] >= 0) & (actual["channel"] < 41))
        # Compare each field to handle NaNs in diagnostics correctly.
        assert actual.dtype == reference.dtype and actual.shape == reference.shape
        for field in actual.dtype.names:
            np.testing.assert_array_equal(
                actual[field], reference[field], err_msg=f"{target}.{field}"
            )


@pytest.mark.parametrize("bounds", [(0, 40), (1, 1), (0, -1)])
def test_missing_or_nonprefix_science_channels_rejected(acquisition, bounds):
    st, _, _ = acquisition
    st.set_config(dict(channel_map=immutabledict(kids=bounds)))
    with pytest.raises(ValueError, match="available Zarr channel prefix"):
        st.key_for("100", "records")


@pytest.mark.parametrize("role", ["fine_f", "wide_z", "fres"])
def test_science_scan_mismatch_rejected_with_extra_tones(tmp_path, role):
    st, _, _ = make_acquisition(tmp_path, 63, 41)
    path = st.config["sr5_scan_files"][role]
    np.save(path, np.load(path)[:-1])
    with pytest.raises(ValueError, match="scan shape|fres_requested"):
        st.get_single_plugin("100", "records")


def test_science_selection_is_tracked(tmp_path):
    st, _, _ = make_acquisition(tmp_path, 63, 63)
    before = {target: str(st.key_for("100", target)) for target in ("records", "truth", "hits")}
    st.set_config(dict(channel_map=immutabledict(kids=(0, 39))))
    for target, key in before.items():
        assert str(st.key_for("100", target)) != key
    assert st.get_single_plugin("100", "truth").config["sr5_science_channels"] == 40
