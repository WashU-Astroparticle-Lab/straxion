"""Completed-run production tests independent of private experiment data."""

import json
from pathlib import Path
import shutil

from immutabledict import immutabledict
import numpy as np
import pytest
import strax
import straxion

from straxion.calibration import (
    average_maps,
    calibration_tables,
    file_sha256,
    prepare_offline_calibration,
    read_dx_map,
    scan_config,
)
from straxion.offline import OfflineContext
from straxion.plugins.records import DxRecords


def write_scan(folder, stamp, scale=1.0):
    f0 = 1e9
    fine = np.linspace(f0 + 2e5, f0 - 2e5, 101)[None, :]
    wide = np.linspace(f0 + 2e6, f0 - 2e6, 201)[None, :]

    def response(f):
        return 1 - 0.5 / (1 + 2j * 10000 * scale * (f - f0) / f0)

    arrays = dict(fine_z=response(fine), fine_f=fine, wide_z=response(wide), wide_f=wide)
    for name, array in arrays.items():
        np.save(folder / f"iq_{name}_test-{stamp}.npy", array)
    np.save(folder / f"fres_test-{stamp}.npy", [f0])


@pytest.fixture
def science_run(tmp_path):
    folder = tmp_path / "input"
    folder.mkdir()
    for stamp, scale in [(90, 1.0), (190, 1.04), (290, 0.98), (29990, 1.1)]:
        write_scan(folder, stamp, scale)
    for stamp in [100, 200, 300, 30000]:
        # Real miniature IQ waveforms for the end-to-end processing test.
        rng = np.random.default_rng(stamp)
        data = 0.5 + 1e-4 * (rng.normal(size=(1, 1000)) + 1j * rng.normal(size=(1, 1000)))
        np.save(folder / f"ts_38kHz-{stamp}.npy", data)
    with pytest.warns(UserWarning, match="weak median"):
        manifest = prepare_offline_calibration(folder, tmp_path / "calibration")
    return folder, manifest


def context(science_run, **kwargs):
    folder, manifest = science_run
    config = dict(channel_map=immutabledict(kids=(0, 0)), record_length=1000)
    config.update(kwargs.pop("config", {}))
    return straxion.qualiphide_thz_offline(
        input_folder=folder,
        calibration_manifest=manifest,
        output_folder=str(manifest.parent.parent / "processed"),
        config=config,
        **kwargs,
    )


def test_pointwise_median_matches_notebook_and_is_not_mean():
    x = np.array([-2.0, -0.3, 0.0, 0.2, 2.0])
    tables = [[(x, factor * (x + 0.1 * x**3))] for factor in [1.0, 1.1, 20.0]]
    actual = average_maps(tables, n_grid=2000)
    grid = np.linspace(-2, 2, 2000)
    ys = np.array([np.interp(grid, t[0][0], t[0][1]) for t in tables])
    np.testing.assert_array_equal(actual["x"][0], grid)
    np.testing.assert_array_equal(actual["y_med"][0], np.median(ys, axis=0))
    np.testing.assert_array_equal(actual["y16"][0], np.percentile(ys, 16, axis=0))
    assert not np.allclose(actual["y_med"][0], np.mean(ys, axis=0))


def test_frozen_maps_and_scan_plugin_parity(science_run):
    folder, path = science_run
    manifest = json.loads(path.read_text())
    assert manifest["epochs"]["e0"]["runs"] == [100, 200, 300]
    assert manifest["epochs"]["e1"]["runs"] == [30000]
    scan = manifest["scans"]["90"]
    cfg = dict(
        scan_config(folder, scan["files"]),
        widescan_resolution=1000,
        cable_correction_polyfit_order=5,
        dx_map_filename="",
    )
    tables = calibration_tables(cfg)
    plugin = DxRecords()
    plugin.config = cfg
    plugin._setup_iq_correction_and_calibration()
    plugin._setup_frequency_interpolation_models()
    np.testing.assert_array_equal(tables[0][0], plugin.interp_x_data[0])
    f0 = plugin.interpolated_freqs[0]
    np.testing.assert_array_equal(tables[0][1], (plugin.interp_y_data[0] - f0) / f0)
    # Independently reproduce the notebook's normalization/resampling/median.
    scans = [
        calibration_tables(
            dict(
                scan_config(folder, s["files"]),
                widescan_resolution=1000,
                cable_correction_polyfit_order=5,
            )
        )[0]
        for stamp, s in manifest["scans"].items()
        if stamp != "29990"
    ]
    grid = np.linspace(max(x.min() for x, _ in scans), min(x.max() for x, _ in scans), 2000)
    expected = np.median([np.interp(grid, x, y) for x, y in scans], axis=0)
    x, y, _ = read_dx_map(path.parent / manifest["epochs"]["e0"]["filename"])
    np.testing.assert_array_equal(x[0], grid)
    np.testing.assert_array_equal(y[0], expected)
    with pytest.raises(FileExistsError):
        prepare_offline_calibration(folder, path.parent)


def test_explicit_epochs_exclusions_and_boundary(science_run, tmp_path):
    folder, _ = science_run
    with pytest.warns(UserWarning):
        path = prepare_offline_calibration(
            folder,
            tmp_path / "override",
            exclude_runs=[100],
            epochs={"early": [200], "late": [300, 30000]},
        )
    data = json.loads(path.read_text())
    assert data["excluded_runs"] == [100]
    assert "90" not in data["scans"]
    assert "100" not in data["runs"]
    assert data["epochs"]["late"]["scans"] == [290, 29990]
    with pytest.raises(ValueError, match="exactly once"):
        prepare_offline_calibration(folder, tmp_path / "bad", epochs={"x": [100, 100]})
    with pytest.raises(ValueError, match="Excluded acquisitions not found"):
        prepare_offline_calibration(folder, tmp_path / "bad", exclude_runs=[123])
    # A gap exactly equal to the setting is retained in the same epoch.
    with pytest.warns(UserWarning):
        path = prepare_offline_calibration(folder, tmp_path / "boundary", gap_hours=100 / 3600)
    assert json.loads(path.read_text())["epochs"]["e0"]["runs"] == [100, 200, 300]


def test_incomplete_or_absent_scan_is_not_skipped(science_run, tmp_path):
    folder, _ = science_run
    np.save(folder / "fres_test-95.npy", [1e9])
    with pytest.raises(ValueError, match="Incomplete scan 95"):
        prepare_offline_calibration(folder, tmp_path / "incomplete")
    (folder / "fres_test-95.npy").unlink()
    np.save(folder / "ts_38kHz-80.npy", np.zeros((1, 1000)))
    with pytest.raises(ValueError, match="No scan strictly before"):
        prepare_offline_calibration(folder, tmp_path / "absent")


@pytest.mark.parametrize("bad", ["nan", "descending", "shape", "zero", "channels"])
def test_invalid_maps_fail(tmp_path, bad):
    x = np.linspace(-1, 1, 11)[None, :]
    y = x * 1e-5
    if bad == "nan":
        y[0, 0] = np.nan
    elif bad == "descending":
        x = x[:, ::-1]
    elif bad == "shape":
        y = y[:, :-1]
    elif bad == "zero":
        x += 2
    path = tmp_path / "map.npz"
    np.savez(path, x=x, y_med=y)
    with pytest.raises(ValueError):
        read_dx_map(path, n_channels=2 if bad == "channels" else 1)


def test_default_requires_avgmap_and_legacy_is_explicit(tmp_path):
    st = straxion.qualiphide_thz_offline(output_folder=str(tmp_path / "default"))
    with pytest.raises(ValueError, match="requires an averaged map"):
        st.key_for("100", "records")
    legacy = straxion.qualiphide_thz_offline(
        output_folder=str(tmp_path / "legacy"), calibration_mode="per_scan"
    )
    assert legacy.key_for("100", "records")
    assert isinstance(st.new_context(config={"record_length": 1000}), OfflineContext)
    with pytest.raises(ValueError, match="requires an averaged map"):
        st.is_stored("100", "records", record_length=1000)


def test_run_resolution_copy_and_portability(science_run, tmp_path):
    st = context(science_run)
    assert st.run_ids == ["100", "200", "300", "30000"]
    first = st.key_for("100", "hit_classification")
    second = st.key_for("30000", "hit_classification")
    assert (
        first.lineage["records"][2]["dx_map_sha256"]
        != second.lineage["records"][2]["dx_map_sha256"]
    )
    assert str(st.key_for("100", "hit_classification")) == str(first)
    assert str(st.new_context().key_for("100", "hit_classification")) == str(first)
    assert "daq_input_dir" not in st.config
    with pytest.raises(ValueError, match="absent or excluded"):
        st.key_for("101", "records")
    with pytest.raises(ValueError, match="conflicts"):
        st.new_context(config={"fs": 1000}).key_for("100", "records")
    moved = tmp_path / "moved"
    shutil.copytree(science_run[1].parent, moved / "calibration")
    shutil.copytree(science_run[0], moved / "input")
    other = context((moved / "input", moved / "calibration" / "manifest.json"))
    assert str(other.key_for("100", "hit_classification")) == str(first)


def test_frozen_map_and_scan_drift_rejected_before_lookup(science_run):
    st = context(science_run)
    st.key_for("100", "records")
    folder, manifest = science_run
    data = json.loads(manifest.read_text())
    path = manifest.parent / data["epochs"]["e0"]["filename"]
    original = path.read_bytes()
    x, y, _ = read_dx_map(path)
    np.savez(path, x=x, y_med=y * 1.01)
    with pytest.raises(ValueError, match="Calibration file changed"):
        st.is_stored("100", "records")
    path.write_bytes(original)
    scan_file = folder / data["scans"]["90"]["files"]["fres"]
    np.save(scan_file, [1.1e9])
    with pytest.raises(ValueError, match="Calibration file changed"):
        st.is_stored("100", "records")


def test_manual_map_content_changes_lineage_and_rechecks_at_setup(science_run):
    folder, manifest = science_run
    data = json.loads(manifest.read_text())
    path = manifest.parent / data["epochs"]["e0"]["filename"]
    st = straxion.qualiphide_thz_offline(
        output_folder=str(manifest.parent / "manual"),
        config=dict(
            channel_map=immutabledict(kids=(0, 0)),
            record_length=1000,
            dx_map_dir=str(path.parent),
            dx_map_filename=path.name,
        ),
    )
    before = st.key_for("100", "hit_classification")
    x, y, _ = read_dx_map(path)
    np.savez(path, x=x, y_med=y * 1.1)
    after = st.key_for("100", "hit_classification")
    assert str(before) != str(after)
    plugin = DxRecords()
    plugin.config = dict(
        dx_map_dir=str(path.parent),
        dx_map_filename=path.name,
        dx_map_sha256=before.lineage["records"][2]["dx_map_sha256"],
    )
    plugin.fres = np.array([1e9])
    with pytest.raises(ValueError, match="Calibration file changed"):
        plugin._apply_averaged_dx_map()


def test_manifest_edits_are_not_adopted_mid_production(science_run):
    st = context(science_run)
    path = science_run[1]
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="manifest changed"):
        st.key_for("100", "records")


def test_template_contents_enter_lineage(science_run, tmp_path):
    folder = tmp_path / "templates"
    folder.mkdir()
    path = folder / "ch0_template_interp.pkl"
    path.write_bytes(b"template-v1")  # key_for does not unpickle templates
    st = context(science_run, config={"template_interp_folder": str(folder)})
    before = st.key_for("100", "hit_classification")
    path.write_bytes(b"template-v2")
    assert str(st.key_for("100", "hit_classification")) != str(before)


def test_template_drift_between_lineage_and_setup_is_rejected(science_run, tmp_path):
    folder = tmp_path / "templates"
    folder.mkdir()
    path = folder / "ch0_template_interp.pkl"
    path.write_bytes(b"template-v1")
    st = context(science_run, config={"template_interp_folder": str(folder)})
    plugin = st._get_plugins(("records",), "100")["records"]
    path.write_bytes(b"template-v2")
    with pytest.raises(ValueError, match="changed after lineage creation"):
        st._set_plugin_config(plugin, "100", tolerant=False)


def test_prepare_and_resume_factory(science_run, tmp_path):
    folder, manifest = science_run
    output = tmp_path / "automatic"
    with pytest.warns(UserWarning):
        st = straxion.qualiphide_thz_offline(input_folder=folder, output_folder=str(output))
    frozen = (output / "calibration" / "manifest.json").read_bytes()
    resumed = straxion.qualiphide_thz_offline(input_folder=folder, output_folder=str(output))
    assert resumed.run_ids == st.run_ids
    assert (output / "calibration" / "manifest.json").read_bytes() == frozen
    with pytest.raises(ValueError, match="already frozen"):
        straxion.qualiphide_thz_offline(
            input_folder=folder, output_folder=str(output), calibration_options={"gap_hours": 12}
        )


def test_miniature_production_matches_plain_plugin_and_cached_drift_fails(science_run):
    st = context(science_run)
    result = st.get_array("100", "records", save="records", progress_bar=False)
    plugin = st._get_plugins(("records",), "100")["records"]
    reference_config = dict(st.config)
    for dependency in st._get_plugins(("records",), "100").values():
        reference_config.update(dependency.config)
    reference = strax.Context(storage=[], config=reference_config)
    reference.register(list(st._plugin_class_registry.values()))
    expected = reference.get_array("100", "records", progress_bar=False)
    np.testing.assert_array_equal(result, expected)
    assert st.is_stored("100", "records")
    # Existing cached records must not bypass manifest/map verification.
    path = Path(plugin.config["dx_map_dir"]) / plugin.config["dx_map_filename"]
    assert file_sha256(path) == plugin.config["dx_map_sha256"]
    x, y, _ = read_dx_map(path)
    np.savez(path, x=x, y_med=y * 1.2)
    with pytest.raises(ValueError, match="Calibration file changed"):
        st.get_array("100", "records", progress_bar=False)
