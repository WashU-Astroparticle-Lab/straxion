"""Export scalar inputs and auditable reconstruction views for local analysis."""

import json
from pathlib import Path
import numpy as np
from straxion.processing_provenance import template_manifest, file_sha256, template_sha256
from straxion.substrate_reconstruction import hit_coincidences, timing, apply_veto


def export_peak_contract(context, run_id, destination, *, source_revision, waveform_count=0):
    """Regenerate/export one complete run into a new directory (never overwrite).

    Call with the original per-run calibration configuration and a fresh writable
    cache. All detected hits are exported; reconstruction outputs are independent
    views and do not restrict the scalar table. No production storage is modified.
    """
    if waveform_count < 0:
        raise ValueError("waveform_count must be nonnegative")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    upstream = context.get_single_plugin(run_id, "hit_classification")
    assets = template_manifest(
        upstream.config["template_interp_path"], upstream.config["template_interp_folder"]
    )
    if (
        template_sha256(
            upstream.config["template_interp_path"], upstream.config["template_interp_folder"]
        )
        != upstream.config["template_content_sha256"]
    ):
        raise ValueError("Template manifest mismatch")
    hits = context.get_array(run_id, "hit_classification")
    plugin = context.get_single_plugin(run_id, "substrate_events")
    counts, links = hit_coincidences(
        hits, upstream.config["hit_coincidence_window_ns"], with_links=True
    )
    if not (
        np.array_equal(counts["n_hits"], hits["hit_coincidence"])
        and np.array_equal(counts["n_channels"], hits["hit_coincidence_channels"])
    ):
        raise ValueError("Exported coincidence differs from full-acquisition reconstruction")
    arrays = {"hit_classification": hits, "hit_coincidence_members": links}
    unavailable = {}
    _add_reconstruction_views(hits, plugin, arrays, unavailable)
    if waveform_count:
        arrays["validation_waveforms"] = _validation_waveform_subset(
            context, run_id, hits, waveform_count
        )
    configs, lineage, map_hash = _processing_manifest(context, run_id)
    destination.mkdir(parents=True)
    for name, array in arrays.items():
        np.save(destination / f"{name}.npy", array, allow_pickle=False)
    manifest = {
        "schema_version": 1,
        "run_id": str(run_id),
        "source_revision": source_revision,
        "n_detected_hits": len(hits),
        "unavailable_views": unavailable,
        "config": configs,
        "lineage": lineage,
        "template_files_sha256": assets,
        "dx_map_sha256": map_hash,
        "default_time_basis": plugin.config["substrate_time_basis"],
        "hit_coincidence_basis": "ma",
        "hit_coincidence_endpoints": "inclusive; self included",
        "seed_mask_frozen_across_time_bases": True,
        "support_counts_reference": (
            "alignment and named peaks; excludes center; physical zeros valid"
        ),
        "validation_status": (
            "Generated package; representative waveform/real-data review still required"
        ),
        "files": {
            f"{name}.npy": {
                "sha256": file_sha256(destination / f"{name}.npy"),
                "rows": len(array),
                "dtype": array.dtype.descr,
            }
            for name, array in arrays.items()
        },
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=_json_value) + "\n"
    )
    return manifest


def _add_reconstruction_views(hits, plugin, arrays, unavailable):
    # Freeze the selected seed mask for start-vs-peak comparisons; basis changes
    # only membership/time, even when a peak-based seed alternative was selected.
    seed_mask = plugin.select_substrate_hits(hits, plugin.config)
    arrays["substrate_seed_mask"] = seed_mask
    if np.all(hits["legacy_spike_coincidence_valid"]):
        legacy_config = dict(plugin.config, substrate_seed_coincidence="record_threshold")
        legacy_mask = plugin.select_substrate_hits(hits, legacy_config)
        arrays["legacy_record_seed_mask"] = legacy_mask
        _add_group_view(hits, legacy_mask, plugin, arrays, unavailable, "start", "legacy_")
    for basis in ("start", "raw", "ma", "convolved"):
        _add_group_view(hits, seed_mask, plugin, arrays, unavailable, basis)
    _add_fixed_membership_views(hits, arrays, unavailable)
    try:
        arrays["substrate_events"] = plugin.compute(hits)
    except ValueError as error:
        unavailable["substrate_events"] = str(error)


def _add_group_view(hits, mask, plugin, arrays, unavailable, basis, prefix=""):
    from straxion.substrate_reconstruction import prompt_groups

    try:
        groups, members = prompt_groups(
            hits,
            mask,
            basis=basis,
            gap_ns=plugin.coincidence_window_ns,
            min_hits=plugin.min_hit_coincidence,
        )
        arrays[f"{prefix}groups_{basis}"] = groups
        arrays[f"{prefix}members_{basis}"] = members
        for veto_ms in (0, 5, 50):
            arrays[f"{prefix}veto_{basis}_{veto_ms}ms"] = apply_veto(groups, veto_ms * 1_000_000)
    except ValueError as error:
        unavailable[f"{prefix}groups_{basis}"] = str(error)


def _add_fixed_membership_views(hits, arrays, unavailable):
    # First comparison: identical start-time membership, alternate timing only.
    if "groups_start" not in arrays:
        for basis in ("raw", "ma", "convolved"):
            unavailable[f"fixed_start_membership_{basis}"] = (
                "Start membership unavailable: " + unavailable.get("groups_start", "missing view")
            )
        return
    groups, members = arrays["groups_start"], arrays["members_start"]
    by_group = {}
    for m in members:
        by_group.setdefault(m["group_id"], []).append(m["hit_index"])
    for basis in ("raw", "ma", "convolved"):
        try:
            fixed = groups.copy()
            for group in fixed:
                ix = np.asarray(by_group[group["group_id"]])
                times = timing(hits[ix], basis)
                group["first_time"], group["last_time"] = times.min(), times.max()
                anchor = ix[np.flatnonzero(hits["hit_id"][ix] == group["brightest_hit_id"])[0]]
                group["brightest_time"] = hits[f"peak_{basis}_time"][anchor]
            arrays[f"fixed_start_membership_{basis}"] = fixed
        except ValueError as error:
            unavailable[f"fixed_start_membership_{basis}"] = str(error)


def _validation_waveform_subset(context, run_id, hits, waveform_count):
    waveforms = context.get_array(run_id, "hits")
    # Deterministic validation subset, including boundary/invalid/intactness cases.
    special = np.flatnonzero(
        ~hits["intact25_alignment"]
        | hits["peak_ma_at_left_boundary"]
        | hits["peak_ma_at_right_boundary"]
    )
    uniform = np.linspace(0, max(0, len(hits) - 1), min(waveform_count, len(hits)), dtype=int)
    ix = np.unique(np.r_[special[:waveform_count], uniform])
    wanted = set(hits["hit_id"][ix])
    return waveforms[np.array([x in wanted for x in waveforms["hit_id"]], bool)]


def _processing_manifest(context, run_id):
    configs = {}
    lineage = {}
    for name in (
        "records",
        "noises",
        "hits",
        "hit_classification",
        "substrate_events",
    ):
        configs[name] = dict(context.get_single_plugin(run_id, name).config)
        key = context.key_for(run_id, name)
        lineage[name] = {"hash": key.lineage_hash, "lineage": key.lineage}
    map_config = configs["records"]
    map_hash = None
    if map_config.get("dx_map_filename"):
        map_hash = file_sha256(Path(map_config["dx_map_dir"]) / map_config["dx_map_filename"])
        if map_hash != map_config["dx_map_sha256"]:
            raise ValueError("Averaged map manifest mismatch")
    return configs, lineage, map_hash


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "items"):
        return dict(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")
