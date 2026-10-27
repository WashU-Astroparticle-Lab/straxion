# Completed-run offline production

`qualiphide_thz_offline` requires epoch-averaged calibration for records and all
downstream products by default. The online context is unchanged. Preparation
loads scans only; waveform processing follows once the maps are frozen.

## Normal production

```python
import straxion

st = straxion.qualiphide_thz_offline(
    input_folder="/data/completed_science_run",
    output_folder="/data/production",
    config={"template_interp_folder": "/data/templates_for_this_science_run"},
)

for run_id in st.run_ids:
    st.make(run_id, "hit_classification")
```

On first use this creates `production/calibration/manifest.json` and one map per
epoch. On subsequent calls it reuses the manifest, with exactly the same
acquisitions, exclusions and epoch membership. `st.make` reuses matching saved
products, allowing interrupted production to resume. Preparation never replaces
an existing calibration directory. An interrupted preparation without a finished
manifest requires a fresh directory.

The directory should contain one completed science run. Acquisition files use
`ts_38kHz-<timestamp>.npy` by default. Each scan timestamp needs matching
`iq_fine_z_*`, `iq_fine_f_*`, `iq_wide_z_*`, `iq_wide_f_*`, and `fres_*` npy files.
Ambiguous timestamps and incomplete assigned scans fail with an error. The latest
scan strictly before each acquisition is selected, matching the reprocessing
notebooks. No waveform files are loaded during preparation.

By default, acquisition **start-time** gaps strictly greater than six hours split
epochs, as in the existing notebooks. This is an initial grouping convention;
detector changes and known bad blocks require explicit overrides. Epochs with
fewer than three distinct scans warn and remain separate. Each distinct assigned
scan contributes once, regardless of the number of acquisitions it serves.

## Epochs and exclusions

Prepare explicitly to inspect the manifest before starting waveform processing:

```python
manifest = straxion.prepare_offline_calibration(
    "/data/completed_science_run",
    "/data/calibration_release_1",  # must not already exist
    exclude_runs=[1750000100],
    epochs={
        "early": [1750000200, 1750000300],
        "late": [1750090000, 1750090100],
    },
)
st = straxion.qualiphide_thz_offline(
    input_folder="/data/completed_science_run",
    calibration_manifest=manifest,
    output_folder="/data/production",
    config={"template_interp_folder": "/data/templates_for_this_science_run"},
)
```

The timestamps above are illustrative. Explicit epochs must cover every retained
acquisition exactly once. Exclusions are acquisition timestamps, not indices into
a run list. A scan used only by excluded acquisitions is omitted from the median;
if the same scan also serves retained acquisitions, it still contributes. Straxion
does not hardcode science-run names or known bad blocks.

Alternatively pass `calibration_options={"exclude_runs": [...], "gap_hours": 6.0}`
to the context on first use. Omit preparation options when resuming. A changed
selection, epoch definition or map recipe requires a new calibration directory.
New files added to the input directory do not extend an already frozen production.

## Calculation and identities

The builder uses `DxRecords`' own gain correction, circle fit, phase reference and
per-scan interpolation. It normalizes each scan to `(f - f(0)) / f(0)`, resamples
onto 2000 equally spaced phase points over the scans' common interval, then takes
the pointwise median. Production uses the **unsmoothed** median, as the adopted
notebooks do; the stored 16th/84th percentiles are scan scatter. Per-scan gain,
circle centre, rotation and phase reference remain per scan during production.

The manifest records scan membership and hashes, map hashes, exclusions, recipe
settings and source-code hashes. Paths in the manifest are relative filenames,
so the same release can be copied to another machine. Provide its local input
folder and manifest path to reconstruct the same processing keys.

The offline context verifies the selected scan and map before constructing
strax lineage, including when reading cached products. Frozen manifest or file
drift raises an error. Map, scan and template content identities enter records
lineage and therefore downstream lineage. Template directory paths are not part
of that identity. Map bytes are checked again when the plugin loads them.

Maps must have finite, matching arrays for all detector channels, strictly
increasing phase grids, and a common interval bracketing zero. Production does
not fall back to per-scan calibration for missing or invalid channels.

Template generation is separate: supply the run's prepared per-channel template
folder. The low-level bundled template defaults remain available, so omitting
this option still uses the bundled templates; straxion does not infer the correct
science-run templates. The default template and all per-channel overrides are
content-hashed. An explicitly supplied nonexistent template directory is an error.
Records and classification use the same pinned bundled-template identity from
`straxion.processing_provenance`. If `template_content_sha256` is supplied, the
context verifies it before lookup and never replaces an incorrect expected hash.

## Existing maps and legacy processing

Existing maps can be used without rebuilding them:

```python
st = straxion.qualiphide_thz_offline()
st.set_config({
    "dx_map_dir": "/data/calibration",
    "dx_map_filename": "dx_map_epoch.npz",
    # Supply the acquisition's ordinary DAQ and scan options as before.
})
```

The context computes the content hash before every lookup. For this manual mode,
changing map contents changes the processing key. To pin the file and reject
changes, additionally provide `dx_map_sha256`. For automatic run/scan selection
and frozen provenance, use a prepared manifest.

Explicit legacy processing is `qualiphide_thz_offline(calibration_mode="per_scan")`
with ordinary DAQ and scan configuration. This restores the per-scan calculation,
not historical cache keys: the new tracked options change lineage. Old products
are not overwritten or relabelled. Exact historical cache access needs the
historical software/configuration. New production has distinct keys even if it
shares a storage directory with old processing.

## Validation before full production

`tests/test_offline_calibration.py` uses synthetic scans and IQ waveforms to check
the notebook median recipe, per-scan calibration parity, overrides/exclusions,
strict validation, path portability, resume, content-dependent keys, and rejection
of changed calibration even when matching records are already stored. Existing
per-scan tests request that mode explicitly.

## Compatibility with PR #99

This branch integrates PR #99 at `c0ac5ec38c8a8fc567f739587d54521f53c57849`.
Classification retains that PR's `hits`/`noises` dependencies, acquisition-wide
PSD aggregation, native peak/support metadata, MA-peak coincidence default and
substrate reconstruction semantics. The avgmap workflow resolves the same
template hash for records and classification before lineage is constructed.

The public `export_peak_contract(st, run_id, destination, source_revision=...)`
function accepts a manifest-backed offline context directly. It exports the
scalar contract and the effective map/template identities selected for that run.
See [the peak contract](PEAK_CONTRACT.md) for export views and reconstruction
settings; its CLI explicitly chooses the 1-ms gap/5-ms veto comparison, while the
library's historical substrate gap/veto defaults remain unchanged.

For the standalone `scripts/export_peak_contract.py`, supply the existing per-run
JSON with pinned expected `template_content_sha256` and `dx_map_sha256`. The CLI
continues to reject missing or incorrect expected hashes before creating its
context. Its default calibration mode is averaged; a deliberate historical
per-scan export now requires `--calibration-mode per_scan`.

Combined synthetic tests cover actual avgmap waveform processing with per-run
templates through classification and scalar export, including validation
waveforms, matching hashes in both plugin lineages, preserved peak fields and
rejection of changed pinned templates even when classifications are cached.

The remaining data-machine check is to prepare maps from exactly the scans used
by an existing completed-run notebook, compare `x` and `y_med`, then process a few
acquisitions spanning epochs with identical settings and templates. Compare
records, hit counts, thresholds and classification fields before a full run.
The private full-run waveform data are not part of the local test suite.
