# SR5: process one completed CRS acquisition

Install in the processing environment:

```bash
conda activate straxion
python -m pip install -e '.[sr5]'
```

The `sr5` extra adds TensorStore's Zarr v3 reader. It coexists with the Zarr 2
dependency of strax 2.2.x; upgrading `zarr` itself to version 3 conflicts with
strax's declared requirements. The usual strax DataDirectory remains the output
format.

The DAQ macro supplies a completed, immutable group and the five scan files
actually used for that acquisition. It owns the decision about when to call
this function. No scan discovery or averaged calibration is involved.

```python
from straxion import qualiphide_sr5_online


def process_sr5_run(run_id, group_path, scan_files, sr3_templates, output_folder):
    st = qualiphide_sr5_online(
        group_path=group_path,
        scan_files=scan_files,
        template_interp_folder=sr3_templates,
        output_folder=output_folder,
        config=dict(
            substrate_time_basis="ma",
            substrate_seed_coincidence="hit_peaks",
            substrate_coincidence_window_ms=1.0,
            substrate_veto_ms=5.0,
        ),
    )
    # Classification, hits and noises are dependencies and are also saved.
    st.make(str(run_id), "substrate_events", progress_bar=False)
    return st


st = process_sr5_run(
    run_id="1700000060",
    group_path="/data/SR5_set-1700000000/timestreams-1700000000.zarr/ts_38kHz-1700000060",
    scan_files={
        "fine_f": "/data/SR5_set-1700000000/iq_fine_f_a0p5-1700000000.npy",
        "fine_z": "/data/SR5_set-1700000000/iq_fine_z_a0p5-1700000000.npy",
        "wide_f": "/data/SR5_set-1700000000/iq_wide_f_a0p5-1700000000.npy",
        "wide_z": "/data/SR5_set-1700000000/iq_wide_z_a0p5-1700000000.npy",
        "fres": "/data/SR5_set-1700000000/fres_a0p5-1700000000.npy",
    },
    sr3_templates="/calibration/sr3_template_interps",
    output_folder="/processed/sr5",
)
```

Use a unique acquisition ID for `run_id`; it need not encode timing. One Zarr
group is one run. The reader preserves its complete sample count, including
the last partial storage chunk. It converts int32 I/Q with the per-channel
`counts_to_s21` factors into the existing float32 normalized S21 record fields.
Reads are blocked to avoid a full complex128 temporary. Processing still holds
the full raw acquisition in memory, including off-resonance tones.

## Science channels and off-resonance tones

The total tone count is read from Zarr; there is no 61- or 63-channel setting to
enter. `raw_records` retains every acquired channel and its original channel ID.
The default science channel map remains **0–40**. Only this prefix enters
calibrated `records`, including common-mode removal, and therefore hits,
classification, noise-bank and substrate products. Simulated pulse injection
(`truth`) also selects only these science channels. A group missing a required
science channel is rejected.

Each fine/wide scan pair and the scan `fres` file may contain either 41 rows or
all acquired tones, with the science channels first in the same order as Zarr.
Only the science rows are calibrated and checked against their requested and
actual tone frequencies. Extra tones do not require resonator scans or templates;
unused trailing scan rows are ignored. Off-resonance monitoring is not yet
implemented, but those tones remain available in `raw_records` for a future branch.

The processor call above is unchanged. For small synthetic acquisitions, an
explicit `immutabledict(kids=(0, n_science_channels - 1))` channel map selects a
smaller science prefix. Its size is tracked in the science plugins' lineage.

## Native timing and calibration

- `fs = 1 / dt` and record length come from the group, before strax constructs
  dtypes and cache keys. Conflicting explicit overrides are rejected. For the
  pilot, `fs = 38146.97265625 Hz`, with 2,021,789 samples per acquisition and
  52.9999855616 seconds of exposure. There is no resampling or truncation.
- Group `timestamp` is California wall-clock time (`America/Los_Angeles`) to
  one second, converted to UTC nanoseconds. The group name is only an identity.
  A timestamp in an ambiguous or nonexistent DST hour raises an error because
  the recorded string alone cannot determine the instant.
- Strax's legacy `dt` field remains integer nanoseconds. **Use `fs` or the hit
  `sampling_frequency_hz` field for precise sample timing**, not `length * dt`.
  Fractional rates use a float64 `sampling_frequency_hz` field. Integer-rate
  contexts retain their original integer schema and processing defaults.
- Scan `fres` must exactly match the science prefix of the group's `fres_requested`,
  including channel order. The per-scan calibration uses the actual dithered `fres` as
  its operating frequencies. Existing offline hit/noise/classification and
  substrate algorithms are reused, with explicitly supplied SR3 templates.
- Metadata, scan contents and template contents enter processing lineage.
  Local paths are untracked so identical inputs can be relocated. Waveform
  bytes are not hashed; callers must not replace or extend a processed group.
  Missing consumed payloads raise errors rather than becoming zero fill.

Raw records and calibrated full records are not saved unless explicitly
requested. Repeating `st.make` with unchanged inputs reuses matching products.
The science-record plugin version and SR5-specific truth plugin invalidate older
downstream cache keys; existing files are retained, and the next processing call
builds products under the updated lineage.

## Tests

```bash
conda activate straxion
python -m pip install -e '.[sr5]' pytest
pytest -q
```

The SR5 tests build small Zarr v3 acquisitions with sharding, Zstandard and
CRC32C indices. They exercise native timing, scaling, chunk boundaries, missing
payload rejection, explicit calibration, content identities and an end-to-end
classification/substrate pass without private DAQ data. Paired 41/61/63-tone
captures verify identical science outputs after appending large extra-tone
signals, including PCA, salting, noise and substrate processing. Both scan layouts
and invalid science-channel/scan selections are covered.
