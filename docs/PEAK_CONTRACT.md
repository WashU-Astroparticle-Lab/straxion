# Peak timing and local substrate reconstruction

The offline dependency graph is now:

```
records -> hits (DxHits)
records + hits -> noises (NoiseBank)
hits + noises -> hit_classification -> substrate_events
```

The offline classifier no longer depends on records. Optimal-filter fitting and
per-channel noise PSD estimation remain in `DxHitClassification`; no PSD vector
is copied into each hit. The online theta pipeline is unchanged. Hits retain their
four waveform arrays and carry the native timing/support metadata and actual-hit
coincidence counts. An optional upstream calculation supplies legacy record
coincidence for comparisons. It is disabled by default and is not a prerequisite
for new coincidence or selection.

## Scalar contract

`hit_classification` retains all previous amplitudes, OF estimates, kappa, tau,
coincidence and quality fields (the new defaults are described below), plus:

- `run_id`, `record_time`, `record_endtime`, `candidate_record_i`, and `hit_id`.
  The ID is SHA256 of run identity, native record origin, channel, and original
  above-threshold candidate start index. It is independent of output row order.
  The exported lineage must accompany IDs: changing hit finding may change the
  candidate set. Separate acquisitions/runs never share clustering or veto state.
- `sampling_frequency_hz`, `alignment_record_i`, `alignment_time`, and
  `peak_{raw,ma,convolved}_{record_i,time,valid}`. Times are int64 ns, computed as
  `record_time + floor(record_index * 1e9 / fs)` without converting the epoch to
  floating point. Original hit `time`/`endtime` are retained unchanged.
- The producer alignment is the convolved maximum in the above-threshold
  candidate. The three amplitude maxima are searched over the clipped extraction
  support; they can differ from alignment. Boundary flags identify maxima at
  the first or last copied sample. A valid maximum can still be a tail.
- Half-open native and waveform-array support bounds, and left/right sample
  counts around alignment and each peak. Counts exclude the reference sample;
  valid physical zeros count as support. "Clean" means not padded or cut off by
  the producer's neighboring-candidate boundaries, **not proven pulse isolation**.
- `intact25_alignment` requires 25 copied samples on both sides of alignment.
  This is an analysis flag, never a substrate seed veto. Separately named
  `legacy_nonzero_*_alignment` counts reproduce the historical contiguous-nonzero
  convention, which can stop at a physical zero. Neither definition certifies
  that a tail is a separate pulse.
- PSD fallback flag, noise-window count and SHA256 of the actual float64 PSD
  used for the hit's channel. PSDs remain the mean of the same per-channel
  noise-window FFT powers over the same complete native acquisition. OF templates,
  weighting, shifts, amplitudes, chi-square and kappa estimators are unchanged.

Unsupported record layouts fail explicitly: legacy coincidence/noise estimators
require one synchronous record per contiguous channel in channel order. Do not
combine multiple acquisitions into a records compute batch.

## Two different coincidence estimators

With `compute_legacy_spike_coincidence=True`, `n_spikes_coinciding` retains the
legacy convolved **record-threshold** count, including its historical negative-slice behavior at the acquisition's left edge.
It does not establish a separate detected pulse. Otherwise its value is -1 and
`legacy_spike_coincidence_valid=False`; it must not be interpreted as zero.
`is_coincident_with_spikes` is False when unavailable and is meaningful only
where `legacy_spike_coincidence_valid` is True. For new analyses use
`is_coincident_with_hits`; the tutorial now uses this field.
Its configured `spike_coincidence_window` is a half-width rounded to samples. The code default
0.131 ms at 38 kHz rounds to 5 samples and uses `[peak_i-5:peak_i+5)`: ten samples,
about 0.263 ms total. These are code defaults, not verified production overrides.

`hit_coincidence` counts **all detected hits** with valid moving-average peaks in
`[reference_peak-window, reference_peak+window]`. `hit_coincidence_channels`
counts distinct channels in that neighborhood. Self is included; another hit on
the same channel increases only the hit count. Invalid MA peaks neither contribute
nor receive a count. There is no quality, energy, truncation or old-coincidence
prerequisite. The tracked `hit_coincidence_window_ns` is an independent half-width
(default 131000 ns), not the clustering gap. The records-based hit producer sees the
whole acquisition, avoiding chunk-edge neighborhoods. `hit_coincidences(...,
with_links=True)` reconstructs directed ID links from the scalar table.
Scalar counts use vectorized window bounds and a compiled sliding channel tally;
each member enters and leaves the tally at most once per acquisition. The first
call may incur Numba compilation/cache-load time. Explicit link generation still
scales with the number of neighborhood pairs.

## Groups, events and comparison modes

Moving-average peak time is the new `substrate_time_basis` default. `start` is the
explicit legacy comparison; `raw` and `convolved` are additional choices. The
plugin retains the historical 0.2 ms gap and zero veto defaults. The export
entry point explicitly selects the offline analysis settings: 1 ms gap, at least
5 seed hits, and 5 ms veto, and exports independent 0/5/50 ms views.

Consecutive gaps **equal to** the gap join. Single-link chains may span much more
than the gap. Five hits do not imply five channels. Qualifying groups accept when
`first_member >= previous_accepted_last_member + veto`; equality accepts.
Rejected and subthreshold groups never extend the veto. Every pre-veto group,
including subthreshold groups, and its hit-ID/input-index linkage is available
from `SubstrateEvents.reconstruct` and exported for alternate local scans.

`classification_coincidence_mode="hit_peaks"` is the new photon-selection default:
`is_coincident_with_hits` tests total actual-hit count against `max_hit_coincidence`
(default 1, including self). Invalid MA timing excludes the reference hit from
photon selection even when its count is zero; `peak_ma_valid` identifies this
unknown state. Counts describe valid MA peaks only: an invalid neighbor is not
included, so they are not a guarantee of isolation from every detected hit.
`is_coincident_with_spikes` remains separately named.
For a cut flow in `hit_peaks` mode, include unknown timing explicitly:

```python
unknown_coincidence = ~hits["peak_ma_valid"]
rejected = (
    unknown_coincidence
    | hits["is_coincident_with_hits"]
    | hits["is_symmetric_spike"]
    | hits["is_truncated_hit"]
    | hits["is_invalid_kappa"]
)
# is_photon_candidate == ~rejected; individual rejection counts can overlap.
```

The tutorial plots unknown MA timing separately from measured coincidence.
Set `classification_coincidence_mode="legacy_record"` and enable the upstream
legacy count to reproduce the old photon-selection flags. OF/shape estimators do
not depend on this choice.

`substrate_seed_coincidence="hit_peaks"` is the new seed default. It uses MA
`hit_coincidence_channels >= substrate_min_peak_channels` (default 5), retains
strict kappa/chi requirements and excludes symmetric spikes/invalid kappa. It does
not consult the legacy count or reuse `is_photon_candidate`. The explicit
`record_threshold` seed mode retains the old channel threshold and reconstructs
the old photon predicate from legacy flags, independently of the new photon flag.
It requires a valid upstream legacy count. A config dictionary omitting the
seed-mode key uses the same `hit_peaks` default as the plugin; callers using old
tables must explicitly select `record_threshold`. No energy cut or new truncation veto
is introduced. Neither selection establishes that a group is physical.

Normal event reconstruction excludes selected seeds whose requested peak time
is invalid and logs the number excluded and the time basis. The scalar table and
seed-selection predicate retain those rows. Exported comparisons deliberately keep
the seed mask frozen: if a basis cannot represent every selected seed, that view
is marked unavailable instead of silently changing the comparison population.

Events retain original window-support envelopes for framework `time`/`endtime`,
are sorted by those starts, and additionally report first/last/brightest member
basis times, group and brightest-hit IDs, hit count, channel count, repeated-channel
count and maximum hits/channel. Per-channel patterns still use the brightest raw
amplitude representative. Start mode retains stable input tie order; peak modes
sort equal times by stable hit ID, with the first maximum winning brightest-member
ties. This deterministic peak tie rule can differ from legacy tied anchors.

The substrate plugin buffers **scalar** classifications to run EOF, then groups separately
per acquisition. This removes the invalid finite-overlap assumption for unbounded
chains and overlapping hit windows. Cost: memory proportional to run hit count
and output latency until EOF. Ordinary runs are supported; reconstruct before
combining runs into superruns. Arbitrarily fragmented legal chunks give the same
answer. Full streaming with a peak-order watermark is not implemented.

Classification separately buffers aligned **hit waveforms and noise windows** to
EOF, then estimates PSDs per native acquisition origin. This preserves the old
whole-record PSD scope even if cached hits and noises have different chunk
boundaries. The base iterator drains both dependencies together to avoid bounded
mailbox deadlocks. Partial acquisition coverage and superrun input are rejected
when computing new classifications. This stage has waveform-sized memory costs;
the current reader already represents a full acquisition per run. Streaming
acquisition flushes are future optimization work.

## Lineage and migration

Producer, noise, classifier, calibration and event versions invalidate old
products. Noise windows now also carry native acquisition time/endtime. New schema rows
must be regenerated; old starts cannot be converted into
native peaks by adding fixed padding. Tracked `template_content_sha256` covers the
default template and every per-channel template file; `dx_map_sha256` is required
when an averaged map is selected. Setup verifies supplied digests. The export
entry point requires an expected
`template_content_sha256` in the per-run JSON and an expected `dx_map_sha256`
whenever an averaged map is selected; missing or mismatched values fail before
processing. It never replaces these expected values. Pin hashes from the original
processing provenance or an independently verified asset set. A newly calculated
hash identifies today's files but does not establish their historical identity.
For manual contexts, use `template_sha256` and `file_sha256` from
`straxion.processing_provenance` when deliberately selecting a new asset set.
The bundled-template expected digest is pinned in `DEFAULT_TEMPLATE_CONTENT_SHA256`
and included in lineage. Import and configuration/key resolution do not read
those files. Setup still hashes their full contents before estimation and rejects
changed assets; no file-metadata cache is used. When intentionally updating bundled
templates, update the digest in the same change (the tests verify it against the
shipped files). Custom assets require their own explicit tracked digest.
Use the original per-SR own-template folder and avgmap configuration;
never substitute a default asset to make regeneration succeed.

Export schema version 2 adds `event_seed_mask_{start,raw,ma,convolved}.npy`
and corresponding `event_excluded_hit_ids_*` arrays. Each effective mask is the
frozen primary seed mask AND the validity of that basis (all rows are timing-valid
in `start` mode). The manifest's `event_seed_selection` identifies the active
basis, file names, retained count and invalid-timing exclusion count for every
basis. To reproduce `substrate_events.npy`, use the active effective mask with
`prompt_groups` and the saved gap/minimum/veto settings. Removing an invalid seed
can split a chain; the exported masks make this change explicit. These masks do
not replace the frozen `substrate_seed_mask` used for timing-only comparisons.

`export_peak_contract` writes the complete scalar table, directed coincidence
links, frozen seed mask, groups/members for four time bases, independent veto
views, fixed-start-membership peak summaries, explicit legacy-record seed views when
available, event summaries, optional validation
waveforms, and a JSON manifest with effective configurations, lineage, asset hashes,
source revision, array schemas and output checksums. Invalid peak metadata never
removes rows from the scalar export: an unavailable alternate view and its reason
are recorded in the manifest. Directed coincidence links can be quadratic in a
pathologically dense neighborhood; scalar counts and peak timestamps are also
available for independent reconstruction. Run-specific paths and
operational instructions belong in a local runbook/wiki, not a public PR.

Synthetic regression tests cover the unchanged classification fields (including
PSD fallback and per-channel templates), clipped/neighbor-overlapping windows,
physical zeros, alignment/peak offsets, exact gap/veto boundaries, repeated
channels, long chains, acquisition isolation, chunk fragmentation, stable IDs,
lineage and export round trips. Representative data-machine waveform validation
remains necessary before interpreting peak maxima or follower features physically.
