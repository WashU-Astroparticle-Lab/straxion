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
(default 1, including self). `is_coincident_with_spikes` remains separately named.
Set `classification_coincidence_mode="legacy_record"` and enable the upstream
legacy count to reproduce the old photon-selection flags. OF/shape estimators do
not depend on this choice.

`substrate_seed_coincidence="hit_peaks"` is the new seed default. It uses MA
`hit_coincidence_channels >= substrate_min_peak_channels` (default 5), retains
strict kappa/chi requirements and excludes symmetric spikes/invalid kappa. It does
not consult the legacy count or reuse `is_photon_candidate`. The explicit
`record_threshold` seed mode retains the old channel threshold and reconstructs
the old photon predicate from legacy flags, independently of the new photon flag.
It requires a valid upstream legacy count. No energy cut or new truncation veto
is introduced. Neither selection establishes that a group is physical.

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
entry point computes these identities from the selected files before requesting
any products. For manual contexts, use `template_sha256` and `file_sha256` from
`straxion.processing_provenance`, and update the tracked config whenever assets
change. Use the original per-SR own-template folder and avgmap configuration;
never substitute a default asset to make regeneration succeed.

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
