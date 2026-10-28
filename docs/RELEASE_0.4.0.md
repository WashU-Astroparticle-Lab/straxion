# straxion 0.4.0

This release brings completed-run averaged calibration, native hit timing and
support metadata, and substrate-event reconstruction into the offline pipeline.

- Offline production defaults to frozen epoch-averaged calibration maps. Scan,
  map and template content identities enter processing lineage; drift is rejected.
- `prepare_offline_calibration` and the run-aware offline context support explicit
  epochs and exclusions, portable manifests, and resumable processing.
- Classification consumes hits and acquisition-wide noise banks, preserving native
  peak times and clean support. Detected-hit MA-peak coincidence is the default.
- Substrate reconstruction supports explicit timing, seed, prompt-gap and veto
  settings, with retained cluster information and per-channel energy patterns.
- Records include dissipation-direction amplitude and pulse-decay fits; utilities
  add hit/run/sample lookup and detector-position trace plotting.

## Migration

Python 3.11 or newer is required. Install with `python -m pip install straxion==0.4.0`.
Use a new production directory: tracked options and plugin changes produce new
lineage, and existing cached products are not relabelled as this release.

Supply each science run's own template folder explicitly. Offline processing now
requires averaged maps; deliberate legacy processing needs
`calibration_mode="per_scan"`. Automatic gap-based epochs may differ from historical
notebook boundaries, so specify membership when reproducing those exact maps.

The substrate plugin defaults remain a 0.2-ms prompt gap and zero additional veto;
set these explicitly for the intended catalogue (for the new SR1–SR4 campaign:
1-ms prompt gap and 5-ms nonextending veto). Package version and individual plugin
versions are separate; this release does not change plugin versions just to match
the package version.

See [offline production](OFFLINE_PRODUCTION.md) and [peak contract](PEAK_CONTRACT.md)
for configuration, provenance and data-machine validation details.
