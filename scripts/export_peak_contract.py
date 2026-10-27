"""Regenerate a scalar package using an explicit, local per-run config JSON."""

import argparse
import json
from pathlib import Path
import subprocess
import straxion
from straxion.export_peak_contract import export_peak_contract
from straxion.processing_provenance import template_sha256, file_sha256
from straxion.constants import DEFAULT_TEMPLATE_INTERP_PATH, TEMPLATE_INTERP_FOLDER


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path, help="New private regeneration cache")
    parser.add_argument("--output", required=True, type=Path, help="New export directory")
    parser.add_argument("--waveform-count", default=0, type=int)
    args = parser.parse_args()
    if args.cache.exists() or args.output.exists():
        parser.error("Cache and export destinations must both be new directories")
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True
    ).strip()
    if dirty:
        parser.error("Commit the implementation before regeneration so provenance is reproducible")
    config = json.loads(args.config.read_text())
    config["template_content_sha256"] = template_sha256(
        config.get("template_interp_path", DEFAULT_TEMPLATE_INTERP_PATH),
        config.get("template_interp_folder", TEMPLATE_INTERP_FOLDER),
    )
    if config.get("dx_map_filename"):
        config["dx_map_sha256"] = file_sha256(
            Path(config["dx_map_dir"]) / config["dx_map_filename"]
        )
    # The export comparison uses the requested offline gap while the plugin retains
    # its historical 0.2-ms gap default for callers that do not opt in.
    config.setdefault("substrate_coincidence_window_ms", 1.0)
    config.setdefault("substrate_veto_ms", 5.0)
    config.setdefault("substrate_time_basis", "ma")
    config.setdefault("compute_legacy_spike_coincidence", True)
    context = straxion.qualiphide_thz_offline(output_folder=str(args.cache))
    context.set_config(config)
    export_peak_contract(
        context,
        args.run_id,
        args.output,
        source_revision=revision,
        waveform_count=args.waveform_count,
    )


if __name__ == "__main__":
    main()
