"""Run-aware offline context: resolve calibration before strax builds lineage."""

from copy import copy
import hashlib
import json
from pathlib import Path

import strax

from .calibration import SCHEMA_VERSION, read_dx_map, scan_config, verify_file
from .processing_provenance import template_sha256, DEFAULT_TEMPLATE_CONTENT_SHA256


@strax.takes_config(
    strax.Option("calibration_mode", default="averaged", type=str),
    strax.Option("calibration_manifest", default="", type=str),
    strax.Option("calibration_manifest_sha256", default="", type=str),
    strax.Option("science_run_input", default="", type=str),
)
class OfflineContext(strax.Context):
    """Completed-run context with strict averaged calibration by default.

    Paths are resolved before plugin lineage or storage lookup. No shared config
    is mutated when switching acquisitions. Plugin caching is disabled because
    external calibration files must be verified even on repeated storage reads.
    """

    def _plugins_are_cached(self, targets, chunk_number=None):
        return False

    def _plugins_to_cache(self, plugins, chunk_number=None):
        pass

    @property
    def run_ids(self):
        """Acquisitions included in the frozen production, in timestamp order."""
        manifest = self._manifest()
        if manifest is None:
            raise ValueError("run_ids requires a calibration manifest")
        return sorted(manifest["runs"], key=int)

    def _manifest(self):
        path = self.context_config["calibration_manifest"]
        if not path:
            return None
        content = Path(path).read_bytes()
        expected = self.context_config["calibration_manifest_sha256"]
        if not expected or hashlib.sha256(content).hexdigest() != expected:
            raise ValueError("Calibration manifest changed or is not pinned; recreate the context")
        manifest = json.loads(content)
        if manifest["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Unsupported calibration manifest schema")
        return manifest

    def _resolved_config(self, run_id, plugin):
        cfg = self.config.copy()
        mode = self.context_config["calibration_mode"]
        if mode not in ("averaged", "per_scan"):
            raise ValueError("calibration_mode must be 'averaged' or 'per_scan'")
        manifest = self._manifest()
        if manifest is not None:
            if mode != "averaged":
                raise ValueError("A frozen avgmap manifest requires calibration_mode='averaged'")
            if run_id not in manifest["runs"]:
                raise ValueError(f"Acquisition {run_id} is absent or excluded from the manifest")
            base = Path(self.context_config["science_run_input"])
            if not self.context_config["science_run_input"]:
                raise ValueError("A calibration manifest requires input_folder")
            run = manifest["runs"][run_id]
            scan = manifest["scans"][run["scan"]]
            epoch = manifest["epochs"][run["epoch"]]
            settings = manifest["settings"]
            owned = dict(
                scan_config(base, scan["files"]),
                daq_input_dir=str(base / run["filename"]),
                dx_map_dir=str(Path(self.context_config["calibration_manifest"]).parent),
                dx_map_filename=epoch["filename"],
                dx_map_sha256=epoch["sha256"],
                fs=settings["fs"],
                widescan_resolution=settings["widescan_resolution"],
                cable_correction_polyfit_order=settings["cable_correction_polyfit_order"],
            )
            for key, value in owned.items():
                if key in cfg and cfg[key] != value:
                    raise ValueError(f"{key} conflicts with the frozen calibration manifest")
            cfg.update(owned)
            if "dx_map_sha256" in plugin.takes_config:
                for name, digest in scan["sha256"].items():
                    verify_file(base / name, digest)
                cfg["scan_calibration_sha256"] = hashlib.sha256(
                    json.dumps(scan["sha256"], sort_keys=True).encode()
                ).hexdigest()

        if "dx_map_sha256" in plugin.takes_config:
            filename = cfg.get("dx_map_filename", "")
            if mode == "per_scan":
                if filename:
                    raise ValueError("per_scan mode cannot also specify an averaged map")
            else:
                if not filename:
                    raise ValueError(
                        "Offline production requires an averaged map. Supply input_folder to "
                        "prepare a completed run, a calibration_manifest, or dx_map_dir and "
                        "dx_map_filename. For legacy processing use calibration_mode='per_scan'."
                    )
                if cfg.get("dx_map_require_all") is False:
                    raise ValueError(
                        "Offline avgmap production requires a valid map for every channel"
                    )
                path = Path(cfg.get("dx_map_dir", "")) / filename
                channel_map = cfg["channel_map"]
                n_channels = max(bounds[1] for bounds in channel_map.values()) + 1
                _, _, digest = read_dx_map(path, cfg.get("dx_map_sha256", ""), n_channels)
                cfg.update(dx_map_sha256=digest, dx_map_require_all=True)
        # PR #99's classifier has its own tracked template identity and no direct
        # records dependency. Resolve the same identity for every template consumer.
        if "template_content_sha256" in plugin.takes_config:
            digest = self._template_hash(cfg, plugin)
            expected = cfg.get("template_content_sha256")
            if not expected and not any(
                key in cfg for key in ("template_interp_path", "template_interp_folder")
            ):
                expected = DEFAULT_TEMPLATE_CONTENT_SHA256
            if expected and expected != digest:
                raise ValueError(
                    "Template content differs from the expected template_content_sha256"
                )
            cfg["template_content_sha256"] = digest
        return cfg

    @staticmethod
    def _template_hash(cfg, plugin):
        """Track the default template and the per-channel overrides, independent of paths."""
        names = ("template_interp_path", "template_interp_folder")
        path, folder = [Path(cfg.get(k, plugin.takes_config[k].default)) for k in names]
        if "template_interp_folder" in cfg and not folder.is_dir():
            raise FileNotFoundError(f"Template folder not found: {folder}")
        return template_sha256(path, folder)

    def _set_plugin_config(self, p, run_id, tolerant=True):
        # strax calls this both before lineage creation and before plugin setup.
        # Use a shallow context copy to preserve normal option validation without
        # changing self.config during multi-run iteration or concurrent readers.
        resolved = copy(self)
        resolved.config = self._resolved_config(run_id, p)
        previous_config = p.config.copy() if hasattr(p, "lineage") else {}
        strax.Context._set_plugin_config(resolved, p, run_id, tolerant=tolerant)
        if hasattr(p, "lineage"):
            for key in ("dx_map_sha256", "scan_calibration_sha256", "template_content_sha256"):
                previous = previous_config.get(key, "")
                if previous and previous != p.config.get(key, ""):
                    raise ValueError(
                        f"{key} changed after lineage creation; retry with stable files"
                    )

    def new_context(
        self,
        storage=tuple(),
        config=None,
        register=None,
        register_all=None,
        processors=None,
        replace=False,
        **kwargs,
    ):
        """Preserve calibration resolution in strax's per-call context copies."""
        if not isinstance(storage, (list, tuple)):
            storage = [storage]
        config = {} if config is None else config
        if not replace:
            storage = self.storage + list(storage)
            config = strax.combine_configs(self.config, config, mode="update")
            kwargs = strax.combine_configs(self.context_config, kwargs, mode="update")
        result = type(self)(storage=storage, config=config, processors=processors, **kwargs)
        if not replace:
            result._plugin_class_registry = self._plugin_class_registry.copy()
        result.register_all(register_all)
        result.register([] if register is None else register)
        return result
