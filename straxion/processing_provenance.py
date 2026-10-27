"""Content identities for calibration assets; pass these as tracked configuration."""

import hashlib
from pathlib import Path

# Expected identity of the bundled default + per-channel templates. Update only
# when deliberately changing those assets; setup verifies the full file contents.
DEFAULT_TEMPLATE_CONTENT_SHA256 = "0b93bb85b9dd6fa6bb7412322f293ab10500bc86ea01f0dc94f6f824e88b81e2"


def file_sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def template_manifest(path, folder):
    files = {"default": file_sha256(path)}
    directory = Path(folder)
    if directory.is_dir():
        files.update({p.name: file_sha256(p) for p in sorted(directory.glob("*.pkl"))})
    return files


def template_sha256(path, folder):
    import json

    return hashlib.sha256(
        json.dumps(template_manifest(path, folder), sort_keys=True).encode()
    ).hexdigest()


def verify_export_assets(config):
    """Verify pinned regeneration assets without replacing the expected identities."""
    from straxion.constants import DEFAULT_TEMPLATE_INTERP_PATH, TEMPLATE_INTERP_FOLDER

    expected = config.get("template_content_sha256")
    if not expected:
        raise ValueError("Export config requires an expected template_content_sha256")
    actual = template_sha256(
        config.get("template_interp_path", DEFAULT_TEMPLATE_INTERP_PATH),
        config.get("template_interp_folder", TEMPLATE_INTERP_FOLDER),
    )
    if actual != expected:
        raise ValueError("Template content differs from the expected template_content_sha256")
    if config.get("dx_map_filename"):
        expected = config.get("dx_map_sha256")
        if not expected:
            raise ValueError("Export config requires an expected dx_map_sha256")
        actual = file_sha256(Path(config.get("dx_map_dir", "")) / config["dx_map_filename"])
        if actual != expected:
            raise ValueError("Averaged map differs from the expected dx_map_sha256")
