"""Content identities for calibration assets; pass these as tracked configuration."""

import hashlib
from pathlib import Path


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
