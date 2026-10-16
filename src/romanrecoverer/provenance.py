"""Hashes, software versions, and the derived-artifact manifest.

Inputs are read and hashed. This module never writes back to an input path.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from romanrecoverer import __version__
from romanrecoverer.schemas import MANIFEST_SCHEMA, SCHEMA_VERSION

CORE_PACKAGES = (
    "numpy",
    "scipy",
    "pandas",
    "astropy",
    "matplotlib",
    "photutils",
    "pyarrow",
    "romanisim",
    "roman_datamodels",
    "galsim",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path, *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def software_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {
        "romanrecoverer": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for name in CORE_PACKAGES:
        versions[name] = package_version(name)
    versions["git_revision"] = _git_revision()
    return versions


def _git_revision() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    revision = completed.stdout.strip()
    return revision or None


def file_record(path: Path, *, role: str) -> dict[str, Any]:
    path = Path(path)
    return {
        "role": role,
        "path": str(path),
        "name": path.name,
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_manifest(
    *,
    inputs: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_name": MANIFEST_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "immutable_inputs": True,
        "note": (
            "Inputs were hashed and not modified by RomanRecoverer. "
            "Derived artifacts are listed under outputs."
        ),
        "software": software_versions(),
        "inputs": inputs,
        "outputs": outputs,
    }
    if extra:
        manifest["extra"] = extra
    return manifest
