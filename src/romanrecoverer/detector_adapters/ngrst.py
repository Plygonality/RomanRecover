"""Adapter for the NGRST-Discovery-Toolkit column contract and CLI.

``ngrst-discover`` (``ngrst.cli:main``) takes one positional image path. That
path is a ``.npz`` whose science array is named ``data``, or a Roman
``.asdf`` file. The process writes:

* ``<stem>_catalog.csv`` and ``<stem>_run.json`` under ``--processed-dir``
* candidate PNGs under ``--candidate-dir/<stem>/``

The catalogue columns written by ``ngrst.export`` include ``label``,
``x_centroid``, ``y_centroid``, ``snr``, and ``anomaly_score``. ``label`` is
the detection id. ``anomaly_score`` is copied and is not a discovery.
Synthetic truth from the toolkit uses ``x``, ``y``, ``amplitude``, ``sigma``,
and ``class``. ``amplitude`` and ``sigma`` are a Gaussian peak and width, not
calibrated fluxes or magnitudes.

This module does not implement a detector. When the executable or the
``ngrst`` package is absent, callers receive ``SKIPPED_DEPENDENCY``. A
command is built from ``--help`` only when that text shows a positional
``input`` together with ``--processed-dir`` and ``--candidate-dir``.
Otherwise the caller must pass ``--detector-cmd``.

``--top``, ``--threshold-sigma``, ``--n-pixels``, ``--kernel-sigma``,
``--min-snr``, ``--seed``, ``--no-refine-background``, ``--legacy-demo``,
``--reference``, and ``--match-radius`` stay at the toolkit defaults.
``--scene`` belongs to ``make_demo.py``, not to ``ngrst-discover``.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from romanrecoverer.errors import DataError, DependencyError, DetectorError
from romanrecoverer.schemas import STATUS_SKIPPED

REQUIRED_OPTIONS = ("--processed-dir", "--candidate-dir")


@dataclass(frozen=True)
class NgrstOutputs:
    """Files ``ngrst-discover`` writes for one input stem."""

    catalog: Path
    run_json: Path
    candidate_directory: Path


def probe_ngrst() -> dict[str, Any]:
    executable = shutil.which("ngrst-discover")
    spec = importlib.util.find_spec("ngrst")
    available = executable is not None or spec is not None
    payload: dict[str, Any] = {
        "package": "ngrst",
        "available": available,
        "executable": executable,
        "importable": spec is not None,
        "status": "AVAILABLE" if available else STATUS_SKIPPED,
        "inspected_repository": False,
        "repository_note": (
            "The adapter invokes ngrst-discover only when --help shows a positional input "
            "and both --processed-dir and --candidate-dir. The catalogue is "
            "<stem>_catalog.csv and the run record is <stem>_run.json."
        ),
    }
    if not available:
        payload["reason"] = "ngrst-discover is not on PATH and the ngrst package is not installed."
    return payload


def help_text(executable: str) -> str:
    completed = subprocess.run(
        [executable, "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return (completed.stdout or "") + "\n" + (completed.stderr or "")


def _option_in_help(help_blob: str, flag: str) -> bool:
    """True when ``flag`` is declared on its own help line, not merely mentioned."""

    for line in help_blob.splitlines():
        stripped = line.strip()
        if stripped.startswith(flag) and (len(stripped) == len(flag) or stripped[len(flag)].isspace()):
            return True
    return False


def _positional_input(help_blob: str) -> bool:
    """True when argparse lists a positional argument named ``input``."""

    in_positional = False
    for line in help_blob.splitlines():
        if line.strip().lower().startswith("positional arguments"):
            in_positional = True
            continue
        if not in_positional:
            continue
        if line and not line[0].isspace():
            break
        parts = line.split()
        if parts and parts[0] == "input":
            return True
    return False


def command_from_help(
    help_blob: str,
    *,
    image: str,
    processed_dir: str,
    candidate_dir: str,
    scene: str | None = None,
) -> list[str]:
    """Build an argument list using only the verified ``ngrst-discover`` interface."""

    missing: list[str] = []
    if not _positional_input(help_blob):
        missing.append("a positional input")
    for flag in REQUIRED_OPTIONS:
        if not _option_in_help(help_blob, flag):
            missing.append(flag)
    if missing:
        raise DependencyError(
            "The installed ngrst-discover help text does not show "
            + " and ".join(missing)
            + ". Refusing to guess the CLI. Pass --detector-cmd.",
            status="UNVERIFIED_INTERFACE",
        )
    command = [image, "--processed-dir", processed_dir, "--candidate-dir", candidate_dir]
    if scene is not None:
        if not _option_in_help(help_blob, "--scene"):
            raise DataError(
                "A --scene value was requested but the installed help text has no --scene flag."
            )
        command.extend(["--scene", scene])
    return command


def require_ngrst() -> dict[str, Any]:
    probe = probe_ngrst()
    if not probe["available"]:
        raise DependencyError(probe["reason"], status=STATUS_SKIPPED)
    if probe["executable"] is None:
        raise DependencyError(
            "The ngrst package is importable but ngrst-discover is not on PATH. "
            "RomanRecoverer does not call private Python functions whose signatures "
            "were not read from the installed package. Pass --detector-cmd.",
            status="UNVERIFIED_INTERFACE",
        )
    return probe


def ngrst_output_paths(image: str | Path, processed_dir: str | Path, candidate_dir: str | Path) -> NgrstOutputs:
    """Return the paths ``run_pipeline`` writes for this image stem."""

    stem = Path(image).stem
    processed = Path(processed_dir)
    candidates = Path(candidate_dir)
    return NgrstOutputs(
        catalog=processed / f"{stem}_catalog.csv",
        run_json=processed / f"{stem}_run.json",
        candidate_directory=candidates / stem,
    )


def load_run_record(path: Path) -> dict[str, Any]:
    """Read the JSON object written by ``write_run_record``."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DetectorError(f"Run record {path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise DetectorError(f"Run record {path} must be a JSON object.")
    return payload


def require_ngrst_outputs(outputs: NgrstOutputs) -> dict[str, Any]:
    """Fail when the process did not write the catalogue, run record, and candidate directory."""

    if not outputs.catalog.is_file():
        raise DetectorError(f"ngrst-discover did not write the catalogue {outputs.catalog}.")
    if not outputs.run_json.is_file():
        raise DetectorError(f"ngrst-discover did not write the run record {outputs.run_json}.")
    if not outputs.candidate_directory.is_dir():
        raise DetectorError(
            f"ngrst-discover did not create the candidate directory {outputs.candidate_directory}."
        )
    return load_run_record(outputs.run_json)


def resolve_run_json(detection_csv: Path) -> Path | None:
    """Find the run record that sits beside a toolkit catalogue."""

    name = detection_csv.name
    if name.endswith("_catalog.csv"):
        stem = name[: -len("_catalog.csv")]
        sibling = detection_csv.with_name(f"{stem}_run.json")
        if sibling.is_file():
            return sibling
    candidate = detection_csv.with_suffix(".json")
    if candidate.is_file():
        return candidate
    sibling = detection_csv.parent / "run.json"
    if sibling.is_file():
        return sibling
    return None
