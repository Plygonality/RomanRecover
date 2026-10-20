"""Adapter for the NGRST-Discovery-Toolkit column contract and CLI.

The column contract implemented here is:

* synthetic truth CSV: ``x``, ``y``, ``amplitude``, ``sigma``, ``class``
* ``ngrst-discover`` detection CSV: ``x_centroid``, ``y_centroid``, ``snr``,
  ``anomaly_score``, plus any extra columns, which are ignored
* a run JSON may sit beside the detection CSV and is hashed, not interpreted

``amplitude`` and ``sigma`` are the toolkit's Gaussian peak and width. They
are not calibrated astronomical fluxes or magnitudes.

This module does not implement a detector. When the ``ngrst-discover``
executable or the ``ngrst`` package is absent, callers receive
``SKIPPED_DEPENDENCY``. A command is built from the installed ``--help`` text
only when the help text actually contains an image flag and an output flag.
Otherwise the caller must pass an explicit ``--detector-cmd``.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path
from typing import Any

from romanrecoverer.errors import DataError, DependencyError
from romanrecoverer.schemas import STATUS_SKIPPED

IMAGE_FLAGS = ("--image", "--input-image", "--npz", "--input")
OUTPUT_FLAGS = ("--output", "--detections", "--out-csv", "--catalog")
RUN_JSON_FLAGS = ("--run-json", "--run", "--summary-json")
SCENE_FLAGS = ("--scene",)


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
            "The adapter follows the documented column contract and will only invoke a local "
            "executable whose --help text exposes an explicit image and output flag."
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


def command_from_help(
    help_blob: str,
    *,
    image: str,
    detections: str,
    run_json: str | None = None,
    scene: str | None = None,
) -> list[str]:
    """Build an argument list using only flags that appear in ``help_blob``."""

    image_flag = next((flag for flag in IMAGE_FLAGS if flag in help_blob), None)
    output_flag = next((flag for flag in OUTPUT_FLAGS if flag in help_blob), None)
    if image_flag is None or output_flag is None:
        raise DependencyError(
            "The installed ngrst-discover help text does not expose a verified image flag and "
            "output flag. Refusing to guess the CLI. Pass --detector-cmd with an explicit "
            "{image} and {detections} mapping.",
            status="UNVERIFIED_INTERFACE",
        )
    command = [image_flag, image, output_flag, detections]
    if run_json is not None:
        run_flag = next((flag for flag in RUN_JSON_FLAGS if flag in help_blob), None)
        if run_flag is not None:
            command.extend([run_flag, run_json])
    if scene is not None:
        scene_flag = next((flag for flag in SCENE_FLAGS if flag in help_blob), None)
        if scene_flag is None:
            raise DataError(
                "A --scene value was requested but the installed help text has no --scene flag."
            )
        command.extend([scene_flag, scene])
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


def resolve_run_json(detection_csv: Path) -> Path | None:
    candidate = detection_csv.with_suffix(".json")
    if candidate.is_file():
        return candidate
    sibling = detection_csv.parent / "run.json"
    if sibling.is_file():
        return sibling
    return None
