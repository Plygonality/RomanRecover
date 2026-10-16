"""Run an external detector without a shell and without hidden downloads."""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from romanrecoverer.errors import DataError, DetectorError


@dataclass
class DetectorRun:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    command_line: str


def render_template(template: str, mapping: dict[str, str]) -> list[str]:
    if "{detections}" not in template:
        raise DataError(
            "Detector command must include the {detections} placeholder so the output path is explicit."
        )
    if "{image}" not in template and "{truth}" not in template:
        raise DataError(
            "Detector command must include {image} or {truth} so the input path is explicit."
        )
    try:
        rendered = template.format(**mapping)
    except KeyError as exc:
        raise DataError(f"Detector command placeholder {exc} has no value.") from exc
    args = shlex.split(rendered)
    if not args:
        raise DataError("Detector command is empty.")
    return args


def run_detector(command: list[str], *, cwd: Path | None = None, timeout: float = 600) -> DetectorRun:
    if not command:
        raise DataError("Refusing to run an empty detector command.")
    # Drop variables that would point a library at an interactive credential helper.
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            cwd=str(cwd) if cwd else None,
            env=env,
            timeout=timeout,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise DetectorError(f"Detector timed out after {timeout} seconds: {command[0]}") from exc
    except OSError as exc:
        raise DetectorError(f"Could not start detector {command[0]!r}: {exc}") from exc
    return DetectorRun(
        command=list(command),
        returncode=int(completed.returncode),
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
        command_line=" ".join(shlex.quote(part) for part in command),
    )


def persist_run(run: DetectorRun, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "command.txt").write_text(run.command_line + "\n", encoding="utf-8")
    (out_dir / "stdout.txt").write_text(run.stdout, encoding="utf-8")
    (out_dir / "stderr.txt").write_text(run.stderr, encoding="utf-8")
    (out_dir / "exit_code.txt").write_text(str(run.returncode) + "\n", encoding="utf-8")
