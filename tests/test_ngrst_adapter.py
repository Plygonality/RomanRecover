"""Command construction for the real ngrst-discover CLI, using a mocked program."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from romanrecoverer.cli import main
from romanrecoverer.detector_adapters.ngrst import (
    command_from_help,
    help_text,
    ngrst_output_paths,
    require_ngrst_outputs,
    resolve_run_json,
)
from romanrecoverer.errors import DataError, DependencyError, DetectorError

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

# Argparse definition copied from ngrst.cli.main. The mock's --help text has to
# match that definition, including help strings and defaults.
MOCK_NGRST = """#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="NGRST astronomical discovery toolkit")
    parser.add_argument(
        "input",
        type=Path,
        help="Input .npz test data or Roman .asdf file",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="Number of candidate objects to export",
    )
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--candidate-dir", type=Path, default=None)
    parser.add_argument("--threshold-sigma", type=float, default=4.0)
    parser.add_argument("--n-pixels", type=int, default=5)
    parser.add_argument("--kernel-sigma", type=float, default=1.2)
    parser.add_argument("--min-snr", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42, help="Isolation Forest seed.")
    parser.add_argument(
        "--no-refine-background",
        action="store_true",
        help="Fit the background once, without masking detected sources.",
    )
    parser.add_argument(
        "--legacy-demo",
        action="store_true",
        help="Detect with 8 connected pixels and no smoothing kernel.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Local reference CSV, or a directory of CSV files. Offline only.",
    )
    parser.add_argument(
        "--match-radius",
        type=float,
        default=1.0,
        help="Cross-match radius in arcseconds.",
    )
    args = parser.parse_args()
    if args.processed_dir is None or args.candidate_dir is None:
        raise SystemExit("mock ngrst-discover requires --processed-dir and --candidate-dir")
    processed = args.processed_dir
    processed.mkdir(parents=True, exist_ok=True)
    stem = Path(args.input).stem
    catalog = processed / f"{stem}_catalog.csv"
    catalog.write_text(
        "label,x_centroid,y_centroid,area,segment_flux,max_value,eccentricity,"
        "semimajor_axis,semiminor_axis,orientation,anomaly_score,snr,"
        "anomaly_percentile,ra,dec,known,reference_id\\n"
        "4,3.5,4.5,10,1.0,0.2,0.1,1.0,1.0,0.0,0.4,9.0,80,,,,\\n",
        encoding="utf-8",
    )
    run_path = processed / f"{stem}_run.json"
    run_path.write_text(
        json.dumps(
            {
                "input_path": str(args.input),
                "seed": int(args.seed),
                "threshold_sigma": args.threshold_sigma,
                "n_pixels": args.n_pixels,
                "kernel_sigma": args.kernel_sigma,
                "min_snr": args.min_snr,
            },
            indent=2,
            sort_keys=True,
        )
        + "\\n",
        encoding="utf-8",
    )
    candidate_directory = args.candidate_dir / stem
    candidate_directory.mkdir(parents=True, exist_ok=True)
    (candidate_directory / "001_object_4.png").write_bytes(b"")


if __name__ == "__main__":
    main()
"""

UNKNOWN_HELP = """#!/usr/bin/env python3
import sys

if "--help" in sys.argv or "-h" in sys.argv:
    print("usage: ngrst-discover --image IMAGE --output DETECTIONS")
    raise SystemExit(0)
raise SystemExit("unknown interface should not be executed")
"""

HELP_SNIPPETS = (
    "Input .npz test data or Roman .asdf file",
    "Number of candidate objects to export",
    "--processed-dir",
    "--candidate-dir",
    "--threshold-sigma",
    "--n-pixels",
    "--kernel-sigma",
    "--min-snr",
    "Isolation Forest seed.",
    "Fit the background once, without masking detected sources.",
    "Detect with 8 connected pixels and no smoothing kernel.",
    "Local reference CSV, or a directory of CSV files. Offline only.",
    "Cross-match radius in arcseconds.",
    "positional arguments:",
)


def _install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "ngrst-discover"
    script.write_text(source, encoding="utf-8")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("COLUMNS", "160")
    return script


def test_command_from_mocked_help_uses_positional_interface(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = _install(tmp_path, monkeypatch, MOCK_NGRST)
    blob = help_text(str(script))
    for snippet in HELP_SNIPPETS:
        assert snippet in blob
    command = command_from_help(
        blob,
        image="scene.npz",
        processed_dir="processed",
        candidate_dir="candidates",
    )
    assert command == [
        "scene.npz",
        "--processed-dir",
        "processed",
        "--candidate-dir",
        "candidates",
    ]
    with pytest.raises(DataError, match="--scene"):
        command_from_help(
            blob,
            image="scene.npz",
            processed_dir="processed",
            candidate_dir="candidates",
            scene="points",
        )


def test_unknown_help_is_not_guessed() -> None:
    with pytest.raises(DependencyError) as caught:
        command_from_help(
            "usage: ngrst-discover [options]\n",
            image="a.npz",
            processed_dir="processed",
            candidate_dir="candidates",
        )
    assert caught.value.status == "UNVERIFIED_INTERFACE"
    with pytest.raises(DependencyError) as caught:
        command_from_help(
            "usage: ngrst-discover --image IMAGE --output DETECTIONS --scene {points,grid}\n",
            image="a.npz",
            processed_dir="processed",
            candidate_dir="candidates",
        )
    assert caught.value.status == "UNVERIFIED_INTERFACE"
    prose = (
        "usage: ngrst-discover [options]\n"
        "Notes mention --processed-dir and --candidate-dir but do not define them.\n"
        "positional arguments:\n"
        "  image                 path\n"
    )
    with pytest.raises(DependencyError) as caught:
        command_from_help(prose, image="a.npz", processed_dir="processed", candidate_dir="candidates")
    assert caught.value.status == "UNVERIFIED_INTERFACE"


def test_output_paths_follow_export_layout(tmp_path: Path) -> None:
    outputs = ngrst_output_paths(tmp_path / "demo_scene.npz", tmp_path / "processed", tmp_path / "candidates")
    assert outputs.catalog == tmp_path / "processed" / "demo_scene_catalog.csv"
    assert outputs.run_json == tmp_path / "processed" / "demo_scene_run.json"
    assert outputs.candidate_directory == tmp_path / "candidates" / "demo_scene"
    outputs.catalog.parent.mkdir(parents=True)
    outputs.catalog.write_text("label,x_centroid,y_centroid\n1,0,0\n", encoding="utf-8")
    outputs.run_json.write_text('{"seed": 42}\n', encoding="utf-8")
    outputs.candidate_directory.mkdir(parents=True)
    assert resolve_run_json(outputs.catalog) == outputs.run_json
    record = require_ngrst_outputs(outputs)
    assert record["seed"] == 42
    outputs.catalog.unlink()
    with pytest.raises(DetectorError, match="catalogue"):
        require_ngrst_outputs(outputs)


def test_run_reads_catalog_from_mocked_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install(tmp_path, monkeypatch, MOCK_NGRST)
    synth_out = tmp_path / "synthetic"
    assert main(["synth", "--config", str(EXAMPLES / "controlled_scene.toml"), "--out", str(synth_out)]) == 0
    image = synth_out / "image.npz"
    out = tmp_path / "reports"
    assert (
        main(
            [
                "run",
                "--image",
                str(image),
                "--truth",
                str(synth_out / "truth.csv"),
                "--detector",
                "ngrst-discover",
                "--out",
                str(out),
                "--radius-px",
                "2",
            ]
        )
        == 0
    )
    catalog = out / "detector" / "processed" / "image_catalog.csv"
    run_json = out / "detector" / "processed" / "image_run.json"
    assert catalog.is_file()
    assert run_json.is_file()
    assert (out / "detector" / "candidates" / "image" / "001_object_4.png").is_file()
    assert not (out / "detector" / "detections.csv").exists()
    command = (out / "detector" / "command.txt").read_text(encoding="utf-8")
    assert "--processed-dir" in command
    assert "--candidate-dir" in command
    assert "--image" not in command
    assert "--output" not in command
    assert command.index(str(image)) < command.index("--processed-dir")
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["adapters"]["detection"] == "ngrst_detection"
    assert metrics["adapters"]["detection_ids_synthesized"] is False
    assert metrics["run"]["detector"]["run_record"]["seed"] == 42
    assert metrics["run"]["detector"]["catalog"].endswith("image_catalog.csv")
    assert metrics["photometry"]["status"] == "NOT_APPLICABLE"


def test_unknown_mock_exits_before_running(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _install(tmp_path, monkeypatch, UNKNOWN_HELP)
    image = tmp_path / "scene.npz"
    image.write_bytes(b"not an image")
    truth = tmp_path / "truth.csv"
    truth.write_text("source_id,x,y\na,1,1\n", encoding="utf-8")
    status = main(
        [
            "run",
            "--image",
            str(image),
            "--truth",
            str(truth),
            "--detector",
            "ngrst-discover",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert status == 4
    assert "Refusing to guess" in capsys.readouterr().err
    assert not (tmp_path / "out" / "detector" / "processed").exists()
