"""Command line interface for the catalogue, synthetic, and doctor workflows."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tomllib
from pathlib import Path
from typing import Any, Sequence

from romanrecoverer import __version__
from romanrecoverer.detector_adapters.ngrst import command_from_help, help_text, probe_ngrst, require_ngrst
from romanrecoverer.detector_adapters.romanisim_l2 import probe_romanisim
from romanrecoverer.detector_adapters.subprocess_runner import persist_run, render_template, run_detector
from romanrecoverer.errors import DataError, DetectorError, RomanRecovererError
from romanrecoverer.evaluate import EvaluationRequest, evaluate
from romanrecoverer.image_io import read_image
from romanrecoverer.inject import default_config, generate_scene, write_scene
from romanrecoverer.plots import write_all_plots
from romanrecoverer.provenance import file_record, software_versions, utc_now, write_json
from romanrecoverer.reports import write_products

EXIT_OK = 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return int(args.handler(args))
    except RomanRecovererError as exc:
        print(f"romanrecoverer: {exc}", file=sys.stderr)
        return int(exc.exit_code)
    except BrokenPipeError:
        return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="romanrecoverer",
        description=(
            "Injection-and-recovery benchmarks for Roman WFI source catalogues. "
            "Does not claim discoveries and does not replace romanisim."
        ),
    )
    parser.add_argument("--version", action="version", version=f"romanrecoverer {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = sub.add_parser("evaluate", help="Match a truth catalogue to a detection catalogue.")
    evaluate_parser.add_argument("--truth", required=True, type=Path)
    evaluate_parser.add_argument("--detections", required=True, type=Path)
    evaluate_parser.add_argument("--radius-px", required=True, type=float)
    evaluate_parser.add_argument("--out", required=True, type=Path)
    evaluate_parser.add_argument("--image", type=Path)
    evaluate_parser.add_argument("--image-shape", nargs=2, type=int, metavar=("NY", "NX"))
    evaluate_parser.add_argument("--edge-margin-px", type=float)
    evaluate_parser.add_argument("--pixel-origin", type=int, choices=(0, 1))
    evaluate_parser.add_argument(
        "--assumption",
        choices=("unspecified", "truth_complete_synthetic", "populated_background"),
    )
    evaluate_parser.add_argument("--baseline", type=Path)
    evaluate_parser.add_argument("--baseline-radius-px", type=float)
    evaluate_parser.add_argument("--calibration-status")
    evaluate_parser.add_argument("--mode", choices=("catalogue_only", "synthetic_demonstration", "romanisim_l2", "real_data"))
    evaluate_parser.add_argument("--crowd-radius-px", type=float, default=5.0)
    evaluate_parser.add_argument("--bootstrap-seed", type=int, default=12345)
    evaluate_parser.add_argument("--n-bootstrap", type=int, default=2000)
    evaluate_parser.add_argument("--brightness-edges", type=float, nargs="+")
    evaluate_parser.add_argument("--truth-x-col")
    evaluate_parser.add_argument("--truth-y-col")
    evaluate_parser.add_argument("--detection-x-col")
    evaluate_parser.add_argument("--detection-y-col")
    evaluate_parser.set_defaults(handler=command_evaluate)

    synth = sub.add_parser("synth", help="Write a seeded dimensionless synthetic scene.")
    synth.add_argument("--config", type=Path)
    synth.add_argument("--preset", choices=("controlled",), default="controlled")
    synth.add_argument("--out", required=True, type=Path)
    synth.add_argument("--seed", type=int)
    synth.set_defaults(handler=command_synth)

    run = sub.add_parser("run", help="Run an external detector, then evaluate its catalogue.")
    run.add_argument("--image", required=True, type=Path)
    run.add_argument("--truth", required=True, type=Path)
    run.add_argument("--detector", required=True)
    run.add_argument("--out", required=True, type=Path)
    run.add_argument("--radius-px", type=float, default=2.0)
    run.add_argument("--detector-cmd")
    run.add_argument("--scene")
    run.add_argument("--assumption", choices=("unspecified", "truth_complete_synthetic", "populated_background"))
    run.add_argument("--timeout", type=float, default=600)
    run.set_defaults(handler=command_run)

    doctor = sub.add_parser("doctor", help="Report installed dependencies and skipped science adapters.")
    doctor.add_argument("--json", action="store_true")
    doctor.add_argument("--require", action="append", default=[], choices=("ngrst", "romanisim"))
    doctor.set_defaults(handler=command_doctor)

    compare = sub.add_parser("compare", help="Compare two metrics files without choosing a threshold.")
    compare.add_argument("--a", required=True, type=Path)
    compare.add_argument("--b", required=True, type=Path)
    compare.add_argument("--out", required=True, type=Path)
    compare.set_defaults(handler=command_compare)
    return parser


def command_evaluate(args: argparse.Namespace) -> int:
    extra: dict[str, Any] = {}
    if args.edge_margin_px is not None:
        extra["edge_margin_from_cli"] = True
    if args.pixel_origin is not None:
        extra["pixel_origin_from_cli"] = True
    detector_record = getattr(args, "detector_record", None)
    if detector_record:
        extra["detector"] = detector_record
    request = EvaluationRequest(
        truth_path=args.truth,
        detection_path=args.detections,
        radius=args.radius_px,
        image_shape=tuple(args.image_shape) if args.image_shape else None,
        edge_margin_px=0.0 if args.edge_margin_px is None else args.edge_margin_px,
        pixel_origin=0 if args.pixel_origin is None else args.pixel_origin,
        dataset_assumption=args.assumption,
        calibration_status=args.calibration_status,
        baseline_path=args.baseline,
        baseline_radius=args.baseline_radius_px,
        crowd_radius=args.crowd_radius_px,
        bootstrap_seed=args.bootstrap_seed,
        n_bootstrap=args.n_bootstrap,
        brightness_edges=args.brightness_edges,
        truth_x=args.truth_x_col,
        truth_y=args.truth_y_col,
        detection_x=args.detection_x_col,
        detection_y=args.detection_y_col,
        mode=args.mode,
        extra=extra,
    )
    result = evaluate(request)
    image = None
    if args.image is not None:
        image, _, _ = read_image(args.image)
    plots = write_all_plots(
        result.matches,
        result.strata,
        args.out / "plots",
        radius=args.radius_px,
        image=image,
        precision_status=result.metrics["precision"]["status"],
    )
    inputs = [
        file_record(args.truth, role="truth"),
        file_record(args.detections, role="detections"),
    ]
    if args.image is not None:
        inputs.append(file_record(args.image, role="image"))
    if args.baseline is not None:
        inputs.append(file_record(args.baseline, role="baseline"))
    write_products(
        args.out,
        metrics=result.metrics,
        matches=result.matches,
        strata=result.strata,
        plot_paths=plots,
        inputs=inputs,
    )
    counts = result.metrics["counts"]
    print(
        f"completeness={result.metrics['completeness']} "
        f"recovered={counts['n_recovered']}/{counts['n_eligible']} "
        f"precision_status={result.metrics['precision']['status']} "
        f"photometry_status={result.metrics['photometry']['status']}"
    )
    print(f"wrote {args.out}")
    return EXIT_OK


def command_synth(args: argparse.Namespace) -> int:
    if args.config is not None:
        if not args.config.is_file():
            raise DataError(f"Config not found: {args.config}")
        with args.config.open("rb") as handle:
            config = tomllib.load(handle)
    else:
        config = default_config()
    if args.seed is not None:
        config["seed"] = args.seed
    before = _hash_inputs_note(args.out)
    scene = generate_scene(config)
    paths = write_scene(scene, args.out)
    config_copy = args.out / "scene_config.toml"
    if args.config is not None:
        shutil.copyfile(args.config, config_copy)
    else:
        config_copy.write_text(_toml_fallback_note(config), encoding="utf-8")
    inputs = [file_record(config_copy, role="scene_config")]
    write_json(
        args.out / "manifest.json",
        {
            "schema_name": "romanrecoverer.manifest",
            "schema_version": "1.0.0",
            "created_utc": utc_now(),
            "immutable_inputs": True,
            "note": "Synthetic products. Not a Roman observation.",
            "mode": scene.metadata["mode"],
            "calibration_status": scene.metadata["calibration_status"],
            "seed": scene.metadata["seed"],
            "n_sources": int(len(scene.truth)),
            "image_shape": scene.metadata["image_shape"],
            "inputs": inputs,
            "outputs": [file_record(path, role=name) for name, path in paths.items()],
            "output_directory_existed": before,
            "software": software_versions(),
        },
    )
    print(
        f"wrote {paths['image']} and {paths['truth']} "
        f"n_sources={len(scene.truth)} seed={scene.metadata['seed']} "
        f"calibration_status={scene.metadata['calibration_status']}"
    )
    return EXIT_OK


def _hash_inputs_note(out: Path) -> bool:
    return out.exists()


def _toml_fallback_note(config: dict[str, Any]) -> str:
    return (
        "# Preset configuration snapshot (JSON embedded because the preset was not a TOML file).\n"
        + json.dumps(config, indent=2, sort_keys=True)
        + "\n"
    )


def command_run(args: argparse.Namespace) -> int:
    if not args.image.is_file():
        raise DataError(f"Image not found: {args.image}")
    if not args.truth.is_file():
        raise DataError(f"Truth catalogue not found: {args.truth}")
    out = args.out
    detector_dir = out / "detector"
    detector_dir.mkdir(parents=True, exist_ok=True)
    detections = detector_dir / "detections.csv"
    run_json = detector_dir / "run.json"
    image_before = file_record(args.image, role="image")
    truth_before = file_record(args.truth, role="truth")

    detector_name = args.detector
    if detector_name in {"ngrst-discover", "ngrst"}:
        if args.detector_cmd:
            command = render_template(
                args.detector_cmd,
                _mapping(args, detections, run_json),
            )
        else:
            probe = require_ngrst()
            assert probe["executable"] is not None
            built = command_from_help(
                help_text(probe["executable"]),
                image=str(args.image),
                detections=str(detections),
                run_json=str(run_json),
                scene=args.scene,
            )
            command = [probe["executable"], *built]
    elif detector_name == "command":
        if not args.detector_cmd:
            raise DataError("detector 'command' requires --detector-cmd with {image} and {detections}.")
        command = render_template(args.detector_cmd, _mapping(args, detections, run_json))
    else:
        raise DataError(
            f"Unknown detector {detector_name!r}. Use ngrst-discover, or command with --detector-cmd. "
            "RomanRecoverer does not substitute its own detector."
        )

    run = run_detector(command, timeout=args.timeout)
    persist_run(run, detector_dir)
    image_after = file_record(args.image, role="image_after")
    truth_after = file_record(args.truth, role="truth_after")
    if image_before["sha256"] != image_after["sha256"] or truth_before["sha256"] != truth_after["sha256"]:
        print(
            "romanrecoverer: detector modified an input file. The hash change is recorded. "
            "RomanRecoverer did not rewrite the inputs itself.",
            file=sys.stderr,
        )
    if run.returncode != 0:
        write_json(
            out / "metrics.json",
            {
                "status": "DETECTOR_FAILED",
                "exit_code": run.returncode,
                "command": run.command_line,
                "mode": "detector_failed",
                "note": "No completeness was computed because the detector exited non-zero.",
            },
        )
        raise DetectorError(f"Detector exited {run.returncode}. See {detector_dir / 'stderr.txt'}.")
    if not detections.is_file():
        raise DetectorError(f"Detector exited 0 but did not write {detections}.")

    evaluate_args = argparse.Namespace(
        truth=args.truth,
        detections=detections,
        radius_px=args.radius_px,
        out=out,
        image=args.image,
        image_shape=None,
        edge_margin_px=None,
        pixel_origin=None,
        assumption=args.assumption,
        baseline=None,
        baseline_radius_px=None,
        calibration_status=None,
        mode=None,
        crowd_radius_px=5.0,
        bootstrap_seed=12345,
        n_bootstrap=2000,
        brightness_edges=None,
        truth_x_col=None,
        truth_y_col=None,
        detection_x_col=None,
        detection_y_col=None,
        detector_record={
            "name": detector_name,
            "command": run.command_line,
            "exit_code": run.returncode,
            "stdout_path": str(detector_dir / "stdout.txt"),
            "stderr_path": str(detector_dir / "stderr.txt"),
            "input_modified": image_before["sha256"] != image_after["sha256"]
            or truth_before["sha256"] != truth_after["sha256"],
        },
    )
    return command_evaluate(evaluate_args)


def _mapping(args: argparse.Namespace, detections: Path, run_json: Path) -> dict[str, str]:
    return {
        "image": str(args.image),
        "truth": str(args.truth),
        "detections": str(detections),
        "run_json": str(run_json),
        "out": str(args.out),
    }


def command_doctor(args: argparse.Namespace) -> int:
    roman = probe_romanisim()
    ngrst = probe_ngrst()
    payload = {
        "romanrecoverer": __version__,
        "time_utc": utc_now(),
        "software": software_versions(),
        "catalogue_only": "available",
        "synthetic_demonstration": "available",
        "romanisim_l2": roman,
        "ngrst": ngrst,
        "stips": {
            "used": False,
            "note": (
                "STIPS is maintenance-only per the Roman User Documentation "
                "(https://roman-docs.stsci.edu/simulation-tools/additional-simulation-tools/"
                "stips-space-telescope-imaging-product-simulator). Roman I-Sim is the "
                "high-fidelity path and is not silently replaced by STIPS or by a Gaussian."
            ),
        },
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"romanrecoverer {__version__}")
        print(f"python {payload['software']['python']}")
        for name in ("numpy", "scipy", "pandas", "astropy", "matplotlib", "photutils"):
            print(f"  {name}: {payload['software'].get(name)}")
        print(f"romanisim: {roman['status']} version={roman.get('romanisim_version')}")
        if roman.get("reason"):
            print(f"  {roman['reason']}")
        print(f"ngrst-discover: {ngrst['status']} executable={ngrst.get('executable')}")
        if ngrst.get("reason"):
            print(f"  {ngrst['reason']}")
        print("Catalogue-only and synthetic demonstration modes do not need those packages.")
    missing = []
    if "ngrst" in args.require and not ngrst["available"]:
        missing.append("ngrst")
    if "romanisim" in args.require and not roman.get("available"):
        missing.append("romanisim")
    if missing:
        print(f"required dependencies missing: {', '.join(missing)}", file=sys.stderr)
        return 4
    return EXIT_OK


def command_compare(args: argparse.Namespace) -> int:
    left = json.loads(args.a.read_text(encoding="utf-8"))
    right = json.loads(args.b.read_text(encoding="utf-8"))
    left_seed = _dig_seed(left)
    right_seed = _dig_seed(right)
    same_seed = left_seed is not None and left_seed == right_seed
    report = {
        "schema_name": "romanrecoverer.comparison",
        "schema_version": "1.0.0",
        "created_utc": utc_now(),
        "a": str(args.a),
        "b": str(args.b),
        "completeness_a": left.get("completeness"),
        "completeness_b": right.get("completeness"),
        "completeness_difference_b_minus_a": _difference(left.get("completeness"), right.get("completeness")),
        "seed_a": left_seed,
        "seed_b": right_seed,
        "same_seed": same_seed,
        "controlled_comparison": same_seed,
        "threshold_policy": (
            "No threshold was selected. Overlapping or non-overlapping Wilson intervals "
            "are descriptive. Choosing the configuration with the higher completeness on "
            "this same sample is not an unbiased evaluation."
        ),
        "wilson_a": (left.get("confidence_intervals") or {}).get("wilson"),
        "wilson_b": (right.get("confidence_intervals") or {}).get("wilson"),
        "modes": [left.get("mode"), right.get("mode")],
        "calibration_status": [left.get("calibration_status"), right.get("calibration_status")],
    }
    if not same_seed:
        report["warning"] = (
            "Seeds differ or are absent. Placements are not a paired comparison of detector "
            "configurations. Do not attribute the completeness difference to the detector alone."
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, report)
    print(json.dumps({k: report[k] for k in ("completeness_a", "completeness_b", "same_seed", "controlled_comparison")}, indent=2))
    return EXIT_OK


def _dig_seed(metrics: dict[str, Any]) -> int | None:
    for key in ("seed",):
        if key in metrics and isinstance(metrics[key], int):
            return metrics[key]
    run = metrics.get("run")
    if isinstance(run, dict) and isinstance(run.get("seed"), int):
        return run["seed"]
    return None


def _difference(left: object, right: object) -> float | None:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(right) - float(left)
    return None
