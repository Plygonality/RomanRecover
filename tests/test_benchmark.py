"""Mandatory catalogue, injection, WCS, and adapter tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from romanrecoverer.cli import main
from romanrecoverer.detector_adapters.ngrst import command_from_help, help_text, probe_ngrst
from romanrecoverer.detector_adapters.romanisim_l2 import (
    inject_l2,
    probe_romanisim,
    validate_catalog_table,
    validate_l2_model,
)
from romanrecoverer.eligibility import evaluate_eligibility
from romanrecoverer.errors import DataError, DependencyError
from romanrecoverer.evaluate import EvaluationRequest, evaluate
from romanrecoverer.inject import generate_scene, render_sources
from romanrecoverer.metrics import clopper_pearson_interval, wilson_interval
from romanrecoverer.wcsutil import low_level_pix2world, pixel_to_sky, sky_to_pixel

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _evaluate(truth: pd.DataFrame, detections: pd.DataFrame, tmp_path: Path, **kwargs):
    truth_path = tmp_path / "truth.csv"
    det_path = tmp_path / "det.csv"
    _write_csv(truth_path, truth)
    _write_csv(det_path, detections)
    request = EvaluationRequest(
        truth_path=truth_path,
        detection_path=det_path,
        radius=kwargs.pop("radius", 2.0),
        **kwargs,
    )
    return evaluate(request)


def test_all_recovered_completeness_is_one(tmp_path: Path) -> None:
    truth = pd.DataFrame(
        {
            "source_id": ["a", "b", "c"],
            "x": [10.0, 20.0, 30.0],
            "y": [10.0, 12.0, 18.0],
            "amplitude": [3.0, 4.0, 5.0],
            "sigma": [1.0, 1.0, 1.0],
            "class": ["point", "point", "point"],
        }
    )
    detections = pd.DataFrame(
        {
            "detection_id": ["d1", "d2", "d3"],
            "x_centroid": [10.1, 19.9, 30.2],
            "y_centroid": [10.0, 12.1, 17.9],
            "snr": [10.0, 11.0, 12.0],
            "anomaly_score": [0.1, 0.1, 0.1],
        }
    )
    result = _evaluate(
        truth,
        detections,
        tmp_path,
        dataset_assumption="truth_complete_synthetic",
        image_shape=(64, 64),
    )
    assert result.metrics["counts"]["n_recovered"] == 3
    assert result.metrics["counts"]["n_eligible"] == 3
    assert result.metrics["completeness"] == 1.0
    assert result.metrics["precision"]["value"] == 1.0
    assert result.metrics["photometry"]["status"] == "NOT_APPLICABLE"


def test_empty_detections_completeness_zero_and_undefined_precision(tmp_path: Path) -> None:
    truth = pd.DataFrame({"source_id": ["a", "b"], "x": [5.0, 15.0], "y": [5.0, 15.0]})
    detections = pd.DataFrame(columns=["detection_id", "x_centroid", "y_centroid", "snr", "anomaly_score"])
    result = _evaluate(
        truth,
        detections,
        tmp_path,
        dataset_assumption="truth_complete_synthetic",
        image_shape=(32, 32),
    )
    assert result.metrics["completeness"] == 0.0
    assert result.metrics["counts"]["n_unmatched_truth"] == 2
    assert result.metrics["precision"]["status"] == "UNDEFINED_ZERO_DENOMINATOR"
    assert result.metrics["precision"]["value"] is None


def test_one_to_one_matching_does_not_inflate_recovery(tmp_path: Path) -> None:
    truth = pd.DataFrame({"source_id": ["a", "b"], "x": [10.0, 10.4], "y": [10.0, 10.0]})
    detections = pd.DataFrame(
        {
            "detection_id": ["d1", "d2", "d3"],
            "x": [10.0, 10.1, 10.2],
            "y": [10.0, 10.0, 10.05],
        }
    )
    result = _evaluate(truth, detections, tmp_path, radius=2.0, image_shape=(32, 32))
    assert result.metrics["counts"]["n_recovered"] == 2
    assert result.metrics["counts"]["n_recovered"] < 3
    assert result.metrics["counts"]["n_ambiguous_truth"] == 2
    assert bool(result.matches.loc[result.matches["row_kind"] == "truth", "ambiguous"].all())

    single = _evaluate(
        truth.iloc[[0]].copy(),
        detections,
        tmp_path / "single",
        radius=2.0,
        image_shape=(32, 32),
    )
    assert single.metrics["counts"]["n_recovered"] == 1
    assert single.metrics["counts"]["n_unmatched_detected"] == 2


def test_tie_prefers_earlier_detection_row(tmp_path: Path) -> None:
    truth = pd.DataFrame({"source_id": ["a"], "x": [0.0], "y": [0.0]})
    detections = pd.DataFrame(
        {
            "detection_id": ["first", "second"],
            "x": [1.0, -1.0],
            "y": [0.0, 0.0],
        }
    )
    result = _evaluate(truth, detections, tmp_path, radius=2.0, image_shape=(16, 16), edge_margin_px=0.0)
    # The source at the corner is ineligible with margin 0? x=0 is inside [0, 15].
    matched = result.matches[result.matches["recovered"] == True]  # noqa: E712
    assert list(matched["detection_id"]) == ["first"]


def test_masked_and_edge_sources_are_ineligible(tmp_path: Path) -> None:
    truth = pd.DataFrame(
        {
            "source_id": ["interior", "masked", "edge", "outside"],
            "x": [16.0, 16.0, 1.0, -2.0],
            "y": [16.0, 16.0, 16.0, 16.0],
            "mask": [0, 1, 0, 0],
        }
    )
    detections = pd.DataFrame({"detection_id": ["d"], "x": [16.0], "y": [16.0]})
    result = _evaluate(
        truth,
        detections,
        tmp_path,
        radius=1.0,
        image_shape=(32, 32),
        edge_margin_px=2.0,
        extra={"edge_margin_from_cli": True, "pixel_origin_from_cli": True},
    )
    reasons = dict(zip(result.truth["truth_id"], result.truth["eligibility_reason"], strict=True))
    assert reasons["interior"] == "eligible"
    assert reasons["masked"] == "masked"
    assert reasons["edge"] == "edge_or_outside"
    assert reasons["outside"] == "edge_or_outside"
    assert result.metrics["counts"]["n_eligible"] == 1
    assert result.metrics["counts"]["n_recovered"] == 1
    assert result.metrics["completeness"] == 1.0
    # The masked source sits on the same pixel but must not consume the detection.
    assert result.metrics["counts"]["n_ineligible"] == 3


def test_no_image_shape_does_not_pretend_to_cut_edges(tmp_path: Path) -> None:
    truth = pd.DataFrame({"source_id": ["a"], "x": [-5.0], "y": [0.0]})
    detections = pd.DataFrame(columns=["detection_id", "x", "y"])
    result = _evaluate(truth, detections, tmp_path, extra={"edge_margin_from_cli": True})
    assert result.metrics["eligibility"]["edge_exclusion_applied"] is False
    assert result.metrics["counts"]["n_eligible"] == 1


def test_unmatched_detections_are_not_false_positives_by_default(tmp_path: Path) -> None:
    truth = pd.DataFrame({"source_id": ["a"], "x": [10.0], "y": [10.0], "amplitude": [2.0], "sigma": [1.0], "class": ["point"]})
    detections = pd.DataFrame(
        {
            "detection_id": ["kept", "extra"],
            "x_centroid": [10.0, 40.0],
            "y_centroid": [10.0, 40.0],
            "snr": [8.0, 9.0],
            "anomaly_score": [0.2, 0.8],
        }
    )
    populated = _evaluate(
        truth,
        detections,
        tmp_path,
        dataset_assumption="populated_background",
        image_shape=(64, 64),
    )
    assert populated.metrics["precision"]["status"] == "NOT_APPLICABLE"
    assert populated.metrics["precision"]["false_positive_count"] is None
    assert populated.metrics["precision"]["labelled_unmatched_as_false_positive"] is False
    assert populated.metrics["counts"]["n_unmatched_detected"] == 1

    assumed = _evaluate(
        truth,
        detections,
        tmp_path / "assumed",
        dataset_assumption="truth_complete_synthetic",
        image_shape=(64, 64),
    )
    assert assumed.metrics["precision"]["labelled_unmatched_as_false_positive"] is True
    assert assumed.metrics["precision"]["false_positive_count"] == 1
    assert assumed.metrics["precision"]["value"] == pytest.approx(0.5)
    assert "assumption" in assumed.metrics["precision"]["assumption"]


def test_baseline_subtraction_explains_preexisting_sources(tmp_path: Path) -> None:
    truth = pd.DataFrame({"source_id": ["inj"], "x": [10.0], "y": [10.0]})
    detections = pd.DataFrame(
        {
            "detection_id": ["inj-det", "real", "noise"],
            "x": [10.1, 30.0, 50.0],
            "y": [10.0, 30.0, 10.0],
        }
    )
    baseline = tmp_path / "baseline.csv"
    _write_csv(baseline, pd.DataFrame({"x": [30.0], "y": [30.2]}))
    result = _evaluate(
        truth,
        detections,
        tmp_path,
        dataset_assumption="populated_background",
        baseline_path=baseline,
        image_shape=(64, 64),
    )
    assert result.metrics["precision"]["status"] == "ESTIMATED_AFTER_BASELINE_SUBTRACTION"
    assert result.metrics["precision"]["baseline_explained_count"] == 1
    assert result.metrics["precision"]["false_positive_count"] == 1
    assert result.metrics["precision"]["value"] == pytest.approx(0.5)


def test_same_seed_repeats_scene_and_other_seed_moves_sources() -> None:
    config = {
        "seed": 11,
        "image_shape": [40, 40],
        "edge_margin_px": 3,
        "noise_sigma": 0.0,
        "point_sources": {"n": 6, "peak_amplitude_min": 1.0, "peak_amplitude_max": 2.0},
        "extended_sources": {"n": 1, "profile": "exponential", "peak_amplitude_min": 1.0, "peak_amplitude_max": 1.0, "scale_px_min": 1.5, "scale_px_max": 1.5},
        "psf": {"sigma_px": 1.0},
    }
    first = generate_scene(config)
    second = generate_scene(dict(config))
    moved = generate_scene(dict(config, seed=12))
    assert list(first.truth["source_id"]) == list(second.truth["source_id"])
    assert np.allclose(first.truth["x"], second.truth["x"])
    assert np.allclose(first.truth["y"], second.truth["y"])
    assert np.allclose(first.image, second.image)
    assert not np.allclose(first.truth[["x", "y"]].to_numpy(), moved.truth[["x", "y"]].to_numpy())
    assert first.metadata["calibration_status"] == "dimensionless_synthetic"
    assert "not an AB magnitude" in first.metadata["amplitude_meaning"].lower() or "Not an AB magnitude" in first.metadata["amplitude_meaning"]


def test_invalid_flux_and_shape_raise() -> None:
    image = np.zeros((12, 12))
    base = {"source_id": "a", "x": 6.0, "y": 6.0, "profile": "gaussian", "sigma": 1.0, "scale_px": 1.0}
    with pytest.raises(DataError, match="negative"):
        render_sources(image, pd.DataFrame([{**base, "peak_amplitude": -0.2}]))
    with pytest.raises(DataError, match="non-finite"):
        render_sources(image, pd.DataFrame([{**base, "peak_amplitude": np.nan}]))
    with pytest.raises(DataError, match="zero"):
        render_sources(image, pd.DataFrame([{**base, "peak_amplitude": 0.0}]))
    zero = render_sources(image, pd.DataFrame([{**base, "peak_amplitude": 0.0}]), allow_zero=True)
    assert zero[1].loc[0, "zero_reason"] == "zero_amplitude"
    with pytest.raises(DataError, match="sigma"):
        render_sources(image, pd.DataFrame([{**base, "peak_amplitude": 1.0, "sigma": 0.0}]))
    with pytest.raises(DataError, match="image_shape"):
        generate_scene({"image_shape": [0, 10], "point_sources": {"n": 0}, "extended_sources": {"n": 0}})
    with pytest.raises(DataError, match="AB magnitude"):
        generate_scene({"image_unit": "ab_mag", "point_sources": {"n": 1}, "extended_sources": {"n": 0}})


def test_wcs_round_trip_axis_order_and_origin() -> None:
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [1.0, 1.0]
    wcs.wcs.crval = [270.0, 66.0]
    wcs.wcs.cdelt = [0.11 / 3600.0, 0.11 / 3600.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.cunit = ["deg", "deg"]

    sky = pixel_to_sky(wcs, np.array([0.0]), np.array([0.0]), origin=0)
    assert np.allclose(sky.ra.deg, 270.0, atol=1e-8)
    assert np.allclose(sky.dec.deg, 66.0, atol=1e-8)

    # FITS pixel (1, 1) is the same place as 0-based pixel (0, 0).
    sky_fits = pixel_to_sky(wcs, np.array([1.0]), np.array([1.0]), origin=1)
    assert np.allclose(sky_fits.ra.deg, sky.ra.deg)
    assert np.allclose(sky_fits.dec.deg, sky.dec.deg)

    low_ra, low_dec = low_level_pix2world(wcs, np.array([1.0]), np.array([1.0]), origin=1)
    high = pixel_to_sky(wcs, np.array([0.0]), np.array([0.0]), origin=0)
    assert np.allclose(low_ra, high.ra.deg)
    assert np.allclose(low_dec, high.dec.deg)

    x_back, y_back = sky_to_pixel(wcs, sky, origin=0)
    assert np.allclose(x_back, 0.0, atol=1e-6)
    assert np.allclose(y_back, 0.0, atol=1e-6)

    # x is the column, y is the row. A source at column 3, row 10 is not at (10, 3).
    image = np.zeros((16, 12))
    image[10, 3] = 5.0
    assert image[10, 3] == 5.0
    column, row = 3.0, 10.0
    sky_xy = pixel_to_sky(wcs, np.array([column]), np.array([row]), origin=0)
    sky_swapped = pixel_to_sky(wcs, np.array([row]), np.array([column]), origin=0)
    separation = sky_xy.separation(sky_swapped).arcsec
    assert separation > 0.1
    round_trip = sky_to_pixel(wcs, sky_xy, origin=0)
    assert np.allclose(round_trip[0], column, atol=1e-5)
    assert np.allclose(round_trip[1], row, atol=1e-5)
    # The array sample lives at [row, column], which is [y, x].
    nearest_row = int(round(float(round_trip[1][0])))
    nearest_col = int(round(float(round_trip[0][0])))
    assert image[nearest_row, nearest_col] == 5.0
    assert isinstance(sky_xy, SkyCoord)


def test_wilson_and_clopper_edges() -> None:
    assert wilson_interval(0, 0) is None
    lower, upper = wilson_interval(0, 8)
    assert lower == 0.0
    assert 0.0 < upper < 1.0
    lower, upper = wilson_interval(8, 8)
    assert upper == 1.0
    assert 0.0 < lower < 1.0
    cp = clopper_pearson_interval(0, 10)
    assert cp is not None and cp[0] == 0.0
    assert cp[1] < 1.0
    seven = wilson_interval(7, 8)
    assert seven is not None
    assert seven[0] < 0.875 < seven[1]


def test_example_catalogue_benchmark() -> None:
    result = evaluate(
        EvaluationRequest(
            truth_path=EXAMPLES / "points_truth.csv",
            detection_path=EXAMPLES / "points_catalog.csv",
            radius=2.0,
        )
    )
    counts = result.metrics["counts"]
    assert counts == {
        "n_injected": 8,
        "n_eligible": 8,
        "n_recovered": 7,
        "n_unmatched_truth": 1,
        "n_unmatched_detected": 2,
        "n_ineligible": 0,
        "n_ambiguous_truth": 0,
        "n_blended_rows": 0,
    }
    assert result.metrics["completeness"] == pytest.approx(7 / 8)
    assert result.metrics["precision"]["status"] == "NOT_APPLICABLE"
    assert result.metrics["photometry"]["status"] == "NOT_APPLICABLE"
    assert result.metrics["adapters"]["truth"] == "ngrst_truth"
    assert result.metrics["adapters"]["detection"] == "ngrst_detection"
    missed = result.matches[(result.matches["truth_id"] == "t08") & (result.matches["row_kind"] == "truth")]
    assert bool(missed["recovered"].item()) is False
    assert "discovery" in result.metrics["anomaly_note"]


def test_calibrated_magnitude_bias_and_refusal_of_amplitude(tmp_path: Path) -> None:
    truth = pd.DataFrame(
        {
            "source_id": ["a", "b"],
            "x": [4.0, 10.0],
            "y": [4.0, 10.0],
            "mag": [20.0, 21.0],
            "amplitude": [5.0, 1.0],
        }
    )
    detections = pd.DataFrame(
        {
            "detection_id": ["d1", "d2"],
            "x": [4.0, 10.0],
            "y": [4.1, 10.0],
            "mag": [20.1, 20.5],
            "amplitude": [4.0, 2.0],
        }
    )
    calibrated = _evaluate(
        truth,
        detections,
        tmp_path,
        calibration_status="calibrated",
        image_shape=(24, 24),
    )
    assert calibrated.metrics["photometry"]["status"] == "DEFINED"
    assert calibrated.metrics["photometry"]["quantity"] == "magnitude"
    # Residuals are +0.1 and -0.5; the median is -0.2.
    assert calibrated.metrics["photometry"]["bias"]["median"] == pytest.approx(-0.2)

    uncalibrated = _evaluate(
        truth.drop(columns=["mag"]),
        detections.drop(columns=["mag"]),
        tmp_path / "amp",
        calibration_status="uncalibrated_amplitude",
        truth_unit="dimensionless_peak",
        detection_unit="dimensionless_peak",
        image_shape=(24, 24),
    )
    assert uncalibrated.metrics["photometry"]["status"] == "NOT_APPLICABLE"
    relative = uncalibrated.metrics["photometry"]["relative_amplitude"]
    assert relative["status"] == "RELATIVE_AMPLITUDE_ONLY"
    assert relative["median_measured_over_injected"] == pytest.approx(np.median([4 / 5, 2 / 1]))


def test_strata_empty_bin_and_zero_denominator(tmp_path: Path) -> None:
    truth = pd.DataFrame(
        {
            "source_id": ["a"],
            "x": [5.0],
            "y": [5.0],
            "amplitude": [1.0],
            "sigma": [1.0],
            "mask": [1],
        }
    )
    detections = pd.DataFrame({"detection_id": ["d"], "x": [5.0], "y": [5.0]})
    result = _evaluate(
        truth,
        detections,
        tmp_path,
        image_shape=(16, 16),
        brightness_edges=[0.0, 0.5, 2.0],
        extra={"edge_margin_from_cli": True},
    )
    assert result.metrics["completeness_status"] == "UNDEFINED_ZERO_DENOMINATOR"
    brightness = result.strata[result.strata["stratum"] == "brightness_proxy_amplitude"]
    assert "EMPTY_BIN" in set(brightness["status"])
    assert set(brightness.loc[brightness["status"] == "EMPTY_BIN", "completeness"].tolist()) == {None} or brightness.loc[
        brightness["status"] == "EMPTY_BIN", "completeness"
    ].isna().all()


def test_installed_ngrst_discover_uses_positional_interface() -> None:
    """Build a command from a real install. Skip when the program is absent."""

    probe = probe_ngrst()
    if not probe["available"] or not probe.get("executable"):
        pytest.skip("SKIPPED_DEPENDENCY: ngrst-discover is not installed.")
    command = command_from_help(
        help_text(probe["executable"]),
        image="scene.npz",
        processed_dir="processed",
        candidate_dir="candidates",
    )
    assert command[:5] == [
        "scene.npz",
        "--processed-dir",
        "processed",
        "--candidate-dir",
        "candidates",
    ]


def test_romanisim_adapter_skips_or_uses_real_signature() -> None:
    probe = probe_romanisim()
    if not probe["available"]:
        assert probe["status"] == "SKIPPED_DEPENDENCY"
        with pytest.raises(DependencyError) as caught:
            inject_l2(object(), object())
        assert caught.value.status == "SKIPPED_DEPENDENCY"
    else:
        assert "model" in probe["installed_parameters"]
        assert "cat" in probe["installed_parameters"]
        with pytest.raises((DataError, DependencyError)):
            inject_l2(object(), {"not": "a model"})

    class ImageModel:
        meta = None

    with pytest.raises(DataError, match="no meta"):
        validate_l2_model(ImageModel())

    table = pd.DataFrame(
        {
            "ra": [270.0],
            "dec": [66.0],
            "type": ["PSF"],
            "n": [-1.0],
            "half_light_radius": [0.0],
            "pa": [0.0],
            "ba": [1.0],
            "F158": [1e-10],
        }
    )
    validate_catalog_table(table, filter_name="F158")
    with pytest.raises(DataError, match="flux column"):
        validate_catalog_table(table, filter_name="F087")


def test_nonfinite_coordinates_are_rejected(tmp_path: Path) -> None:
    truth = tmp_path / "truth.csv"
    truth.write_text("source_id,x,y\na,1,nan\n", encoding="utf-8")
    det = tmp_path / "det.csv"
    det.write_text("detection_id,x,y\n", encoding="utf-8")
    with pytest.raises(DataError, match="non-finite"):
        evaluate(EvaluationRequest(truth_path=truth, detection_path=det, radius=1.0))


def test_cli_evaluate_synth_run_and_doctor(tmp_path: Path) -> None:
    out = tmp_path / "reports" / "points"
    status = main(
        [
            "evaluate",
            "--truth",
            str(EXAMPLES / "points_truth.csv"),
            "--detections",
            str(EXAMPLES / "points_catalog.csv"),
            "--radius-px",
            "2.0",
            "--out",
            str(out),
        ]
    )
    assert status == 0
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["completeness"] == pytest.approx(7 / 8)
    assert (out / "report.html").is_file()
    assert (out / "matches.csv").is_file()
    assert (out / "plots" / "associations.png").is_file()
    html = (out / "report.html").read_text(encoding="utf-8")
    assert "not a false positive" in html or "not automatically false positives" in html
    assert "NOT_APPLICABLE" in html

    synth_out = tmp_path / "synthetic"
    assert main(["synth", "--config", str(EXAMPLES / "controlled_scene.toml"), "--out", str(synth_out)]) == 0
    assert (synth_out / "image.npz").is_file()
    truth = pd.read_csv(synth_out / "truth.csv")
    assert len(truth) == 12
    assert set(truth["class"]) == {"point", "extended"}

    run_out = tmp_path / "pipeline"
    script = EXAMPLES / "detectors" / "offset_catalogue.py"
    command = (
        f"{sys.executable} {script} --truth {{truth}} --output {{detections}} --min-amplitude 0"
    )
    assert (
        main(
            [
                "run",
                "--image",
                str(synth_out / "image.npz"),
                "--truth",
                str(synth_out / "truth.csv"),
                "--detector",
                "command",
                "--detector-cmd",
                command,
                "--out",
                str(run_out),
                "--radius-px",
                "2.0",
            ]
        )
        == 0
    )
    run_metrics = json.loads((run_out / "metrics.json").read_text(encoding="utf-8"))
    assert run_metrics["completeness"] == 1.0
    assert run_metrics["mode"] == "synthetic_demonstration"
    assert run_metrics["run"]["detector"]["exit_code"] == 0
    assert "offset_catalogue.py" in run_metrics["run"]["detector"]["command"]
    assert (run_out / "detector" / "stdout.txt").is_file()

    assert main(["doctor"]) == 0
    if probe_romanisim()["available"]:
        assert main(["doctor", "--require", "romanisim"]) == 0
    else:
        assert main(["doctor", "--require", "romanisim"]) == 4
    if probe_ngrst()["available"]:
        assert main(["doctor", "--require", "ngrst"]) == 0
    else:
        assert main(["doctor", "--require", "ngrst"]) == 4

    skipped = main(
        [
            "run",
            "--image",
            str(synth_out / "image.npz"),
            "--truth",
            str(synth_out / "truth.csv"),
            "--detector",
            "ngrst-discover",
            "--out",
            str(tmp_path / "missing-ngrst"),
        ]
    )
    if not probe_ngrst()["available"]:
        assert skipped == 4


def test_compare_warns_when_seeds_differ(tmp_path: Path) -> None:
    synth_a = tmp_path / "a"
    synth_b = tmp_path / "b"
    assert main(["synth", "--config", str(EXAMPLES / "controlled_scene.toml"), "--out", str(synth_a), "--seed", "1"]) == 0
    assert main(["synth", "--config", str(EXAMPLES / "controlled_scene.toml"), "--out", str(synth_b), "--seed", "2"]) == 0
    # Evaluate each scene against an empty detection list so the metrics carry the seed.
    empty = tmp_path / "empty.csv"
    empty.write_text("detection_id,x_centroid,y_centroid,snr,anomaly_score\n", encoding="utf-8")
    report_a = tmp_path / "ra"
    report_b = tmp_path / "rb"
    assert main(["evaluate", "--truth", str(synth_a / "truth.csv"), "--detections", str(empty), "--radius-px", "2", "--out", str(report_a)]) == 0
    assert main(["evaluate", "--truth", str(synth_b / "truth.csv"), "--detections", str(empty), "--radius-px", "2", "--out", str(report_b)]) == 0
    comparison = tmp_path / "compare.json"
    assert main(["compare", "--a", str(report_a / "metrics.json"), "--b", str(report_b / "metrics.json"), "--out", str(comparison)]) == 0
    payload = json.loads(comparison.read_text(encoding="utf-8"))
    assert payload["same_seed"] is False
    assert "warning" in payload
    assert "unbiased" in payload["threshold_policy"]


def test_eligibility_origin_one() -> None:
    # FITS pixel 1 is 0-based pixel 0. With margin 0 and shape 8, FITS x=1 is inside.
    result = evaluate_eligibility(
        np.array([1.0, 9.0]),
        np.array([1.0, 1.0]),
        None,
        image_shape=(8, 8),
        edge_margin_px=0.0,
        pixel_origin=1,
    )
    assert list(result.eligible) == [True, False]
