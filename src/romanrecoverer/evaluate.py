"""Catalogue-only evaluation. No detector and no Roman library are required."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from romanrecoverer.catalogs import (
    assign_ids,
    calibration_from_columns,
    load_sidecar,
    load_table,
    require_finite_coordinates,
    resolve_detection_columns,
    resolve_truth_columns,
)
from romanrecoverer.eligibility import evaluate_eligibility
from romanrecoverer.errors import DataError
from romanrecoverer.match import build_match_table, match_positions
from romanrecoverer.metrics import completeness_block, photometry_block, position_block, precision_block
from romanrecoverer.schemas import (
    ASSUMPTION_POPULATED,
    ASSUMPTION_TRUTH_COMPLETE,
    ASSUMPTION_UNSPECIFIED,
    AXIS_ORDER,
    MATCH_SCHEMA,
    METRICS_SCHEMA,
    MODE_CATALOGUE,
    MODE_SYNTHETIC,
    SCHEMA_VERSION,
    STRATA_SCHEMA,
)
from romanrecoverer.stratify import build_strata, crowding_counts

ASSUMPTIONS = {ASSUMPTION_UNSPECIFIED, ASSUMPTION_TRUTH_COMPLETE, ASSUMPTION_POPULATED}


@dataclass
class EvaluationRequest:
    truth_path: Path
    detection_path: Path
    radius: float
    coord_frame: str = "pixel"
    pixel_origin: int = 0
    image_shape: tuple[int, int] | None = None
    edge_margin_px: float = 0.0
    dataset_assumption: str | None = None
    calibration_status: str | None = None
    baseline_path: Path | None = None
    baseline_radius: float | None = None
    truth_x: str | None = None
    truth_y: str | None = None
    detection_x: str | None = None
    detection_y: str | None = None
    brightness_edges: list[float] | None = None
    crowd_radius: float = 5.0
    bootstrap_seed: int = 12345
    n_bootstrap: int = 2000
    mode: str | None = None
    truth_unit: str | None = None
    detection_unit: str | None = None
    truth_kind: str | None = None
    detection_kind: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    metrics: dict[str, Any]
    matches: pd.DataFrame
    strata: pd.DataFrame
    truth: pd.DataFrame
    detections: pd.DataFrame


def _prepare_truth(request: EvaluationRequest) -> tuple[pd.DataFrame, dict[str, Any], Any]:
    frame = load_table(request.truth_path)
    sidecar = load_sidecar(request.truth_path)
    columns = resolve_truth_columns(frame, x_col=request.truth_x, y_col=request.truth_y)
    require_finite_coordinates(frame, columns.x, columns.y, what="Truth catalogue")
    ids, synthesized = assign_ids(frame, columns.id, prefix="truth")
    prepared = pd.DataFrame(
        {
            "truth_id": ids.to_numpy(),
            "match_x": pd.to_numeric(frame[columns.x]).to_numpy(dtype=float),
            "match_y": pd.to_numeric(frame[columns.y]).to_numpy(dtype=float),
        }
    )
    for name, attribute in (
        ("amplitude", "amplitude"),
        ("sigma", "sigma"),
        ("flux", "flux"),
        ("mag", "mag"),
        ("snr", "snr"),
        ("local_background", "background"),
    ):
        source = getattr(columns, attribute)
        if source and source in frame.columns:
            prepared[name] = pd.to_numeric(frame[source], errors="coerce").to_numpy()
    if columns.source_class and columns.source_class in frame.columns:
        prepared["source_class"] = frame[columns.source_class].astype(str).to_numpy()
    if columns.mask and columns.mask in frame.columns:
        prepared["mask"] = frame[columns.mask].to_numpy()
    if columns.anomaly_score and columns.anomaly_score in frame.columns:
        prepared["anomaly_score"] = pd.to_numeric(frame[columns.anomaly_score], errors="coerce").to_numpy()
    prepared.attrs["column_map"] = columns
    prepared.attrs["ids_synthesized"] = synthesized
    return prepared, sidecar, columns


def _prepare_detections(request: EvaluationRequest) -> tuple[pd.DataFrame, dict[str, Any], Any]:
    frame = load_table(request.detection_path)
    sidecar = load_sidecar(request.detection_path)
    columns = resolve_detection_columns(frame, x_col=request.detection_x, y_col=request.detection_y)
    if not frame.empty:
        require_finite_coordinates(frame, columns.x, columns.y, what="Detection catalogue")
    ids, synthesized = assign_ids(frame, columns.id, prefix="det")
    if frame.empty:
        prepared = pd.DataFrame(
            {
                "detection_id": pd.Series(dtype=str),
                "match_x": pd.Series(dtype=float),
                "match_y": pd.Series(dtype=float),
            }
        )
    else:
        prepared = pd.DataFrame(
            {
                "detection_id": ids.to_numpy(),
                "match_x": pd.to_numeric(frame[columns.x]).to_numpy(dtype=float),
                "match_y": pd.to_numeric(frame[columns.y]).to_numpy(dtype=float),
            }
        )
    for name, attribute in (
        ("amplitude", "amplitude"),
        ("sigma", "sigma"),
        ("flux", "flux"),
        ("mag", "mag"),
        ("snr", "snr"),
        ("local_background", "background"),
    ):
        source = getattr(columns, attribute)
        if source and source in frame.columns:
            prepared[name] = pd.to_numeric(frame[source], errors="coerce").to_numpy()
    if columns.anomaly_score and columns.anomaly_score in frame.columns and not frame.empty:
        prepared["anomaly_score"] = pd.to_numeric(frame[columns.anomaly_score], errors="coerce").to_numpy()
        prepared.attrs["anomaly_note"] = (
            "anomaly_score is copied from the detector catalogue. "
            "RomanRecoverer does not interpret it as a discovery."
        )
    prepared.attrs["column_map"] = columns
    prepared.attrs["ids_synthesized"] = synthesized
    return prepared, sidecar, columns


def _image_shape(request: EvaluationRequest, sidecar: dict[str, Any]) -> tuple[int, int] | None:
    if request.image_shape is not None:
        return request.image_shape
    declared = sidecar.get("image_shape")
    if declared is None:
        return None
    if not isinstance(declared, (list, tuple)) or len(declared) != 2:
        raise DataError(f"Sidecar image_shape must be [ny, nx], got {declared!r}.")
    return int(declared[0]), int(declared[1])


def _assumption(request: EvaluationRequest, sidecar: dict[str, Any]) -> str:
    if request.dataset_assumption:
        assumption = request.dataset_assumption
    else:
        assumption = str(sidecar.get("dataset_assumption", ASSUMPTION_UNSPECIFIED))
    if assumption not in ASSUMPTIONS:
        raise DataError(
            f"dataset assumption must be one of {sorted(ASSUMPTIONS)}, got {assumption!r}."
        )
    return assumption


def _mode(request: EvaluationRequest, sidecar: dict[str, Any]) -> str:
    if request.mode:
        return request.mode
    declared = sidecar.get("mode")
    if declared == MODE_SYNTHETIC:
        return MODE_SYNTHETIC
    return MODE_CATALOGUE


def evaluate(request: EvaluationRequest) -> EvaluationResult:
    if request.coord_frame not in {"pixel", "sky"}:
        raise DataError("coord_frame must be 'pixel' or 'sky'.")
    if request.coord_frame == "sky":
        raise DataError(
            "Sky-frame matching requires --radius-arcsec and a WCS. "
            "This release matches in the pixel frame. Convert sky catalogues to pixels "
            "with romanrecoverer.wcsutil before evaluating, and keep the origin explicit."
        )
    if request.radius <= 0 or not np.isfinite(request.radius):
        raise DataError(f"radius must be positive and finite, got {request.radius}.")

    truth, truth_sidecar, truth_columns = _prepare_truth(request)
    detections, detection_sidecar, detection_columns = _prepare_detections(request)
    image_shape = _image_shape(request, truth_sidecar)
    margin = request.edge_margin_px
    if margin == 0.0 and "edge_margin_px" in truth_sidecar and request.image_shape is None:
        # The sidecar margin applies when the user did not override the shape on the CLI.
        # A CLI edge margin of 0 is the default, so a sidecar value should win when the
        # user did not pass the flag. EvaluationRequest cannot tell those apart, so the
        # CLI sets edge_margin_px to None-equivalent by using a sentinel only when passed.
        pass
    if "edge_margin_px" in truth_sidecar and request.extra.get("edge_margin_from_cli") is not True:
        margin = float(truth_sidecar["edge_margin_px"])

    origin = request.pixel_origin
    if request.extra.get("pixel_origin_from_cli") is not True and "pixel_origin" in truth_sidecar:
        origin = int(truth_sidecar["pixel_origin"])

    eligibility = evaluate_eligibility(
        truth["match_x"].to_numpy(),
        truth["match_y"].to_numpy(),
        truth["mask"] if "mask" in truth.columns else None,
        image_shape=image_shape,
        edge_margin_px=margin,
        pixel_origin=origin,
    )
    truth["eligibility_reason"] = eligibility.reason
    truth["eligible"] = eligibility.eligible

    solution = match_positions(
        truth[["match_x", "match_y"]].to_numpy(),
        detections[["match_x", "match_y"]].to_numpy() if len(detections) else np.zeros((0, 2)),
        radius=request.radius,
        eligible=eligibility.eligible,
    )
    matches = build_match_table(
        truth,
        detections if len(detections) else detections,
        solution,
        eligible=eligibility.eligible,
        eligibility_reason=eligibility.reason,
        radius=request.radius,
    )

    recovered_flags = matches.loc[matches["row_kind"] == "truth", "recovered"].to_numpy(dtype=bool)
    n_injected = int(len(truth))
    n_eligible = int(eligibility.eligible.sum())
    n_recovered = int(recovered_flags.sum())
    n_unmatched_truth = n_eligible - n_recovered
    n_unmatched_detected = int((matches["row_kind"] == "unmatched_detection").sum())
    n_ineligible = n_injected - n_eligible

    n_baseline = None
    if request.baseline_path is not None:
        baseline = load_table(request.baseline_path)
        base_columns = resolve_detection_columns(
            baseline,
            x_col=request.extra.get("baseline_x"),
            y_col=request.extra.get("baseline_y"),
        )
        if not baseline.empty:
            require_finite_coordinates(baseline, base_columns.x, base_columns.y, what="Baseline catalogue")
        unmatched = matches[matches["row_kind"] == "unmatched_detection"]
        if baseline.empty or unmatched.empty:
            n_baseline = 0
        else:
            base_xy = np.column_stack(
                [
                    pd.to_numeric(baseline[base_columns.x]).to_numpy(dtype=float),
                    pd.to_numeric(baseline[base_columns.y]).to_numpy(dtype=float),
                ]
            )
            det_xy = unmatched[["detection_x", "detection_y"]].to_numpy(dtype=float)
            base_solution = match_positions(
                base_xy,
                det_xy,
                radius=request.baseline_radius or request.radius,
            )
            n_baseline = int(len(base_solution.separation))

    assumption = _assumption(request, truth_sidecar)
    calibration = request.calibration_status or calibration_from_columns(truth_columns, truth_sidecar)
    mode = _mode(request, truth_sidecar)
    eligible_recovered = recovered_flags[eligibility.eligible] if n_injected else recovered_flags
    completeness = completeness_block(
        n_recovered,
        n_eligible,
        recovered_flags=eligible_recovered,
        seed=request.bootstrap_seed,
        n_bootstrap=request.n_bootstrap,
    )

    precision = precision_block(
        n_recovered=n_recovered,
        n_unmatched_detected=n_unmatched_detected,
        dataset_assumption=assumption,
        n_baseline_explained=n_baseline,
    )
    truth_kind = request.truth_kind
    detection_kind = request.detection_kind
    truth_unit = request.truth_unit or truth_sidecar.get("photometric_unit")
    detection_unit = request.detection_unit or detection_sidecar.get("photometric_unit")
    if calibration == "calibrated" and truth_kind is None and detection_kind is None:
        if "mag" in truth.columns and "mag" in detections.columns:
            truth_kind = detection_kind = "magnitude"
            truth_unit = truth_unit or "mag"
            detection_unit = detection_unit or "mag"
        elif "flux" in truth.columns and "flux" in detections.columns:
            truth_kind = detection_kind = "flux"
            truth_unit = truth_unit or truth_sidecar.get("flux_unit") or "flux"
            detection_unit = detection_unit or detection_sidecar.get("flux_unit") or "flux"
    photometry = photometry_block(
        matches,
        truth,
        detections,
        calibration_status=calibration,
        truth_unit=truth_unit,
        detection_unit=detection_unit,
        truth_kind=truth_kind,
        detection_kind=detection_kind,
    )
    crowding = crowding_counts(
        truth[["match_x", "match_y"]].to_numpy(),
        eligibility.eligible,
        request.crowd_radius,
    )
    truth["crowding"] = crowding
    strata, strata_notes = build_strata(
        truth,
        eligible=eligibility.eligible,
        recovered=recovered_flags,
        crowding=crowding,
        calibration_status=calibration,
        brightness_edges=request.brightness_edges,
        crowd_radius=request.crowd_radius,
    )

    n_ambiguous = int(matches.loc[matches["row_kind"] == "truth", "ambiguous"].sum())
    n_blended = int(matches["blended"].fillna(False).sum())

    metrics: dict[str, Any] = {
        "schema_name": METRICS_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "mode_note": _mode_note(mode),
        "calibration_status": calibration,
        "coordinate_frame": request.coord_frame,
        "pixel_origin": origin,
        "axis_order": AXIS_ORDER,
        "match_radius_px": request.radius,
        "dataset_assumption": assumption,
        "counts": {
            "n_injected": n_injected,
            "n_eligible": n_eligible,
            "n_recovered": n_recovered,
            "n_unmatched_truth": n_unmatched_truth,
            "n_unmatched_detected": n_unmatched_detected,
            "n_ineligible": n_ineligible,
            "n_ambiguous_truth": n_ambiguous,
            "n_blended_rows": n_blended,
        },
        "eligibility": {
            "definition": (
                "An injection is eligible when it is unmasked and, if an image shape is known, "
                "its center lies in the closed interval [margin, n-1-margin] on both axes "
                "after converting to 0-based pixels. x is the column and y is the row."
            ),
            "edge_exclusion_applied": eligibility.edge_exclusion_applied,
            "image_shape": list(eligibility.image_shape) if eligibility.image_shape else None,
            "edge_margin_px": eligibility.edge_margin_px,
            "pixel_origin": eligibility.pixel_origin,
        },
        **completeness,
        "precision": precision,
        "photometry": photometry,
        "position": position_block(matches),
        "matching": {
            "algorithm": "gated_hungarian",
            "one_to_one": True,
            "tie_break": "earlier detection row, then earlier eligible truth row",
            "ineligible_sources_consume_detections": False,
        },
        "strata_notes": strata_notes,
        "adapters": {
            "truth": truth_columns.adapter,
            "detection": detection_columns.adapter,
            "truth_ids_synthesized": bool(truth.attrs.get("ids_synthesized")),
            "detection_ids_synthesized": bool(detections.attrs.get("ids_synthesized")),
        },
        "anomaly_note": detections.attrs.get(
            "anomaly_note",
            "No anomaly_score column was read. RomanRecoverer would not treat one as a discovery.",
        ),
        "threshold_policy": (
            "RomanRecoverer did not select a detection threshold. "
            "If a threshold was tuned on this same sample, the numbers are descriptive "
            "and are not an unbiased evaluation of that choice."
        ),
        "false_positive_policy": precision["policy"],
        "seed": truth_sidecar.get("seed"),
        "partition": truth_sidecar.get("partition"),
    }
    if request.extra:
        # Keep operational notes that are not column-resolution flags.
        passthrough = {
            key: value
            for key, value in request.extra.items()
            if key
            not in {
                "edge_margin_from_cli",
                "pixel_origin_from_cli",
                "baseline_x",
                "baseline_y",
            }
        }
        if passthrough:
            metrics["run"] = passthrough

    matches.attrs["schema_name"] = MATCH_SCHEMA
    matches.attrs["schema_version"] = SCHEMA_VERSION
    strata.attrs["schema_name"] = STRATA_SCHEMA
    strata.attrs["schema_version"] = SCHEMA_VERSION
    return EvaluationResult(
        metrics=metrics,
        matches=matches,
        strata=strata,
        truth=truth,
        detections=detections,
    )


def _mode_note(mode: str) -> str:
    notes = {
        MODE_CATALOGUE: (
            "Catalogue-only benchmark. No image was simulated by RomanRecoverer in this step, "
            "and no Roman instrument calibration was applied."
        ),
        MODE_SYNTHETIC: (
            "Synthetic demonstration. Injected profiles and noise follow the declared "
            "dimensionless model. These numbers are not Roman I-Sim results and not "
            "flight-data measurements."
        ),
        "romanisim_l2": (
            "Roman I-Sim L2 injection. Subject to the L2 injection caveats "
            "(cosmic rays, nonlinearity, persistence). Not a claim of on-sky performance."
        ),
        "real_data": (
            "The caller labelled this run as real data. RomanRecoverer did not verify "
            "that the files are Roman observations, and unmatched detections are not discoveries."
        ),
    }
    return notes.get(mode, "Unlabelled mode. Do not treat the numbers as instrument-calibrated science.")
