"""One-to-one gated matching of injected sources to detections.

The assignment is a rectangular linear-sum assignment (Hungarian algorithm via
``scipy.optimize.linear_sum_assignment``) on the Euclidean separation, with
pairs outside the match radius removed after the solve. A detection is used
at most once, so one detection cannot recover two truth sources, and extra
detections piled on one truth cannot inflate the recovery count.

Ties in separation are broken stably toward the earlier detection row and
then the earlier truth row by adding a perturbation many orders of magnitude
below a pixel. The perturbation does not change which pairs fall inside the
radius.

Ambiguous: an eligible truth has more than one detection inside the radius.
Blended: a detection has more than one eligible truth inside the radius.
Both flags are recorded even when the final assignment is unique. Ineligible
truths do not enter the assignment and cannot consume a detection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from romanrecoverer.errors import DataError


@dataclass(frozen=True)
class MatchSolution:
    """Index-based solution. Indices refer to the frames passed in."""

    truth_index: np.ndarray
    detection_index: np.ndarray
    separation: np.ndarray
    n_candidate_detections: np.ndarray
    n_candidate_truths: np.ndarray
    distance: np.ndarray


def match_positions(
    truth_xy: np.ndarray,
    detection_xy: np.ndarray,
    *,
    radius: float,
    eligible: np.ndarray | None = None,
) -> MatchSolution:
    if radius <= 0 or not np.isfinite(radius):
        raise DataError(f"Match radius must be a positive finite number, got {radius}.")

    truth_xy = np.asarray(truth_xy, dtype=float)
    detection_xy = np.asarray(detection_xy, dtype=float)
    if truth_xy.ndim != 2 or truth_xy.shape[1] != 2:
        raise DataError("truth_xy must have shape (n, 2).")
    if detection_xy.ndim != 2 or detection_xy.shape[1] != 2:
        raise DataError("detection_xy must have shape (n, 2).")

    n_truth = truth_xy.shape[0]
    n_det = detection_xy.shape[0]
    if eligible is None:
        eligible = np.ones(n_truth, dtype=bool)
    else:
        eligible = np.asarray(eligible, dtype=bool)
        if eligible.shape != (n_truth,):
            raise DataError("eligible mask must have one entry per truth row.")

    n_candidates_det = np.zeros(n_truth, dtype=int)
    n_candidates_truth = np.zeros(n_det, dtype=int)
    if n_truth == 0 or n_det == 0 or not bool(eligible.any()):
        distance = np.zeros((n_truth, n_det), dtype=float)
        return MatchSolution(
            truth_index=np.array([], dtype=int),
            detection_index=np.array([], dtype=int),
            separation=np.array([], dtype=float),
            n_candidate_detections=n_candidates_det,
            n_candidate_truths=n_candidates_truth,
            distance=distance,
        )

    # Ineligible rows are kept in the distance matrix so indices stay aligned,
    # but they are excluded from the assignment by giving them no finite cost.
    eligible_index = np.flatnonzero(eligible)
    distance = cdist(truth_xy[eligible_index], detection_xy, metric="euclidean")
    within = distance <= radius
    n_candidates_det[eligible_index] = within.sum(axis=1).astype(int)
    n_candidates_truth[:] = within.sum(axis=0).astype(int)

    finite = distance[np.isfinite(distance)]
    scale = float(np.max(finite)) if finite.size else 1.0
    large = scale + float(radius) + 1.0
    cost = np.where(within, distance, large)
    n_eligible = cost.shape[0]
    # Stable tie break. eps is far below any separation that would cross the radius gate.
    eps = 1e-9
    det_index = np.arange(n_det, dtype=float)[None, :]
    truth_index = np.arange(n_eligible, dtype=float)[:, None]
    cost = cost + eps * (truth_index * n_det + det_index) / max(n_eligible * n_det, 1)

    rows, cols = linear_sum_assignment(cost)
    keep_truth: list[int] = []
    keep_det: list[int] = []
    keep_sep: list[float] = []
    for row, col in zip(rows, cols, strict=True):
        separation = float(distance[row, col])
        if separation <= radius:
            keep_truth.append(int(eligible_index[row]))
            keep_det.append(int(col))
            keep_sep.append(separation)

    full_distance = np.full((n_truth, n_det), np.nan, dtype=float)
    if n_eligible:
        full_distance[eligible_index, :] = distance

    return MatchSolution(
        truth_index=np.asarray(keep_truth, dtype=int),
        detection_index=np.asarray(keep_det, dtype=int),
        separation=np.asarray(keep_sep, dtype=float),
        n_candidate_detections=n_candidates_det,
        n_candidate_truths=n_candidates_truth,
        distance=full_distance,
    )


def build_match_table(
    truth: pd.DataFrame,
    detections: pd.DataFrame,
    solution: MatchSolution,
    *,
    eligible: np.ndarray,
    eligibility_reason: np.ndarray,
    radius: float,
) -> pd.DataFrame:
    """One row per truth, plus one row per detection that recovered nobody."""

    matched_truth = {int(i): k for k, i in enumerate(solution.truth_index)}
    matched_det = {int(j): k for k, j in enumerate(solution.detection_index)}
    rows: list[dict[str, object]] = []

    for index in range(len(truth)):
        record = {
            "row_kind": "truth",
            "truth_row": index,
            "detection_row": None,
            "truth_id": truth.iloc[index]["truth_id"],
            "detection_id": None,
            "truth_x": float(truth.iloc[index]["match_x"]),
            "truth_y": float(truth.iloc[index]["match_y"]),
            "detection_x": None,
            "detection_y": None,
            "separation": None,
            "dx": None,
            "dy": None,
            "eligible": bool(eligible[index]),
            "eligibility_reason": str(eligibility_reason[index]),
            "recovered": False,
            "ambiguous": bool(solution.n_candidate_detections[index] > 1),
            "blended": False,
            "n_candidate_detections": int(solution.n_candidate_detections[index]),
            "n_candidate_truths": None,
            "match_radius": radius,
        }
        if index in matched_truth:
            slot = matched_truth[index]
            det_index = int(solution.detection_index[slot])
            dx = float(detections.iloc[det_index]["match_x"] - truth.iloc[index]["match_x"])
            dy = float(detections.iloc[det_index]["match_y"] - truth.iloc[index]["match_y"])
            record.update(
                {
                    "detection_row": det_index,
                    "detection_id": detections.iloc[det_index]["detection_id"],
                    "detection_x": float(detections.iloc[det_index]["match_x"]),
                    "detection_y": float(detections.iloc[det_index]["match_y"]),
                    "separation": float(solution.separation[slot]),
                    "dx": dx,
                    "dy": dy,
                    "recovered": bool(eligible[index]),
                    "blended": bool(solution.n_candidate_truths[det_index] > 1),
                    "n_candidate_truths": int(solution.n_candidate_truths[det_index]),
                }
            )
        rows.append(record)

    for index in range(len(detections)):
        if index in matched_det:
            continue
        rows.append(
            {
                "row_kind": "unmatched_detection",
                "truth_row": None,
                "detection_row": index,
                "truth_id": None,
                "detection_id": detections.iloc[index]["detection_id"],
                "truth_x": None,
                "truth_y": None,
                "detection_x": float(detections.iloc[index]["match_x"]),
                "detection_y": float(detections.iloc[index]["match_y"]),
                "separation": None,
                "dx": None,
                "dy": None,
                "eligible": None,
                "eligibility_reason": None,
                "recovered": False,
                "ambiguous": False,
                "blended": bool(solution.n_candidate_truths[index] > 1) if len(solution.n_candidate_truths) else False,
                "n_candidate_detections": None,
                "n_candidate_truths": int(solution.n_candidate_truths[index]) if len(solution.n_candidate_truths) else 0,
                "match_radius": radius,
            }
        )

    return pd.DataFrame(rows)
