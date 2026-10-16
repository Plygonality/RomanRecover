"""Stratified completeness with raw counts and explicit empty bins.

Bins are defined on truth properties of eligible injections: brightness proxy
(amplitude, or magnitude when the calibration status is calibrated), source
size, crowding, local background, truth SNR, and mask status.

Detected SNR is a property of the detection, so it is not used as a
completeness axis. Conditioning bins on a quantity that exists only after
detection would make the denominator depend on the outcome. When the truth
table has no SNR column, the stratum is reported as not available and the
detected-SNR values of recovered sources are summarised separately as a
detection property, not as completeness.

Crowding is the number of other eligible injections whose centers lie within
``crowd_radius`` in the match frame. Ineligible injections are not neighbors.

Mask status is reported as a count stratum. Completeness itself uses only
eligible sources, so the masked stratum does not enter the headline ratio.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from romanrecoverer.metrics import clopper_pearson_interval, wilson_interval
from romanrecoverer.schemas import CALIBRATION_CALIBRATED, STATUS_EMPTY_BIN, STATUS_NOT_APPLICABLE


def crowding_counts(xy: np.ndarray, eligible: np.ndarray, radius: float) -> np.ndarray:
    n = xy.shape[0]
    counts = np.zeros(n, dtype=int)
    eligible_index = np.flatnonzero(eligible)
    if eligible_index.size == 0:
        return counts
    if radius <= 0:
        return counts
    points = xy[eligible_index]
    if points.shape[0] == 1:
        return counts
    tree = cKDTree(points)
    neighbors = tree.query_ball_point(points, r=radius, return_length=True)
    counts[eligible_index] = np.asarray(neighbors, dtype=int) - 1
    return counts


def _bin_labels(edges: np.ndarray) -> list[str]:
    labels = []
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        labels.append(f"[{left:.6g}, {right:.6g})")
    if labels:
        # The last edge is closed on the right so the maximum value is included.
        labels[-1] = f"[{edges[-2]:.6g}, {edges[-1]:.6g}]"
    return labels


def _edges_from_values(values: np.ndarray, n_bins: int = 4) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.array([], dtype=float)
    unique = np.unique(finite)
    if unique.size == 1:
        value = float(unique[0])
        pad = 1.0 if value == 0 else abs(value) * 1e-6
        return np.array([value - pad, value + pad], dtype=float)
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(finite, quantiles)
    edges = np.unique(edges)
    if edges.size < 2:
        return np.array([], dtype=float)
    # Nudge the right edge so the maximum is inside the closed last bin.
    edges = edges.astype(float)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    return edges


def _strata_for_edges(
    group: str,
    values: np.ndarray,
    eligible: np.ndarray,
    recovered: np.ndarray,
    edges: np.ndarray,
    *,
    unit: str,
) -> list[dict[str, Any]]:
    labels = _bin_labels(edges)
    # digitize with the internal edges; values equal to the last edge fall in the last bin
    # because we nudged the right edge upward when edges came from the data. User-supplied
    # edges are treated as closed on the right for the final bin.
    if edges.size < 2:
        return []
    bins = np.full(values.shape[0], -1, dtype=int)
    finite = np.isfinite(values) & eligible
    # Last bin is closed.
    interior = finite & (values >= edges[0]) & (values < edges[-1])
    if np.any(interior):
        bins[interior] = np.digitize(values[interior], edges[1:-1], right=False)
    on_right = finite & (values == edges[-1])
    bins[on_right] = len(labels) - 1
    # Values above a nudged edge still belong in the last bin.
    above = finite & (values > edges[-1])
    bins[above] = len(labels) - 1

    rows = []
    for index, label in enumerate(labels):
        mask = bins == index
        n_eligible = int(mask.sum())
        n_recovered = int((mask & recovered).sum())
        if n_eligible == 0:
            completeness = None
            status = STATUS_EMPTY_BIN
            wilson = None
            clopper = None
        else:
            completeness = n_recovered / n_eligible
            status = "DEFINED"
            wilson = wilson_interval(n_recovered, n_eligible)
            clopper = clopper_pearson_interval(n_recovered, n_eligible)
        rows.append(
            {
                "stratum": group,
                "bin": label,
                "unit": unit,
                "n_eligible": n_eligible,
                "n_recovered": n_recovered,
                "n_unmatched_truth": n_eligible - n_recovered,
                "completeness": completeness,
                "status": status,
                "wilson_lower": None if wilson is None else wilson[0],
                "wilson_upper": None if wilson is None else wilson[1],
                "clopper_pearson_lower": None if clopper is None else clopper[0],
                "clopper_pearson_upper": None if clopper is None else clopper[1],
            }
        )
    return rows


def build_strata(
    truth: pd.DataFrame,
    *,
    eligible: np.ndarray,
    recovered: np.ndarray,
    crowding: np.ndarray,
    calibration_status: str,
    brightness_edges: list[float] | None = None,
    crowd_radius: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    notes: dict[str, Any] = {
        "crowd_radius": crowd_radius,
        "crowd_definition": (
            "Number of other eligible injected sources within crowd_radius "
            "of the source center, in the match frame."
        ),
    }

    if "amplitude" in truth.columns:
        values = pd.to_numeric(truth["amplitude"], errors="coerce").to_numpy(dtype=float)
        edges = (
            np.asarray(brightness_edges, dtype=float)
            if brightness_edges
            else _edges_from_values(values[eligible])
        )
        rows.extend(
            _strata_for_edges(
                "brightness_proxy_amplitude",
                values,
                eligible,
                recovered,
                edges,
                unit="amplitude_as_declared",
            )
        )
        notes["brightness"] = (
            "Brightness bins use the amplitude column as a brightness proxy. "
            "For NGRST-style tables this is a Gaussian peak, not a magnitude."
        )
    else:
        notes["brightness"] = "No amplitude column; brightness-proxy stratum omitted."

    if calibration_status == CALIBRATION_CALIBRATED and "mag" in truth.columns:
        values = pd.to_numeric(truth["mag"], errors="coerce").to_numpy(dtype=float)
        edges = _edges_from_values(values[eligible])
        rows.extend(
            _strata_for_edges("magnitude", values, eligible, recovered, edges, unit="magnitude")
        )
    else:
        notes["magnitude"] = (
            "Magnitude bins are omitted because calibration_status is not "
            "'calibrated' or no magnitude column is present. "
            "Physical magnitude bins are not invented from peak amplitudes."
        )

    if "sigma" in truth.columns:
        values = pd.to_numeric(truth["sigma"], errors="coerce").to_numpy(dtype=float)
        edges = _edges_from_values(values[eligible])
        rows.extend(_strata_for_edges("source_size_sigma", values, eligible, recovered, edges, unit="pixel"))
    else:
        notes["size"] = "No sigma column; source-size stratum omitted."

    rows.extend(
        _strata_for_edges(
            "crowding",
            crowding.astype(float),
            eligible,
            recovered,
            np.array([0, 1, 2, 4, np.inf], dtype=float),
            unit="count_of_other_eligible_sources",
        )
    )

    if "local_background" in truth.columns:
        values = pd.to_numeric(truth["local_background"], errors="coerce").to_numpy(dtype=float)
        edges = _edges_from_values(values[eligible])
        rows.extend(
            _strata_for_edges(
                "local_background",
                values,
                eligible,
                recovered,
                edges,
                unit="as_declared_on_truth_column",
            )
        )
    else:
        notes["local_background"] = (
            "No local_background column on the truth table. A background map is "
            "not invented during a catalogue-only run. Supply the column or an image "
            "to the synthetic path, which can record an annulus median."
        )

    if "snr" in truth.columns:
        values = pd.to_numeric(truth["snr"], errors="coerce").to_numpy(dtype=float)
        edges = _edges_from_values(values[eligible])
        rows.extend(_strata_for_edges("truth_snr", values, eligible, recovered, edges, unit="snr"))
        notes["snr"] = "SNR bins use the truth-table SNR, which is defined for unrecovered sources."
    else:
        notes["snr"] = {
            "status": STATUS_NOT_APPLICABLE,
            "reason": (
                "Completeness is not stratified by detected SNR, because that "
                "quantity is undefined for unrecovered injections. Pass a truth SNR "
                "column to bin on an injected SNR."
            ),
        }

    # Mask stratum is a count report. Completeness stays on eligible rows only.
    n_masked = int((~eligible & np.array([reason == "masked" or str(reason).startswith("masked") for reason in truth["eligibility_reason"]])).sum()) if "eligibility_reason" in truth.columns else int((~eligible).sum())
    rows.append(
        {
            "stratum": "mask_status",
            "bin": "ineligible",
            "unit": "count",
            "n_eligible": 0,
            "n_recovered": 0,
            "n_unmatched_truth": 0,
            "n_ineligible": int((~eligible).sum()),
            "n_masked_family": n_masked,
            "completeness": None,
            "status": "NOT_IN_COMPLETENESS_DENOMINATOR",
            "wilson_lower": None,
            "wilson_upper": None,
            "clopper_pearson_lower": None,
            "clopper_pearson_upper": None,
        }
    )
    rows.append(
        {
            "stratum": "mask_status",
            "bin": "eligible",
            "unit": "count",
            "n_eligible": int(eligible.sum()),
            "n_recovered": int(recovered.sum()),
            "n_unmatched_truth": int(eligible.sum() - recovered.sum()),
            "completeness": None if eligible.sum() == 0 else float(recovered.sum() / eligible.sum()),
            "status": "DEFINED" if eligible.sum() else "UNDEFINED_ZERO_DENOMINATOR",
            "wilson_lower": None if wilson_interval(int(recovered.sum()), int(eligible.sum())) is None else wilson_interval(int(recovered.sum()), int(eligible.sum()))[0],
            "wilson_upper": None if wilson_interval(int(recovered.sum()), int(eligible.sum())) is None else wilson_interval(int(recovered.sum()), int(eligible.sum()))[1],
            "clopper_pearson_lower": None if clopper_pearson_interval(int(recovered.sum()), int(eligible.sum())) is None else clopper_pearson_interval(int(recovered.sum()), int(eligible.sum()))[0],
            "clopper_pearson_upper": None if clopper_pearson_interval(int(recovered.sum()), int(eligible.sum())) is None else clopper_pearson_interval(int(recovered.sum()), int(eligible.sum()))[1],
        }
    )

    frame = pd.DataFrame(rows)
    return frame, notes
