"""Completeness, position errors, photometric status, and precision.

Statistical intervals
---------------------
Completeness is treated as a binomial proportion ``k / n`` where ``n`` is the
number of eligible injections and ``k`` is the number recovered. The trials
are assumed independent. That assumption is violated by blending, spatially
correlated noise, and shared detection thresholds, so the intervals are
descriptive uncertainties under a binomial model, not a survey-level
covariance.

The primary interval is the Wilson score interval at 95% (z corresponding to
the two-sided normal quantile 0.975). Clopper-Pearson (beta quantiles, equal-
tailed) is reported beside it. A percentile bootstrap of the sample mean of
the Bernoulli outcomes is also reported. The bootstrap uses a NumPy
``Generator`` seeded independently of the injection seed. When every trial has
the same outcome the bootstrap interval collapses; Wilson and Clopper-Pearson
do not, and they are the intervals to quote for small ``n``.

Empty samples (``n = 0``) leave completeness undefined. A completeness of 0
with detections absent and ``n > 0`` is defined. Precision is undefined when
the positive-prediction denominator is 0.

False positives
---------------
An unmatched detection is not a false positive by itself. In a populated
image it may be a real source that was not in the injected list. Precision is
computed only under an explicit assumption:

* ``truth_complete_synthetic``: the scene is assumed to contain only the
  injected sources plus the declared noise. Unmatched detections are then
  counted as false positives *under that assumption*.
* a baseline catalogue: detections that match the baseline and not an
  injection are treated as pre-existing. Detections that match neither are
  candidate false positives under the baseline-subtraction assumption.

Otherwise precision is ``NOT_APPLICABLE``.

Photometry
----------
A bias is computed only when both sides are calibrated and share a unit and a
quantity kind (flux or magnitude). Gaussian peak amplitudes and SNR are not
that. When both sides have a peak amplitude in the same uncalibrated unit, a
relative amplitude ratio is reported separately and is not called a flux bias.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import beta, norm

from romanrecoverer.errors import DataError
from romanrecoverer.schemas import (
    ASSUMPTION_TRUTH_COMPLETE,
    CALIBRATION_CALIBRATED,
    STATUS_DEFINED,
    STATUS_NOT_APPLICABLE,
    STATUS_RELATIVE_AMPLITUDE,
    STATUS_UNDEFINED,
)

WILSON_Z = float(norm.ppf(0.975))


def wilson_interval(k: int, n: int, *, z: float = WILSON_Z) -> tuple[float, float] | None:
    if n < 0 or k < 0 or k > n:
        raise DataError(f"Binomial counts must satisfy 0 <= k <= n, got k={k}, n={n}.")
    if n == 0:
        return None
    phat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    half = z * np.sqrt(phat * (1.0 - phat) / n + z2 / (4.0 * n * n)) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def clopper_pearson_interval(k: int, n: int, *, alpha: float = 0.05) -> tuple[float, float] | None:
    if n < 0 or k < 0 or k > n:
        raise DataError(f"Binomial counts must satisfy 0 <= k <= n, got k={k}, n={n}.")
    if n == 0:
        return None
    if k == 0:
        lower = 0.0
    else:
        lower = float(beta.ppf(alpha / 2.0, k, n - k + 1))
    if k == n:
        upper = 1.0
    else:
        upper = float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lower, upper


def bootstrap_interval(
    recovered: np.ndarray,
    *,
    seed: int,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
) -> tuple[float, float] | None:
    outcomes = np.asarray(recovered, dtype=float)
    n = int(outcomes.shape[0])
    if n == 0:
        return None
    if n_bootstrap < 100:
        raise DataError("n_bootstrap must be at least 100.")
    generator = np.random.default_rng(seed)
    draws = generator.choice(outcomes, size=(n_bootstrap, n), replace=True)
    means = draws.mean(axis=1)
    lower, upper = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lower), float(upper)


def _interval_payload(interval: tuple[float, float] | None) -> dict[str, float | None]:
    if interval is None:
        return {"lower": None, "upper": None}
    return {"lower": interval[0], "upper": interval[1]}


def completeness_block(
    n_recovered: int,
    n_eligible: int,
    *,
    recovered_flags: np.ndarray,
    seed: int,
    n_bootstrap: int,
) -> dict[str, Any]:
    if n_eligible == 0:
        value = None
        status = STATUS_UNDEFINED
    else:
        value = n_recovered / n_eligible
        status = STATUS_DEFINED
    wilson = wilson_interval(n_recovered, n_eligible)
    clopper = clopper_pearson_interval(n_recovered, n_eligible)
    bootstrap = bootstrap_interval(recovered_flags, seed=seed, n_bootstrap=n_bootstrap)
    return {
        "completeness": value,
        "completeness_status": status,
        "confidence_intervals": {
            "alpha": 0.05,
            "primary": "wilson",
            "assumptions": (
                "Eligible injections are treated as independent Bernoulli trials. "
                "Blending, crowded photometry, and a shared detection threshold "
                "violate that assumption, so these intervals are binomial "
                "uncertainties rather than a full survey covariance. "
                "The Wilson interval is the one to quote. The percentile bootstrap "
                "resamples the observed recoveries with replacement and can "
                "collapse when every trial has the same outcome."
            ),
            "wilson": {**_interval_payload(wilson), "z": WILSON_Z},
            "clopper_pearson": _interval_payload(clopper),
            "bootstrap": {
                **_interval_payload(bootstrap),
                "n_bootstrap": n_bootstrap,
                "seed": seed,
                "percentile": [2.5, 97.5],
            },
        },
    }


def position_block(matches: pd.DataFrame) -> dict[str, Any]:
    recovered = matches[(matches["row_kind"] == "truth") & (matches["recovered"] == True)]  # noqa: E712
    if recovered.empty:
        return {
            "status": STATUS_UNDEFINED if matches.empty else "NO_RECOVERED_SOURCES",
            "unit": "pixel",
            "n": 0,
            "median_dx": None,
            "median_dy": None,
            "mean_separation": None,
            "rms_separation": None,
            "separation_percentiles": None,
        }
    dx = recovered["dx"].to_numpy(dtype=float)
    dy = recovered["dy"].to_numpy(dtype=float)
    separation = recovered["separation"].to_numpy(dtype=float)
    return {
        "status": STATUS_DEFINED,
        "unit": "pixel",
        "frame": "match_frame",
        "definition": "dx = x_detection - x_truth, dy = y_detection - y_truth, in the frame used for matching.",
        "n": int(len(recovered)),
        "median_dx": float(np.median(dx)),
        "median_dy": float(np.median(dy)),
        "mean_dx": float(np.mean(dx)),
        "mean_dy": float(np.mean(dy)),
        "mean_separation": float(np.mean(separation)),
        "rms_separation": float(np.sqrt(np.mean(separation**2))),
        "separation_percentiles": {
            "p16": float(np.percentile(separation, 16)),
            "p50": float(np.percentile(separation, 50)),
            "p84": float(np.percentile(separation, 84)),
        },
    }


def precision_block(
    *,
    n_recovered: int,
    n_unmatched_detected: int,
    dataset_assumption: str,
    n_baseline_explained: int | None,
) -> dict[str, Any]:
    """Precision under an explicit dataset assumption, or NOT_APPLICABLE."""

    policy = (
        "Unmatched detections are not called false positives unless the dataset "
        "assumption is truth_complete_synthetic, or a baseline catalogue has "
        "been subtracted. A populated image can contain real sources that were "
        "never injected."
    )
    if dataset_assumption == ASSUMPTION_TRUTH_COMPLETE and n_baseline_explained is None:
        false_positives = n_unmatched_detected
        denominator = n_recovered + false_positives
        if denominator == 0:
            value = None
            status = STATUS_UNDEFINED
        else:
            value = n_recovered / denominator
            status = "ESTIMATED_UNDER_TRUTH_COMPLETE_ASSUMPTION"
        return {
            "status": status,
            "value": value,
            "false_positive_count": false_positives,
            "labelled_unmatched_as_false_positive": True,
            "assumption": (
                "truth_complete_synthetic: the scene is assumed to contain only "
                "the injected sources and the declared noise. Unmatched detections "
                "are counted as false positives only under that assumption. This "
                "is not a demonstration that those detections are astrophysically spurious."
            ),
            "policy": policy,
        }

    if n_baseline_explained is not None:
        false_positives = n_unmatched_detected - n_baseline_explained
        if false_positives < 0:
            raise DataError("Baseline subtraction produced a negative false-positive count.")
        denominator = n_recovered + false_positives
        if denominator == 0:
            value = None
            status = STATUS_UNDEFINED
        else:
            value = n_recovered / denominator
            status = "ESTIMATED_AFTER_BASELINE_SUBTRACTION"
        return {
            "status": status,
            "value": value,
            "false_positive_count": false_positives,
            "baseline_explained_count": n_baseline_explained,
            "labelled_unmatched_as_false_positive": False,
            "assumption": (
                "Detections matching the baseline catalogue and not an injection "
                "are treated as pre-existing sources. Only detections that match "
                "neither the injections nor the baseline are candidate false positives."
            ),
            "policy": policy,
        }

    return {
        "status": STATUS_NOT_APPLICABLE,
        "value": None,
        "false_positive_count": None,
        "labelled_unmatched_as_false_positive": False,
        "reason": (
            "Unmatched detections in a nonempty or unspecified background are not "
            "automatically false positives. They may be legitimate pre-existing sources. "
            "Pass --assumption truth_complete_synthetic only for a scene that contains "
            "no astrophysical sources besides the injections, or pass --baseline."
        ),
        "assumption": dataset_assumption,
        "policy": policy,
    }


def _paired_numeric(matches: pd.DataFrame, truth: pd.DataFrame, detections: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray]:
    recovered = matches[(matches["row_kind"] == "truth") & (matches["recovered"] == True)]  # noqa: E712
    truth_values = []
    det_values = []
    for _, row in recovered.iterrows():
        truth_values.append(truth.iloc[int(row["truth_row"])][column] if column in truth.columns else np.nan)
        det_values.append(
            detections.iloc[int(row["detection_row"])][column] if column in detections.columns else np.nan
        )
    return np.asarray(truth_values, dtype=float), np.asarray(det_values, dtype=float)


def photometry_block(
    matches: pd.DataFrame,
    truth: pd.DataFrame,
    detections: pd.DataFrame,
    *,
    calibration_status: str,
    truth_unit: str | None,
    detection_unit: str | None,
    truth_kind: str | None,
    detection_kind: str | None,
) -> dict[str, Any]:
    """Photometric bias, or an explicit not-applicable status.

    NGRST truth amplitudes are Gaussian peaks. Detection SNR is a different
    quantity. Those are never subtracted.
    """

    note = (
        "Photometric recovery bias is reported only when the injected and "
        "measured values are calibrated and in the same unit and quantity. "
        "A Gaussian peak amplitude is not an AB magnitude and is not a physical flux."
    )
    if calibration_status != CALIBRATION_CALIBRATED:
        relative = _relative_amplitude(matches, truth, detections, truth_unit, detection_unit)
        return {
            "status": STATUS_NOT_APPLICABLE,
            "bias": None,
            "reason": (
                f"calibration_status is {calibration_status!r}, not 'calibrated'. "
                "Refusing to treat uncalibrated amplitudes as fluxes or magnitudes."
            ),
            "note": note,
            "relative_amplitude": relative,
        }
    if not truth_unit or not detection_unit or truth_unit != detection_unit:
        return {
            "status": STATUS_NOT_APPLICABLE,
            "bias": None,
            "reason": "Injected and measured photometric units differ or are undeclared.",
            "note": note,
            "relative_amplitude": None,
        }
    if truth_kind != detection_kind or truth_kind not in {"flux", "magnitude"}:
        return {
            "status": STATUS_NOT_APPLICABLE,
            "bias": None,
            "reason": "Photometric quantity kinds are not both flux or both magnitude.",
            "note": note,
            "relative_amplitude": None,
        }

    column = "mag" if truth_kind == "magnitude" else "flux"
    if column not in truth.columns or column not in detections.columns:
        return {
            "status": STATUS_NOT_APPLICABLE,
            "bias": None,
            "reason": f"Calibrated comparison requested but column {column!r} is missing.",
            "note": note,
            "relative_amplitude": None,
        }
    injected, measured = _paired_numeric(matches, truth, detections, column)
    good = np.isfinite(injected) & np.isfinite(measured)
    if truth_kind == "flux":
        good = good & (injected != 0)
        residual = (measured[good] - injected[good]) / injected[good]
        residual_name = "fractional_flux_residual_(measured-injected)/injected"
    else:
        residual = measured[good] - injected[good]
        residual_name = "magnitude_residual_measured_minus_injected"
    if residual.size == 0:
        return {
            "status": "NO_PAIRED_MEASUREMENTS",
            "bias": None,
            "reason": "No recovered source had finite calibrated values on both sides.",
            "note": note,
            "relative_amplitude": None,
        }
    return {
        "status": STATUS_DEFINED,
        "unit": truth_unit,
        "quantity": truth_kind,
        "residual_definition": residual_name,
        "n": int(residual.size),
        "bias": {
            "median": float(np.median(residual)),
            "mean": float(np.mean(residual)),
            "p16": float(np.percentile(residual, 16)),
            "p84": float(np.percentile(residual, 84)),
        },
        "note": note,
        "relative_amplitude": None,
    }


def _relative_amplitude(
    matches: pd.DataFrame,
    truth: pd.DataFrame,
    detections: pd.DataFrame,
    truth_unit: str | None,
    detection_unit: str | None,
) -> dict[str, Any] | None:
    if "amplitude" not in truth.columns or "amplitude" not in detections.columns:
        return None
    if truth_unit and detection_unit and truth_unit != detection_unit:
        return {
            "status": STATUS_NOT_APPLICABLE,
            "reason": "Amplitude units differ, so even a relative ratio is not reported.",
        }
    injected, measured = _paired_numeric(matches, truth, detections, "amplitude")
    good = np.isfinite(injected) & np.isfinite(measured) & (injected != 0)
    if not np.any(good):
        return None
    ratio = measured[good] / injected[good]
    return {
        "status": STATUS_RELATIVE_AMPLITUDE,
        "reason": (
            "Both catalogues carry an amplitude column in a compatible uncalibrated "
            "unit. The ratio is a relative peak-amplitude comparison, not a flux bias "
            "and not a magnitude."
        ),
        "n": int(good.sum()),
        "median_measured_over_injected": float(np.median(ratio)),
    }
