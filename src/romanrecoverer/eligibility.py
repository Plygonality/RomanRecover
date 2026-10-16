"""Decide which injected sources enter the completeness denominator.

An injected source is eligible when all of the following hold:

* its coordinates are finite (already enforced at load time)
* it is not marked masked
* if an image shape is known, its center lies inside the edge cut

The edge cut is a center cut in 0-based pixel coordinates, after converting
origin 1 to origin 0: ``margin <= x <= nx - 1 - margin`` and the same for
``y``. ``x`` is the column and ``y`` is the row. Sources outside that closed
interval are ineligible with reason ``edge_or_outside``. When no image shape
is supplied, the edge cut is not applied and the metrics say so.

Masked sources are ineligible even when they sit in the interior. They are
reported, and they are not allowed to consume a detection during matching.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from romanrecoverer.errors import DataError


@dataclass(frozen=True)
class EligibilityResult:
    eligible: np.ndarray
    reason: np.ndarray
    edge_exclusion_applied: bool
    image_shape: tuple[int, int] | None
    edge_margin_px: float
    pixel_origin: int


def _mask_is_set(value: object) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        raise DataError("Mask column contains a null. Refusing to guess whether the source is masked.")
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value) != 0
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            raise DataError("Mask column contains a non-finite value.")
        return float(value) != 0.0
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "masked", "in_mask"}:
        return True
    if text in {"0", "false", "no", "unmasked", ""}:
        return False
    raise DataError(f"Unrecognised mask value {value!r}. Use 0/1 or true/false.")


def evaluate_eligibility(
    x: np.ndarray,
    y: np.ndarray,
    mask: pd.Series | np.ndarray | None,
    *,
    image_shape: tuple[int, int] | None,
    edge_margin_px: float = 0.0,
    pixel_origin: int = 0,
) -> EligibilityResult:
    if edge_margin_px < 0:
        raise DataError("edge_margin_px must be >= 0.")
    if pixel_origin not in (0, 1):
        raise DataError(f"pixel origin must be 0 or 1, got {pixel_origin}.")

    x_arr = np.asarray(x, dtype=float).copy()
    y_arr = np.asarray(y, dtype=float).copy()
    if pixel_origin == 1:
        x_arr -= 1.0
        y_arr -= 1.0

    n = x_arr.shape[0]
    eligible = np.ones(n, dtype=bool)
    reason = np.array(["eligible"] * n, dtype=object)

    if mask is not None:
        mask_values = list(pd.Series(mask))
        if len(mask_values) != n:
            raise DataError("Mask column length does not match the truth catalogue.")
        for index, value in enumerate(mask_values):
            if _mask_is_set(value):
                eligible[index] = False
                reason[index] = "masked"

    edge_applied = image_shape is not None
    if image_shape is not None:
        if len(image_shape) != 2:
            raise DataError(f"image_shape must be (ny, nx), got {image_shape!r}.")
        ny, nx = int(image_shape[0]), int(image_shape[1])
        if ny <= 0 or nx <= 0:
            raise DataError(f"image_shape must be positive, got {(ny, nx)!r}.")
        x_lo = float(edge_margin_px)
        x_hi = (nx - 1) - float(edge_margin_px)
        y_lo = float(edge_margin_px)
        y_hi = (ny - 1) - float(edge_margin_px)
        if x_hi < x_lo or y_hi < y_lo:
            raise DataError(
                f"edge_margin_px={edge_margin_px} leaves no interior in an image of shape {(ny, nx)}."
            )
        outside = (x_arr < x_lo) | (x_arr > x_hi) | (y_arr < y_lo) | (y_arr > y_hi)
        for index in np.flatnonzero(outside):
            if reason[index] == "eligible":
                reason[index] = "edge_or_outside"
            elif reason[index] == "masked":
                reason[index] = "masked_and_edge_or_outside"
            eligible[index] = False

    return EligibilityResult(
        eligible=eligible,
        reason=reason,
        edge_exclusion_applied=edge_applied,
        image_shape=(int(image_shape[0]), int(image_shape[1])) if image_shape else None,
        edge_margin_px=float(edge_margin_px),
        pixel_origin=pixel_origin,
    )
