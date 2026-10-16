"""Load truth and detection catalogues and resolve column conventions.

The NGRST-Discovery-Toolkit contract used here is the one stated for the
current private package, which this environment could not read:

* truth columns: ``x``, ``y``, ``amplitude``, ``sigma``, ``class``
* detection columns: ``x_centroid``, ``y_centroid``, ``snr``, ``anomaly_score``

``amplitude`` is a Gaussian peak in the toolkit's synthetic image units. It is
not treated as an AB magnitude or a calibrated flux.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from romanrecoverer.errors import DataError
from romanrecoverer.schemas import (
    CALIBRATION_UNCALIBRATED_AMPLITUDE,
    CALIBRATION_UNKNOWN,
    DETECTION_SCHEMA,
    TRUTH_SCHEMA,
)

TRUTH_ID_CANDIDATES = ("source_id", "truth_id", "id")
DETECTION_ID_CANDIDATES = ("detection_id", "id", "source_id")


@dataclass(frozen=True)
class ColumnMap:
    """Resolved column names. Missing optional columns are ``None``."""

    x: str
    y: str
    id: str | None
    ra: str | None = None
    dec: str | None = None
    amplitude: str | None = None
    sigma: str | None = None
    flux: str | None = None
    mag: str | None = None
    snr: str | None = None
    mask: str | None = None
    source_class: str | None = None
    anomaly_score: str | None = None
    background: str | None = None
    adapter: str = "generic"


def load_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV or Parquet table. A header-only file is an empty catalogue."""

    path = Path(path)
    if not path.is_file():
        raise DataError(f"Catalogue not found: {path}")
    suffix = path.suffix.lower()
    try:
        if suffix == ".parquet":
            frame = pd.read_parquet(path)
        elif suffix in {".csv", ".txt"}:
            frame = pd.read_csv(path)
        else:
            raise DataError(
                f"Unsupported catalogue suffix {suffix!r} for {path}. Use .csv or .parquet."
            )
    except DataError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface a data error to the CLI
        raise DataError(f"Could not read catalogue {path}: {exc}") from exc
    if frame.shape[1] == 0:
        raise DataError(f"Catalogue {path} has no columns.")
    return frame


def load_sidecar(path: str | Path) -> dict[str, Any]:
    """Load ``<stem>.meta.json`` sitting next to a catalogue, if it exists."""

    path = Path(path)
    candidates = [
        path.with_suffix(".meta.json"),
        path.parent / f"{path.name}.meta.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise DataError(f"Sidecar {candidate} is not valid JSON: {exc}") from exc
            if not isinstance(payload, dict):
                raise DataError(f"Sidecar {candidate} must contain a JSON object.")
            payload["_sidecar_path"] = str(candidate)
            return payload
    return {}


def _first_present(columns: list[str], names: tuple[str, ...]) -> str | None:
    lookup = {column.lower(): column for column in columns}
    for name in names:
        if name in columns:
            return name
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def resolve_truth_columns(frame: pd.DataFrame, *, x_col: str | None = None, y_col: str | None = None) -> ColumnMap:
    columns = list(frame.columns)
    ngrst = {"x", "y", "amplitude", "sigma", "class"}.issubset(set(columns))
    x_name = x_col or _first_present(columns, ("x", "x_pix", "x_pixel", "xcentroid"))
    y_name = y_col or _first_present(columns, ("y", "y_pix", "y_pixel", "ycentroid"))
    if x_name is None or y_name is None:
        raise DataError(
            "Truth catalogue needs pixel columns x and y "
            f"(or pass --truth-x-col/--truth-y-col). Found columns: {columns}"
        )
    if x_name not in frame.columns or y_name not in frame.columns:
        raise DataError(f"Truth coordinate columns {x_name!r}, {y_name!r} are not in the table.")
    class_col = "class" if "class" in frame.columns else _first_present(columns, ("source_class", "profile"))
    return ColumnMap(
        x=x_name,
        y=y_name,
        id=_first_present(columns, TRUTH_ID_CANDIDATES),
        ra=_first_present(columns, ("ra", "ra_deg")),
        dec=_first_present(columns, ("dec", "dec_deg")),
        amplitude=_first_present(columns, ("amplitude", "peak_amplitude", "peak")),
        sigma=_first_present(columns, ("sigma", "sigma_px", "psf_sigma_px")),
        flux=_first_present(columns, ("flux", "flux_calibrated", "total_flux")),
        mag=_first_present(columns, ("mag", "ab_mag", "magnitude")),
        snr=_first_present(columns, ("snr", "snr_truth", "injected_snr")),
        mask=_first_present(columns, ("mask", "masked", "in_mask")),
        source_class=class_col,
        background=_first_present(columns, ("local_background", "background")),
        adapter="ngrst_truth" if ngrst else "generic_truth",
    )


def resolve_detection_columns(
    frame: pd.DataFrame,
    *,
    x_col: str | None = None,
    y_col: str | None = None,
) -> ColumnMap:
    columns = list(frame.columns)
    ngrst = {"x_centroid", "y_centroid"}.issubset(set(columns))
    x_name = x_col or _first_present(columns, ("x_centroid", "x", "x_pix", "x_pixel"))
    y_name = y_col or _first_present(columns, ("y_centroid", "y", "y_pix", "y_pixel"))
    if frame.empty and (x_name is None or y_name is None):
        raise DataError(
            "Detection catalogue has no x/y columns. An empty detection file still needs a header "
            "such as detection_id,x_centroid,y_centroid,snr,anomaly_score."
        )
    if x_name is None or y_name is None:
        raise DataError(
            "Detection catalogue needs x_centroid/y_centroid or x/y. "
            f"Found columns: {columns}"
        )
    return ColumnMap(
        x=x_name,
        y=y_name,
        id=_first_present(columns, DETECTION_ID_CANDIDATES),
        ra=_first_present(columns, ("ra", "ra_deg")),
        dec=_first_present(columns, ("dec", "dec_deg")),
        amplitude=_first_present(columns, ("amplitude", "peak_amplitude", "flux_auto")),
        sigma=_first_present(columns, ("sigma", "fwhm_px")),
        flux=_first_present(columns, ("flux", "flux_calibrated", "flux_auto")),
        mag=_first_present(columns, ("mag", "ab_mag", "magnitude")),
        snr="snr" if "snr" in frame.columns else _first_present(columns, ("snr",)),
        source_class=_first_present(columns, ("class", "source_class")),
        anomaly_score="anomaly_score" if "anomaly_score" in frame.columns else None,
        background=_first_present(columns, ("local_background", "background")),
        adapter="ngrst_detection" if ngrst else "generic_detection",
    )


def require_finite_coordinates(frame: pd.DataFrame, x_col: str, y_col: str, *, what: str) -> None:
    if frame.empty:
        return
    x_values = pd.to_numeric(frame[x_col], errors="coerce")
    y_values = pd.to_numeric(frame[y_col], errors="coerce")
    bad = ~np.isfinite(x_values.to_numpy(dtype=float)) | ~np.isfinite(y_values.to_numpy(dtype=float))
    if bool(bad.any()):
        count = int(bad.sum())
        raise DataError(
            f"{what} has {count} row(s) with non-finite {x_col}/{y_col}. "
            "Non-finite coordinates are rejected."
        )


def assign_ids(frame: pd.DataFrame, column: str | None, *, prefix: str) -> tuple[pd.Series, bool]:
    """Return string ids. Synthesized ids are the row order, and that fact is reported."""

    if column is None:
        ids = pd.Series([f"{prefix}-{index:04d}" for index in range(len(frame))], index=frame.index)
        return ids.astype(str), True
    ids = frame[column].astype(str)
    if ids.duplicated().any():
        duplicates = ids[ids.duplicated()].unique().tolist()
        raise DataError(f"Duplicate {column} values: {duplicates[:5]}")
    if ids.isna().any() or (ids.str.strip() == "").any() or (ids == "nan").any():
        raise DataError(f"{column} contains empty identifiers.")
    return ids, False


def calibration_from_columns(column_map: ColumnMap, sidecar: dict[str, Any]) -> str:
    declared = sidecar.get("calibration_status")
    if isinstance(declared, str) and declared.strip():
        return declared.strip()
    if column_map.adapter == "ngrst_truth":
        return CALIBRATION_UNCALIBRATED_AMPLITUDE
    if column_map.mag or column_map.flux:
        return CALIBRATION_UNKNOWN
    if column_map.amplitude:
        return CALIBRATION_UNCALIBRATED_AMPLITUDE
    return CALIBRATION_UNKNOWN


def schema_name_for(kind: str) -> str:
    if kind == "truth":
        return TRUTH_SCHEMA
    if kind == "detection":
        return DETECTION_SCHEMA
    raise DataError(f"Unknown catalogue kind {kind!r}.")
