"""Controlled synthetic injection.

This is workflow B. It draws point sources and simple extended profiles into a
Numpy image under an explicit noise model. It does not call romanisim, and a
failure of the Roman I-Sim adapter must not land here.

Amplitude convention
--------------------
``peak_amplitude`` is the value of the profile at its center, in image units.
For a Gaussian, the integral over the infinite plane is
``peak_amplitude * 2 * pi * sigma**2``. That integral is stored as
``integrated_proxy`` and is still not a calibrated flux unless the image unit
itself is a calibrated flux, which the default dimensionless mode is not.

AB magnitudes are refused unless the caller supplies a complete, explicit
calibration. This generator does not invent a zeropoint.

Invalid values
--------------
Non-finite or negative peak amplitudes always raise ``DataError``. A peak of
exactly zero raises ``DataError`` unless ``allow_zero`` is set, in which case
the source is written with ``eligible=False`` and reason ``zero_amplitude``.
Non-positive or non-finite profile widths always raise. Image shapes must be
positive 2-D integers.

Seeds
-----
One integer seed is split with ``numpy.random.SeedSequence.spawn`` into
independent streams for positions, amplitudes, and noise. Repeating the seed
repeats the source ids and placements. A different seed changes the placements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from romanrecoverer.errors import DataError
from romanrecoverer.image_io import annulus_median, write_npz_image
from romanrecoverer.provenance import utc_now, write_json
from romanrecoverer.schemas import (
    ASSUMPTION_TRUTH_COMPLETE,
    CALIBRATION_DIMENSIONLESS,
    MODE_SYNTHETIC,
    SCHEMA_VERSION,
    SYNTH_SCHEMA,
)

_FORBIDDEN_UNITS = {"ab_mag", "abmag", "magnitude", "mag", "jansky", "jy", "mjy"}


@dataclass
class Scene:
    image: np.ndarray
    mask: np.ndarray
    truth: pd.DataFrame
    metadata: dict[str, Any]


def _reject_forbidden_unit(unit: str) -> None:
    if unit.strip().lower() in _FORBIDDEN_UNITS:
        raise DataError(
            f"Refusing image or amplitude unit {unit!r}. A Gaussian peak amplitude is not an "
            "AB magnitude or a physical flux. Declare a dimensionless or explicitly calibrated "
            "image unit, and do not pass magnitudes without a zeropoint this tool does not apply."
        )


def _validate_shape(image_shape: tuple[int, int] | list[int]) -> tuple[int, int]:
    if len(image_shape) != 2:
        raise DataError(f"image_shape must be [ny, nx], got {image_shape!r}.")
    ny, nx = image_shape
    if isinstance(ny, bool) or isinstance(nx, bool):
        raise DataError("image_shape entries must be integers, not booleans.")
    try:
        ny_i = int(ny)
        nx_i = int(nx)
    except (TypeError, ValueError) as exc:
        raise DataError(f"image_shape entries must be integers, got {image_shape!r}.") from exc
    if ny_i != ny or nx_i != nx:
        raise DataError(f"image_shape entries must be integers, got {image_shape!r}.")
    if ny_i <= 0 or nx_i <= 0:
        raise DataError(f"image_shape must be positive, got {(ny_i, nx_i)}.")
    return ny_i, nx_i


def _check_peak(value: float, *, allow_zero: bool, source_id: str) -> str | None:
    if not np.isfinite(value):
        raise DataError(f"Source {source_id} has a non-finite peak amplitude ({value}).")
    if value < 0:
        raise DataError(
            f"Source {source_id} has a negative peak amplitude ({value}). "
            "Negative injections are rejected."
        )
    if value == 0:
        if allow_zero:
            return "zero_amplitude"
        raise DataError(
            f"Source {source_id} has a peak amplitude of zero. Zero-flux injections are "
            "rejected by default because they are not sources. Pass allow_zero to record "
            "them as ineligible instead."
        )
    return None


def _check_scale(value: float, *, source_id: str, name: str) -> None:
    if not np.isfinite(value) or value <= 0:
        raise DataError(f"Source {source_id} has an invalid {name} ({value}). Widths must be finite and > 0.")


def _stamp_gaussian(image: np.ndarray, x: float, y: float, peak: float, sigma: float) -> float:
    """Add a Gaussian. Returns the discrete sum of the stamp, not a calibrated flux."""

    radius = max(1, int(math.ceil(8.0 * sigma)))
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    y_start = max(0, y0 - radius)
    y_stop = min(image.shape[0], y0 + radius + 1)
    x_start = max(0, x0 - radius)
    x_stop = min(image.shape[1], x0 + radius + 1)
    if y_start >= y_stop or x_start >= x_stop:
        return 0.0
    yy, xx = np.mgrid[y_start:y_stop, x_start:x_stop]
    model = peak * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma * sigma))
    image[y_start:y_stop, x_start:x_stop] += model
    return float(model.sum())


def _stamp_exponential(image: np.ndarray, x: float, y: float, peak: float, scale: float) -> float:
    """Add a circular exponential. ``peak`` is the central value, not a total flux."""

    radius = max(1, int(math.ceil(12.0 * scale)))
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    y_start = max(0, y0 - radius)
    y_stop = min(image.shape[0], y0 + radius + 1)
    x_start = max(0, x0 - radius)
    x_stop = min(image.shape[1], x0 + radius + 1)
    if y_start >= y_stop or x_start >= x_stop:
        return 0.0
    yy, xx = np.mgrid[y_start:y_stop, x_start:x_stop]
    rr = np.hypot(xx - x, yy - y)
    model = peak * np.exp(-rr / scale)
    image[y_start:y_stop, x_start:x_stop] += model
    return float(model.sum())


def render_sources(
    image: np.ndarray,
    sources: pd.DataFrame,
    *,
    allow_zero: bool = False,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Add sources onto a copy of ``image`` and return that copy plus bookkeeping.

    The input array is not modified.
    """

    if image.ndim != 2:
        raise DataError(f"Image must be 2-D, got ndim={image.ndim}.")
    canvas = np.array(image, dtype=float, copy=True)
    records = []
    for index, row in sources.reset_index(drop=True).iterrows():
        source_id = str(row.get("source_id", f"src-{index + 1:04d}"))
        peak = float(row["peak_amplitude"])
        profile = str(row["profile"])
        zero_reason = _check_peak(peak, allow_zero=allow_zero, source_id=source_id)
        x = float(row["x"])
        y = float(row["y"])
        if not np.isfinite(x) or not np.isfinite(y):
            raise DataError(f"Source {source_id} has non-finite coordinates.")
        if profile == "gaussian":
            sigma = float(row["sigma"])
            _check_scale(sigma, source_id=source_id, name="sigma")
            discrete_sum = 0.0 if zero_reason else _stamp_gaussian(canvas, x, y, peak, sigma)
            integrated_proxy = peak * 2.0 * math.pi * sigma * sigma
            scale = sigma
        elif profile == "exponential":
            scale = float(row["scale_px"])
            _check_scale(scale, source_id=source_id, name="scale_px")
            discrete_sum = 0.0 if zero_reason else _stamp_exponential(canvas, x, y, peak, scale)
            # Integral of peak * exp(-r/h) over the plane is 2 * pi * peak * h^2.
            integrated_proxy = 2.0 * math.pi * peak * scale * scale
            sigma = scale
        else:
            raise DataError(
                f"Source {source_id} has profile {profile!r}. Supported profiles are "
                "'gaussian' and 'exponential'."
            )
        records.append(
            {
                "source_id": source_id,
                "discrete_sum": discrete_sum,
                "integrated_proxy": integrated_proxy,
                "sigma_recorded": sigma,
                "zero_reason": zero_reason,
            }
        )
    bookkeeping = pd.DataFrame(records)
    canvas.flags.writeable = True
    return canvas, bookkeeping


def _sample_range(generator: np.random.Generator, count: int, low: float, high: float) -> np.ndarray:
    if count < 0:
        raise DataError("Source counts must be >= 0.")
    if not np.isfinite(low) or not np.isfinite(high) or high < low:
        raise DataError(f"Amplitude or scale range {(low, high)} is invalid.")
    if count == 0:
        return np.array([], dtype=float)
    if high == low:
        return np.full(count, low, dtype=float)
    return generator.uniform(low, high, size=count)


def generate_scene(config: dict[str, Any]) -> Scene:
    """Build a seeded synthetic scene from a configuration mapping."""

    mode = str(config.get("mode", MODE_SYNTHETIC))
    if mode != MODE_SYNTHETIC and mode != "dimensionless_synthetic":
        raise DataError(
            f"Synthetic injection only builds mode {MODE_SYNTHETIC!r} "
            f"(or the unit label 'dimensionless_synthetic'). Got {mode!r}. "
            "Roman I-Sim injection is a separate adapter and is never used as a fallback."
        )
    image_unit = str(config.get("image_unit", "dimensionless"))
    amplitude_unit = str(config.get("amplitude_unit", "dimensionless_peak"))
    _reject_forbidden_unit(image_unit)
    _reject_forbidden_unit(amplitude_unit)
    if config.get("ab_mag") or config.get("zeropoint_ab"):
        raise DataError(
            "Refusing to interpret a zeropoint or AB magnitude inside the synthetic generator. "
            "Peak amplitude stays in image units."
        )
    if config.get("poisson") is True and image_unit != "electron":
        raise DataError(
            "Poisson noise is refused unless image_unit is 'electron'. "
            "A dimensionless image is not a count image."
        )

    ny, nx = _validate_shape(tuple(config.get("image_shape", [64, 64])))
    seed = int(config.get("seed", 0))
    margin = float(config.get("edge_margin_px", 2.0))
    if margin < 0 or margin * 2 >= min(nx, ny):
        raise DataError(f"edge_margin_px={margin} is incompatible with image shape {(ny, nx)}.")

    sequences = np.random.SeedSequence(seed).spawn(3)
    position_rng = np.random.default_rng(sequences[0])
    amplitude_rng = np.random.default_rng(sequences[1])
    noise_rng = np.random.default_rng(sequences[2])

    psf = config.get("psf", {})
    psf_sigma = float(psf.get("sigma_px", 1.2))
    if not np.isfinite(psf_sigma) or psf_sigma <= 0:
        raise DataError(f"PSF sigma_px must be finite and > 0, got {psf_sigma}.")
    if str(psf.get("kind", "gaussian")) != "gaussian":
        raise DataError("The synthetic PSF kind must be 'gaussian'. Other PSFs are not implied.")

    point = config.get("point_sources", {})
    extended = config.get("extended_sources", {})
    n_point = int(point.get("n", 8))
    n_extended = int(extended.get("n", 2))
    allow_zero = bool(config.get("allow_zero", False))

    x_lo, x_hi = margin, (nx - 1) - margin
    y_lo, y_hi = margin, (ny - 1) - margin
    rows: list[dict[str, Any]] = []

    def add_population(count: int, profile: str, block: dict[str, Any], prefix: str) -> None:
        peaks = _sample_range(
            amplitude_rng,
            count,
            float(block.get("peak_amplitude_min", 1.0)),
            float(block.get("peak_amplitude_max", 5.0)),
        )
        if profile == "gaussian":
            scales = np.full(count, psf_sigma, dtype=float)
        else:
            scales = _sample_range(
                amplitude_rng,
                count,
                float(block.get("scale_px_min", 1.5)),
                float(block.get("scale_px_max", 3.0)),
            )
        xs = position_rng.uniform(x_lo, x_hi, size=count) if count else np.array([])
        ys = position_rng.uniform(y_lo, y_hi, size=count) if count else np.array([])
        for index in range(count):
            rows.append(
                {
                    "source_id": f"{prefix}-{index + 1:04d}",
                    "x": float(xs[index]),
                    "y": float(ys[index]),
                    "peak_amplitude": float(peaks[index]),
                    "profile": profile,
                    "sigma": float(scales[index]) if profile == "gaussian" else float(psf_sigma),
                    "scale_px": float(scales[index]),
                    "class": "point" if profile == "gaussian" else "extended",
                }
            )

    add_population(n_point, "gaussian", point, "pt")
    extended_profile = str(extended.get("profile", "exponential"))
    if extended_profile not in {"exponential", "gaussian"}:
        raise DataError(f"Extended profile {extended_profile!r} is not supported.")
    add_population(n_extended, extended_profile, extended, "ex")

    sources = pd.DataFrame(rows)
    background = float(config.get("background", 0.0))
    if not np.isfinite(background):
        raise DataError("background must be finite.")
    noise_sigma = float(config.get("noise_sigma", 0.0))
    if not np.isfinite(noise_sigma) or noise_sigma < 0:
        raise DataError("noise_sigma must be finite and >= 0.")

    image = np.full((ny, nx), background, dtype=float)
    if sources.empty:
        canvas = image
        bookkeeping = pd.DataFrame(columns=["source_id", "discrete_sum", "integrated_proxy", "sigma_recorded", "zero_reason"])
    else:
        canvas, bookkeeping = render_sources(image, sources, allow_zero=allow_zero)
    if noise_sigma > 0:
        canvas = canvas + noise_rng.normal(0.0, noise_sigma, size=canvas.shape)
    if config.get("poisson") is True:
        if np.any(canvas < 0):
            raise DataError("Poisson draw refused because the electron image has negative pixels.")
        canvas = noise_rng.poisson(canvas).astype(float)

    mask = np.zeros(canvas.shape, dtype=np.uint8)
    mask_cfg = config.get("mask", {})
    if mask_cfg.get("enabled"):
        y0, y1 = int(mask_cfg.get("y0", 0)), int(mask_cfg.get("y1", 0))
        x0, x1 = int(mask_cfg.get("x0", 0)), int(mask_cfg.get("x1", 0))
        mask[y0:y1, x0:x1] = 1

    truth_rows = []
    for source, extra in zip(sources.to_dict(orient="records"), bookkeeping.to_dict(orient="records"), strict=True):
        x = float(source["x"])
        y = float(source["y"])
        col = int(np.clip(round(x), 0, nx - 1))
        row = int(np.clip(round(y), 0, ny - 1))
        in_mask = bool(mask[row, col] == 1)
        reason = extra["zero_reason"]
        if in_mask and reason is None:
            reason = "masked"
        elif in_mask and reason is not None:
            reason = f"{reason}_and_masked"
        local_background = annulus_median(canvas, x, y, inner=4.0, outer=8.0)
        truth_rows.append(
            {
                "source_id": source["source_id"],
                "x": x,
                "y": y,
                "amplitude": source["peak_amplitude"],
                "sigma": source["sigma"] if source["profile"] == "gaussian" else source["scale_px"],
                "class": source["class"],
                "profile": source["profile"],
                "peak_amplitude": source["peak_amplitude"],
                "integrated_proxy": extra["integrated_proxy"],
                "discrete_sum": extra["discrete_sum"],
                "mask": int(in_mask or reason is not None),
                "eligibility_note": reason or "",
                "local_background": local_background,
                "snr": (source["peak_amplitude"] / noise_sigma) if noise_sigma > 0 else np.nan,
            }
        )
    truth = pd.DataFrame(truth_rows)
    metadata = {
        "schema_name": SYNTH_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "mode": MODE_SYNTHETIC,
        "calibration_status": str(config.get("calibration_status", CALIBRATION_DIMENSIONLESS)),
        "dataset_assumption": str(config.get("dataset_assumption", ASSUMPTION_TRUTH_COMPLETE)),
        "image_unit": image_unit,
        "amplitude_unit": amplitude_unit,
        "amplitude_meaning": (
            "peak value of the declared profile at the source center, in image units. "
            "Not an AB magnitude. Not a Roman calibrated flux. "
            "integrated_proxy is the analytic integral of that profile and inherits the same limitation."
        ),
        "sigma_unit": "pixel",
        "sigma_meaning": (
            "For gaussian profiles, sigma is the Gaussian standard deviation in pixels. "
            "For exponential profiles, the truth sigma column stores the exponential scale length "
            "in pixels. Neither is a half-light radius from romanisim."
        ),
        "snr_meaning": (
            "snr on the synthetic truth table is peak_amplitude / noise_sigma when noise_sigma > 0. "
            "It is a brightness proxy over the declared noise, not a detection SNR and not a Roman photometric SNR."
        ),
        "coordinate_frame": "pixel",
        "pixel_origin": 0,
        "axis_order": "x_is_column_y_is_row",
        "image_shape": [ny, nx],
        "seed": seed,
        "seed_streams": ["positions", "amplitudes", "noise"],
        "edge_margin_px": margin,
        "psf": {"kind": "gaussian", "sigma_px": psf_sigma},
        "noise": {
            "background": background,
            "noise_sigma": noise_sigma,
            "distribution": "gaussian" if noise_sigma > 0 else "none",
            "poisson": bool(config.get("poisson") is True),
        },
        "gain": config.get("gain"),
        "exposure_s": config.get("exposure_s"),
        "gain_note": (
            "gain and exposure_s are recorded when supplied. They are not applied as a "
            "magnitude calibration."
        ),
        "wcs_kind": "none",
        "partition": config.get("partition", "synthetic-default"),
        "detector_configuration": None,
        "notes": config.get(
            "notes",
            "Dimensionless synthetic demonstration. Not a Roman observation and not a romanisim product.",
        ),
    }
    if config.get("include_demo_wcs"):
        # Approximate WFI plate scale, 0.11 arcsec/pixel, used only so tests can
        # round-trip a tangent plane. This is not a Roman GWCS and not a distortion solution.
        plate_scale_deg = 0.11 / 3600.0
        metadata["wcs"] = {
            "crpix": [1.0, 1.0],
            "crval": [270.0, 66.0],
            "cdelt": [plate_scale_deg, plate_scale_deg],
            "ctype": ["RA---TAN", "DEC--TAN"],
        }
        metadata["wcs_kind"] = "tangent_plane_demonstration"
        metadata["wcs_note"] = (
            "Demonstration TAN WCS with CRPIX 1-based. Plate scale 0.11 arcsec/pixel "
            "is an approximate Roman WFI scale for the round trip only."
        )
    return Scene(image=canvas, mask=mask, truth=truth, metadata=metadata)


def default_config() -> dict[str, Any]:
    return {
        "schema_name": SYNTH_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "mode": MODE_SYNTHETIC,
        "calibration_status": CALIBRATION_DIMENSIONLESS,
        "dataset_assumption": ASSUMPTION_TRUTH_COMPLETE,
        "seed": 7,
        "image_shape": [48, 48],
        "background": 0.0,
        "noise_sigma": 0.02,
        "edge_margin_px": 4.0,
        "image_unit": "dimensionless",
        "amplitude_unit": "dimensionless_peak",
        "psf": {"kind": "gaussian", "sigma_px": 1.1},
        "point_sources": {"n": 10, "peak_amplitude_min": 0.8, "peak_amplitude_max": 6.0},
        "extended_sources": {
            "n": 2,
            "profile": "exponential",
            "peak_amplitude_min": 1.0,
            "peak_amplitude_max": 3.0,
            "scale_px_min": 1.5,
            "scale_px_max": 2.5,
        },
        "partition": "preset-controlled",
        "notes": (
            "Input-free preset. Peak amplitudes are dimensionless profile peaks, "
            "not AB magnitudes and not calibrated fluxes."
        ),
    }


def write_scene(scene: Scene, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    image_path = out / "image.npz"
    truth_path = out / "truth.csv"
    meta_path = out / "truth.meta.json"
    write_npz_image(image_path, scene.image, mask=scene.mask, metadata=scene.metadata)
    scene.truth.to_csv(truth_path, index=False)
    write_json(meta_path, _truth_sidecar(scene.metadata))
    return {"image": image_path, "truth": truth_path, "truth_meta": meta_path}


def _truth_sidecar(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_name": "romanrecoverer.table_meta",
        "schema_version": SCHEMA_VERSION,
        "coordinate_frame": "pixel",
        "pixel_origin": 0,
        "axis_order": "x_is_column_y_is_row",
        "image_shape": metadata["image_shape"],
        "edge_margin_px": metadata["edge_margin_px"],
        "calibration_status": metadata["calibration_status"],
        "amplitude_unit": metadata["amplitude_unit"],
        "amplitude_meaning": metadata["amplitude_meaning"],
        "dataset_assumption": metadata["dataset_assumption"],
        "mode": metadata["mode"],
        "seed": metadata["seed"],
        "partition": metadata["partition"],
        "notes": metadata["notes"],
    }
