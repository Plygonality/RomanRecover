"""Read and write small synthetic image bundles.

Observational FITS, ASDF, and NPZ products are not committed. The synthetic
writer stores a Numpy ``.npz`` plus a JSON metadata sidecar so the array,
mask, and calibration status stay together.

An image array is indexed ``image[y, x]`` = ``image[row, column]``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from romanrecoverer.errors import DataError
from romanrecoverer.wcsutil import wcs_from_header_mapping


def _metadata_path(path: Path) -> Path:
    return path.with_suffix(".json") if path.suffix == ".npz" else path.with_suffix(path.suffix + ".json")


def write_npz_image(
    path: str | Path,
    image: np.ndarray,
    *,
    mask: np.ndarray | None,
    metadata: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.asarray(image)
    if image.ndim != 2:
        raise DataError(f"Synthetic images must be 2-D, got shape {image.shape}.")
    payload: dict[str, Any] = {"image": image}
    if mask is not None:
        mask_arr = np.asarray(mask)
        if mask_arr.shape != image.shape:
            raise DataError(f"Mask shape {mask_arr.shape} does not match image shape {image.shape}.")
        payload["mask"] = mask_arr.astype(np.uint8)
    np.savez_compressed(path, **payload)
    meta_path = path.with_name(path.stem + ".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_image(path: str | Path) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise DataError(f"Image not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path) as bundle:
            if "image" not in bundle:
                raise DataError(f"{path} has no 'image' array.")
            image = np.array(bundle["image"])
            mask = np.array(bundle["mask"]) if "mask" in bundle else None
        meta_path = path.with_name(path.stem + ".meta.json")
        metadata: dict[str, Any] = {}
        if meta_path.is_file():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        return image, mask, metadata
    if suffix in {".fits", ".fit", ".fts"}:
        with fits.open(path) as hdul:
            image = np.array(hdul[0].data)
            header = dict(hdul[0].header)
        if image is None or np.asarray(image).ndim != 2:
            raise DataError(f"{path} does not contain a 2-D image in the primary HDU.")
        return np.asarray(image), None, {"fits_header_cards": {k: _jsonable(v) for k, v in header.items()}}
    raise DataError(f"Unsupported image type {suffix}. Use .npz or a 2-D FITS image.")


def _jsonable(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def wcs_from_image_metadata(metadata: dict[str, Any]):
    mapping = metadata.get("wcs")
    if not isinstance(mapping, dict):
        return None
    return wcs_from_header_mapping(mapping)


def annulus_median(
    image: np.ndarray,
    x: float,
    y: float,
    *,
    inner: float,
    outer: float,
) -> float:
    """Median of pixels in an annulus around a center.

    ``x`` is the column and ``y`` is the row, in 0-based pixels. This is a
    catalogue-free background proxy for synthetic images. It is not a
    photutils background model and it is not a Roman sky subtraction.
    """

    if outer <= inner or inner < 0:
        raise DataError("Annulus radii must satisfy 0 <= inner < outer.")
    yy, xx = np.indices(image.shape)
    radius = np.hypot(xx - x, yy - y)
    annulus = (radius >= inner) & (radius <= outer)
    if not np.any(annulus):
        return float("nan")
    return float(np.median(image[annulus]))
