"""Pixel and sky conversions using Astropy's high-level WCS API.

Conventions, from the Astropy WCS documentation:

* The high-level methods ``pixel_to_world`` and ``world_to_pixel`` are 0-based.
* Integer pixel values fall at pixel centers.
* Methods with "pixel" in the name use Cartesian ``(x, y)`` order, where ``x``
  is the column and ``y`` is the row.
* A Numpy image is indexed ``image[row, column]`` which is ``image[y, x]``.
* The low-level ``all_pix2world`` / ``all_world2pix`` methods take ``origin``
  of 0 or 1. FITS WCS keywords such as CRPIX are 1-based.

RomanRecoverer stores catalogues in the origin declared by the user (default 0)
and converts to the high-level 0-based API at the boundary.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from romanrecoverer.errors import DataError


def as_wcs(wcs: WCS | Any) -> WCS:
    if isinstance(wcs, WCS):
        return wcs
    raise DataError(f"Expected an astropy.wcs.WCS instance, got {type(wcs).__name__}.")


def wcs_from_header_mapping(mapping: dict[str, Any]) -> WCS:
    """Build a 2-D FITS WCS from a small JSON object.

    The object uses FITS semantics: ``crpix`` is 1-based. This helper is for
    demonstration tangent-plane WCS objects written by the synthetic generator.
    It is not a Roman GWCS and it does not apply detector distortion.
    """

    try:
        crpix = list(mapping["crpix"])
        crval = list(mapping["crval"])
        cdelt = list(mapping["cdelt"])
        ctype = list(mapping.get("ctype", ["RA---TAN", "DEC--TAN"]))
    except (KeyError, TypeError) as exc:
        raise DataError(f"WCS mapping is missing crpix/crval/cdelt: {exc}") from exc
    if len(crpix) != 2 or len(crval) != 2 or len(cdelt) != 2:
        raise DataError("WCS mapping must describe exactly two axes.")
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = crpix
    wcs.wcs.crval = crval
    wcs.wcs.cdelt = cdelt
    wcs.wcs.ctype = ctype
    wcs.wcs.cunit = ["deg", "deg"]
    return wcs


def to_high_level_pixels(
    x: np.ndarray,
    y: np.ndarray,
    *,
    origin: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert catalogue pixels to the 0-based high-level WCS convention."""

    if origin not in (0, 1):
        raise DataError(f"pixel origin must be 0 or 1, got {origin}.")
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if origin == 1:
        return x_arr - 1.0, y_arr - 1.0
    return x_arr, y_arr


def pixel_to_sky(
    wcs: WCS,
    x: np.ndarray,
    y: np.ndarray,
    *,
    origin: int = 0,
) -> SkyCoord:
    """Convert (x column, y row) to sky coordinates with the high-level API."""

    wcs = as_wcs(wcs)
    x0, y0 = to_high_level_pixels(x, y, origin=origin)
    return wcs.pixel_to_world(x0, y0)


def sky_to_pixel(
    wcs: WCS,
    sky: SkyCoord,
    *,
    origin: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert sky coordinates to (x column, y row) in the requested origin."""

    wcs = as_wcs(wcs)
    x0, y0 = wcs.world_to_pixel(sky)
    x0 = np.asarray(x0, dtype=float)
    y0 = np.asarray(y0, dtype=float)
    if origin == 1:
        return x0 + 1.0, y0 + 1.0
    return x0, y0


def low_level_pix2world(
    wcs: WCS,
    x: np.ndarray,
    y: np.ndarray,
    *,
    origin: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Low-level conversion. ``origin`` matches ``WCS.all_pix2world``."""

    wcs = as_wcs(wcs)
    if origin not in (0, 1):
        raise DataError(f"pixel origin must be 0 or 1, got {origin}.")
    ra, dec = wcs.all_pix2world(np.asarray(x, dtype=float), np.asarray(y, dtype=float), origin)
    return np.asarray(ra, dtype=float), np.asarray(dec, dtype=float)
