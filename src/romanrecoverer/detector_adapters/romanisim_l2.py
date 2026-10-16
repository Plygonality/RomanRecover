"""Roman I-Sim L2 injection adapter, pinned to the v0.15.0 contract.

The callable is ``romanisim.image.inject_sources_into_l2``. The parameter list
below was read from the v0.15.0 tag of
https://github.com/spacetelescope/romanisim/blob/v0.15.0/romanisim/image.py
and matches the readthedocs page for that function:

    inject_sources_into_l2(model, cat, x=None, y=None, psf=None, seed=50,
                           rng=None, gain=None, psftype='epsf')

v0.15.0 behaviour that this adapter relies on, from that same source:

* If ``x`` or ``y`` is omitted, positions come from
  ``model.meta.wcs.numerical_inverse(cat['ra'], cat['dec'])``.
* The filter is ``model.meta.instrument.optical_element``.
* The catalog is passed through ``romanisim.catalog.table_to_catalog`` with
  that filter name, so the table needs ``ra``, ``dec``, ``type``, ``n``,
  ``half_light_radius``, ``pa``, ``ba``, and a flux column named for the filter.
  Fluxes are maggies (AB magnitude = -2.5 log10(flux)), per the catalog docs.
* The SCA index is ``int(model.meta.instrument.detector[-2:])``.
* The PSF date is ``model.meta.exposure.start_time`` when ``psf`` is omitted.
* The read pattern is ``model.meta.exposure.read_pattern``.
* The function deep-copies the model and returns the copy.
* Gain defaults to ``romanisim.models.parameters.reference_data['gain']``
  in that tag (``image.py`` imports ``parameters`` from ``romanisim.models``).
  The latest readthedocs prose instead says the image photometric calibration
  sets the total flux and that gain only scales Poisson noise. Those
  descriptions disagree. This adapter records the installed version and will
  not call an unpinned version unless ``accept_unpinned`` is set. It still
  does not invent a call.

Documented L2 shortcuts, from the Roman I-Sim image and L2 docs:

* Pixels with cosmic-ray hits are refit as if the hit were absent. The L2
  file does not say which pixels were hit.
* Nonlinearity and saturation are not included.
* The ramp is refit without persistence, while persistence already present in
  the original slope remains.

If romanisim or roman_datamodels is not installed, the status is
``SKIPPED_DEPENDENCY``. A failed call is raised. This module never falls back
to the Gaussian synthetic injector.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any

from romanrecoverer.errors import DataError, DependencyError
from romanrecoverer.schemas import (
    ROMANISIM_INJECT_PARAMS,
    ROMANISIM_PINNED_VERSION,
    ROMANISIM_V0150_IMAGING_FILTERS,
    STATUS_SKIPPED,
)

CATALOG_COLUMNS = (
    "ra",
    "dec",
    "type",
    "n",
    "half_light_radius",
    "pa",
    "ba",
)


def probe_romanisim() -> dict[str, Any]:
    """Inspect the installed library. Does not download reference files."""

    payload: dict[str, Any] = {
        "pinned_version": ROMANISIM_PINNED_VERSION,
        "expected_parameters": list(ROMANISIM_INJECT_PARAMS),
        "available": False,
        "status": STATUS_SKIPPED,
    }
    try:
        romanisim = importlib.import_module("romanisim")
        image = importlib.import_module("romanisim.image")
    except ImportError as exc:
        payload["reason"] = f"romanisim is not installed ({exc})."
        return payload
    function = getattr(image, "inject_sources_into_l2", None)
    if function is None:
        payload["status"] = "INCOMPATIBLE_API"
        payload["reason"] = "romanisim.image.inject_sources_into_l2 is missing."
        payload["romanisim_version"] = getattr(romanisim, "__version__", None)
        return payload
    signature = inspect.signature(function)
    payload.update(
        {
            "available": True,
            "status": "AVAILABLE",
            "romanisim_version": getattr(romanisim, "__version__", None),
            "installed_parameters": list(signature.parameters),
            "signature": str(signature),
            "doc_first_paragraph": _first_paragraph(function.__doc__),
        }
    )
    missing = [name for name in ("model", "cat") if name not in signature.parameters]
    if missing:
        payload["status"] = "INCOMPATIBLE_API"
        payload["available"] = False
        payload["reason"] = f"Installed signature is missing {missing}."
    version = str(payload.get("romanisim_version") or "")
    payload["version_matches_pin"] = version == ROMANISIM_PINNED_VERSION
    if version != ROMANISIM_PINNED_VERSION:
        payload["pin_note"] = (
            f"Installed romanisim {version or 'unknown'} is not the pinned "
            f"{ROMANISIM_PINNED_VERSION}. Injection is refused unless accept_unpinned is set, "
            "and even then the call uses only parameters present in the live signature."
        )
    try:
        datamodels = importlib.import_module("roman_datamodels")
        payload["roman_datamodels_version"] = getattr(datamodels, "__version__", None)
    except ImportError as exc:
        payload["roman_datamodels_version"] = None
        payload["roman_datamodels_import_error"] = str(exc)
    return payload


def _first_paragraph(doc: str | None) -> str | None:
    if not doc:
        return None
    return " ".join(doc.strip().splitlines()[:4])


def validate_catalog_table(catalog: Any, *, filter_name: str) -> None:
    """Check the romanisim 0.15.0 catalog columns without importing romanisim."""

    colnames = _column_names(catalog)
    missing = [name for name in CATALOG_COLUMNS if name not in colnames]
    if missing:
        raise DataError(
            "Roman I-Sim catalog is missing columns "
            f"{missing}. Required by romanisim.catalog.table_to_catalog in v0.15.0: "
            f"{list(CATALOG_COLUMNS)} plus a flux column named {filter_name!r} (maggies)."
        )
    if filter_name not in colnames:
        raise DataError(
            f"Catalog has no flux column named {filter_name!r}, which is the image optical element. "
            "Fluxes must be maggies for that bandpass. They are not Gaussian peak amplitudes."
        )
    types = _column_values(catalog, "type")
    bad = sorted({value for value in types if value not in {"PSF", "SER"}})
    if bad:
        raise DataError(f"Catalog type values must be PSF or SER. Found {bad}.")


def validate_l2_model(model: Any) -> dict[str, Any]:
    """Read the metadata paths used by romanisim 0.15.0. Refuse anything else.

    This does not construct a datamodel and does not substitute a Gaussian image.
    """

    class_name = type(model).__name__
    if class_name != "ImageModel":
        raise DataError(
            f"L2 injection requires a roman_datamodels ImageModel. Got {class_name}. "
            "RomanRecoverer will not inject into a bare array and will not fall back "
            "to a Gaussian stamp."
        )
    meta = getattr(model, "meta", None)
    if meta is None:
        raise DataError("ImageModel has no meta. Refusing to inject.")
    wcs = _dig(meta, "wcs")
    if wcs is None:
        raise DataError("ImageModel.meta.wcs is missing. Refusing to inject.")
    optical = _dig(meta, "instrument", "optical_element")
    detector = _dig(meta, "instrument", "detector")
    read_pattern = _dig(meta, "exposure", "read_pattern")
    start_time = _dig(meta, "exposure", "start_time")
    if not isinstance(optical, str) or not optical:
        raise DataError("ImageModel.meta.instrument.optical_element is missing.")
    if not isinstance(detector, str) or len(detector) < 2 or not detector[-2:].isdigit():
        raise DataError(
            "ImageModel.meta.instrument.detector must end in a two-digit SCA index, "
            f"got {detector!r}."
        )
    sca = int(detector[-2:])
    if not 1 <= sca <= 18:
        raise DataError(f"SCA index {sca} from detector {detector!r} is outside 1..18.")
    if read_pattern is None:
        raise DataError("ImageModel.meta.exposure.read_pattern is missing.")
    if start_time is None:
        raise DataError(
            "ImageModel.meta.exposure.start_time is missing. romanisim 0.15.0 passes it to make_psf."
        )
    data = getattr(model, "data", None)
    if data is None or getattr(data, "ndim", None) != 2:
        raise DataError("ImageModel.data must be a 2-D rate image.")
    return {
        "class_name": class_name,
        "optical_element": optical,
        "detector": detector,
        "sca": sca,
        "has_wcs": True,
        "shape": [int(data.shape[0]), int(data.shape[1])],
        "known_v0150_imaging_filter": optical in ROMANISIM_V0150_IMAGING_FILTERS,
    }


def inject_l2(
    model: Any,
    catalog: Any,
    *,
    x: Any = None,
    y: Any = None,
    psf: Any = None,
    seed: int = 50,
    rng: Any = None,
    gain: float | None = None,
    psftype: str = "epsf",
    accept_unpinned: bool = False,
) -> Any:
    """Inject into a copy of an L2 ImageModel using the installed romanisim.

    The input model is not replaced by a synthetic Gaussian image under any
    failure. Reference files are not downloaded by this function; romanisim
    itself may touch CRDS when it builds a PSF, which is why a missing
    reference-data gain is reported before the call when the pinned module
    layout is present.
    """

    probe = probe_romanisim()
    if probe["status"] == STATUS_SKIPPED or not probe.get("available"):
        raise DependencyError(
            probe.get("reason", "romanisim is not available."),
            status=probe.get("status", STATUS_SKIPPED),
        )
    if probe["status"] == "INCOMPATIBLE_API":
        raise DependencyError(probe["reason"], status="INCOMPATIBLE_API")
    if not probe.get("version_matches_pin") and not accept_unpinned:
        raise DependencyError(
            probe.get("pin_note", "romanisim version is not pinned."),
            status="UNPINNED_VERSION",
        )
    summary = validate_l2_model(model)
    validate_catalog_table(catalog, filter_name=summary["optical_element"])
    if not summary["known_v0150_imaging_filter"]:
        # The live library is authoritative for filter names. When the installed
        # version is the pin, an unknown imaging name is refused before make_psf.
        if probe.get("version_matches_pin"):
            raise DataError(
                f"optical_element {summary['optical_element']!r} is not in the romanisim "
                f"{ROMANISIM_PINNED_VERSION} imaging map {list(ROMANISIM_V0150_IMAGING_FILTERS)} "
                "and is not a documented GRISM/PRISM alias handled by this adapter."
            )
    _refuse_if_reference_gain_missing(gain)
    image = importlib.import_module("romanisim.image")
    function = image.inject_sources_into_l2
    signature = inspect.signature(function)
    arguments: dict[str, Any] = {"model": model, "cat": catalog}
    optional = {
        "x": x,
        "y": y,
        "psf": psf,
        "seed": seed,
        "rng": rng,
        "gain": gain,
        "psftype": psftype,
    }
    for name, value in optional.items():
        if value is None and name in {"x", "y", "psf", "rng", "gain"}:
            continue
        if name not in signature.parameters:
            raise DependencyError(
                f"Installed inject_sources_into_l2 has no parameter {name!r}. "
                "Refusing to drop it silently or to call a different injector.",
                status="INCOMPATIBLE_API",
            )
        arguments[name] = value
    try:
        return function(**arguments)
    except DependencyError:
        raise
    except Exception as exc:  # noqa: BLE001 - the library's own failure must stay a failure
        raise DataError(
            "romanisim.image.inject_sources_into_l2 failed. "
            "No Gaussian fallback was attempted. "
            f"Original error: {exc.__class__.__name__}: {exc}"
        ) from exc


def _refuse_if_reference_gain_missing(gain: float | None) -> None:
    if gain is not None:
        return
    reference = None
    import_error: Exception | None = None
    for module_name in ("romanisim.models.parameters", "romanisim.parameters"):
        try:
            parameters = importlib.import_module(module_name)
        except ImportError as exc:
            import_error = exc
            continue
        candidate = getattr(parameters, "reference_data", None)
        if isinstance(candidate, dict) and candidate.get("gain") is not None:
            reference = candidate
            break
    if reference is None:
        detail = f" Last import error: {import_error}." if import_error else ""
        raise DependencyError(
            "Gain was not provided and romanisim.models.parameters.reference_data['gain'] "
            "is unavailable. Refusing to invent a gain or to skip the physical injection."
            + detail,
            status="MISSING_REFERENCE_DATA",
        )


def _dig(obj: Any, *names: str) -> Any:
    current = obj
    for name in names:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(name)
        else:
            current = getattr(current, name, None)
    return current


def _column_names(catalog: Any) -> set[str]:
    if hasattr(catalog, "colnames"):
        return set(catalog.colnames)
    if hasattr(catalog, "columns"):
        return set(map(str, catalog.columns))
    raise DataError("Catalog must be an astropy Table or a table-like object with columns.")


def _column_values(catalog: Any, name: str) -> list[str]:
    values = catalog[name]
    if hasattr(values, "tolist"):
        raw = values.tolist()
    else:
        raw = list(values)
    return [str(value) for value in raw]
