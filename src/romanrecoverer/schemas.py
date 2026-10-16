"""Versioned schema names shared by tables, metrics, and manifests."""

from __future__ import annotations

SCHEMA_VERSION = "1.0.0"

TRUTH_SCHEMA = "romanrecoverer.truth"
DETECTION_SCHEMA = "romanrecoverer.detections"
METRICS_SCHEMA = "romanrecoverer.metrics"
MATCH_SCHEMA = "romanrecoverer.matches"
STRATA_SCHEMA = "romanrecoverer.strata"
MANIFEST_SCHEMA = "romanrecoverer.manifest"
SYNTH_SCHEMA = "romanrecoverer.synthetic_scene"
TABLE_META_SCHEMA = "romanrecoverer.table_meta"
REPORT_SCHEMA = "romanrecoverer.report"

# Pixel coordinates are (x, y) = (column, row). Numpy images are indexed
# image[row, column] = image[y, x].
AXIS_ORDER = "x_is_column_y_is_row"
PIXEL_ORIGIN_DEFAULT = 0

CALIBRATION_DIMENSIONLESS = "dimensionless_synthetic"
CALIBRATION_UNCALIBRATED_AMPLITUDE = "uncalibrated_amplitude"
CALIBRATION_CALIBRATED = "calibrated"
CALIBRATION_UNKNOWN = "unknown"

MODE_CATALOGUE = "catalogue_only"
MODE_SYNTHETIC = "synthetic_demonstration"
MODE_ROMANISIM = "romanisim_l2"
MODE_REAL = "real_data"

ASSUMPTION_UNSPECIFIED = "unspecified"
ASSUMPTION_TRUTH_COMPLETE = "truth_complete_synthetic"
ASSUMPTION_POPULATED = "populated_background"

STATUS_DEFINED = "DEFINED"
STATUS_NOT_APPLICABLE = "NOT_APPLICABLE"
STATUS_UNDEFINED = "UNDEFINED_ZERO_DENOMINATOR"
STATUS_EMPTY_BIN = "EMPTY_BIN"
STATUS_SKIPPED = "SKIPPED_DEPENDENCY"
STATUS_RELATIVE_AMPLITUDE = "RELATIVE_AMPLITUDE_ONLY"

# romanisim 0.15.0 romanisim/image.py parameter list, read from the tag.
# https://github.com/spacetelescope/romanisim/blob/v0.15.0/romanisim/image.py
ROMANISIM_PINNED_VERSION = "0.15.0"
ROMANISIM_INJECT_PARAMS = (
    "model",
    "cat",
    "x",
    "y",
    "psf",
    "seed",
    "rng",
    "gain",
    "psftype",
)

# Keys of roman2galsim_bandpass built in romanisim 0.15.0
# romanisim/models/bandpass.py. Imaging names only; GRISM/PRISM are separate.
ROMANISIM_V0150_IMAGING_FILTERS = (
    "F062",
    "F087",
    "F106",
    "F129",
    "F146",
    "F158",
    "F184",
    "K213",
)
