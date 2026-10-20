#!/usr/bin/env python3
"""Offline test double used to exercise ``romanrecoverer run``.

This is not a source detector, not an isolation forest, and not
``ngrst-discover``. It copies bright truth positions into a detection CSV
with a fixed 0.05 pixel offset and drops sources fainter than the given peak
threshold. Reports that use it must keep the detector name ``command``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-amplitude", type=float, default=2.0)
    parser.add_argument("--offset", type=float, default=0.05)
    args = parser.parse_args()
    truth = pd.read_csv(args.truth)
    kept = truth[truth["amplitude"] >= args.min_amplitude]
    detections = pd.DataFrame(
        {
            "detection_id": [f"double-{source_id}" for source_id in kept["source_id"]],
            "x_centroid": kept["x"].to_numpy(dtype=float) + args.offset,
            "y_centroid": kept["y"].to_numpy(dtype=float) + args.offset,
            "snr": kept["amplitude"].to_numpy(dtype=float),
            "anomaly_score": 0.0,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    detections.to_csv(args.output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
