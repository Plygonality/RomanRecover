# RomanRecoverer

Injection-and-recovery benchmarks for Nancy Grace Roman Space Telescope Wide Field Instrument source catalogues.

Given injected sources and a detector catalogue, RomanRecoverer reports which injections were recovered, the positional offsets, and the completeness of that recovery. It does not claim that anomalous sources are discoveries, it does not replace [Roman I-Sim](https://romanisim.readthedocs.io/en/latest/romanisim/image.html), and it does not implement a source detector.

The GitHub repository that holds this code is [Plygonality/RomanRecover](https://github.com/Plygonality/RomanRecover). A separate `Plygonality/RomanRecoverer` repository was not created: the authenticated token cannot create repositories (`403`), and the private `NGRST-Discovery-Toolkit` repository was not readable (`404`).

## Five-minute demo

The catalogue benchmark and the synthetic scene need no Roman installation and no network once the dependencies are installed.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

romanrecoverer doctor

romanrecoverer evaluate \
  --truth examples/points_truth.csv \
  --detections examples/points_catalog.csv \
  --radius-px 2.0 \
  --out reports/points

romanrecoverer synth \
  --config examples/controlled_scene.toml \
  --out build/synthetic
```

`reports/points/metrics.json` is the machine-readable result. `reports/points/report.html` is the summary. On the committed demonstration tables the result is:

* 8 injected, 8 eligible, 7 recovered, 1 unmatched injection (`t08`), 2 unmatched detections (`d08`, `d09`)
* completeness `7/8 = 0.875`
* Wilson 95% interval `0.529` to `0.978`; Clopper–Pearson 95% `0.473` to `0.997`; bootstrap percentile interval `0.625` to `1.0` (2000 resamples, seed 12345). The bootstrap collapses toward the high side because 7 of 8 trials succeeded. Quote the Wilson interval.
* precision `NOT_APPLICABLE` — the two unmatched detections are not called false positives
* photometry `NOT_APPLICABLE` — `amplitude` is a dimensionless Gaussian peak, and `snr` is not that quantity
* median residual `dx = 0.20` px, `dy = 0.10` px; RMS separation `0.340` px

The same run is stored in [examples/outputs/points](examples/outputs/points): `metrics.json`, `matches.csv`, `strata.csv`, `report.html`, and the four figures. Those tables are a software demonstration, not a Roman observation and not an NGRST-Discovery-Toolkit product.

`build/synthetic/` contains `image.npz`, `truth.csv`, and a metadata sidecar. The image is a dimensionless demonstration. Peak amplitude is not an AB magnitude.

To push that synthetic scene through an external program:

```bash
romanrecoverer run \
  --image build/synthetic/image.npz \
  --truth build/synthetic/truth.csv \
  --detector command \
  --detector-cmd "python examples/detectors/offset_catalogue.py --truth {truth} --output {detections} --min-amplitude 0" \
  --radius-px 2.0 \
  --out reports/pipeline
```

`examples/detectors/offset_catalogue.py` is a test double, not a detector and not `ngrstd-discover`. The report records the exact command, the exit status, and stdout/stderr.

The NGRST integration command is:

```bash
romanrecoverer run \
  --image build/synthetic/image.npz \
  --truth build/synthetic/truth.csv \
  --detector ngrst-discover \
  --out reports/pipeline
```

Without `ngrstd-discover` on `PATH` this exits `4` and does not invent a detector. If the program is installed, RomanRecoverer reads its `--help` and builds a command only when that text contains an image flag and an output flag. Otherwise pass `--detector-cmd` yourself. There is no hidden download.

```bash
romanrecoverer doctor --require romanisim   # exit 4 until romanisim 0.15.0 is installed
```

## Install

Python 3.11 or 3.12.

```bash
pip install -e ".[dev]"          # catalogue benchmark, synthetic scenes, tests
pip install -e ".[science]"      # optional photutils and pyarrow
pip install -e ".[roman]"        # optional romanisim==0.15.0 and roman_datamodels
```

Core libraries are NumPy, SciPy, Astropy, pandas, and Matplotlib. The catalogue workflow does not import romanisim.

## Commands

| Command | Role |
| --- | --- |
| `evaluate` | Match an existing truth CSV to a detection CSV. |
| `synth` | Write a seeded synthetic image and truth catalogue. |
| `run` | Call an external detector, then evaluate. |
| `doctor` | Show versions and skipped adapters. |
| `compare` | Difference of two metrics files. Does not pick a threshold. |

Exit codes: `0` ok, `2` invalid inputs, `3` the detector process failed, `4` a requested science dependency is missing.

## Modes

Every report states one of `catalogue_only`, `synthetic_demonstration`, `romanisim_l2`, or caller-supplied `real_data`. Synthetic output is labelled `dimensionless_synthetic` unless a real calibration was supplied, and this package does not invent one. Roman I-Sim L2 injection is documented in [docs/roman-adapter.md](docs/roman-adapter.md), including the cosmic-ray, nonlinearity, and persistence shortcuts in romanisim 0.15.0. If that library is absent the adapter status is `SKIPPED_DEPENDENCY`. It never falls back to a Gaussian stamp.

## Documentation

* [docs/scientific-assumptions.md](docs/scientific-assumptions.md) — eligibility, intervals, false positives, photometry
* [docs/matching.md](docs/matching.md) — one-to-one assignment and tie breaks
* [docs/roman-adapter.md](docs/roman-adapter.md) — pinned `inject_sources_into_l2` signature and caveats
* [docs/design.md](docs/design.md) — module map
* [CHANGELOG.md](CHANGELOG.md)

## Tests

```bash
pytest
```

The NGRST `--scene points` integration test is skipped until `ngrstd-discover` can be imported or executed. The Roman tests skip the library call in the same way. Neither skip is a mock of the instrument.

## Licence

MIT. See [LICENSE](LICENSE).
