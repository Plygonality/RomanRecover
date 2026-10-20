# Changelog

## Unreleased

* `ngrst-discover` is called with a positional input, `--processed-dir`, and `--candidate-dir`. The detection catalogue is `<stem>_catalog.csv` in the processed directory, beside `<stem>_run.json`.

## 0.1.0 — 2026-10-16

* Catalogue-only matching with a gated one-to-one assignment, Wilson and Clopper–Pearson intervals, and a percentile bootstrap.
* NGRST column adapter for `x,y,amplitude,sigma,class` and `x_centroid,y_centroid,snr,anomaly_score`. Amplitude is not treated as a flux or an AB magnitude. `anomaly_score` is not treated as a discovery.
* Unmatched detections are not false positives unless the dataset assumption is truth-complete synthetic or a baseline catalogue is subtracted.
* Seeded dimensionless synthetic injection of Gaussian and exponential profiles, with an input-free preset.
* `romanrecoverer` commands: `evaluate`, `synth`, `run`, `doctor`, `compare`.
* Roman I-Sim 0.15.0 L2 adapter. It calls `inject_sources_into_l2` only when that version is installed and the signature matches the tagged source. Otherwise the status is `SKIPPED_DEPENDENCY`. No Gaussian fallback.
* `ngrst-discover --scene points` was not executed for this version.
