# Scientific assumptions

RomanRecoverer measures whether a detector catalogue recovered a controlled list of injected sources. It does not decide that an unmatched or high-`anomaly_score` source is an astrophysical discovery. It does not replace [Roman I-Sim](https://romanisim.readthedocs.io/en/latest/romanisim/image.html), and it does not contain a source detector.

Three result modes are kept apart in every metrics file and HTML report:

| Mode | Meaning |
| --- | --- |
| `catalogue_only` | Truth and detection tables were matched. No image was simulated in that step. |
| `synthetic_demonstration` | Sources were stamped into a Numpy array with a declared profile and noise model. |
| `romanisim_l2` | Sources were injected with `romanisim.image.inject_sources_into_l2` on a real `ImageModel`. |
| `real_data` | The caller labelled the inputs as real data. The label is not checked against a Roman archive. |

A dimensionless synthetic run is never described as instrument-calibrated science.

## What the NGRST columns are

The adapter follows the column contract given for the current `ngrst` package:

* truth: `x`, `y`, `amplitude`, `sigma`, `class`
* detections: `x_centroid`, `y_centroid`, `snr`, `anomaly_score`, plus any further columns, which are kept out of the match

`amplitude` is a Gaussian peak in that toolkit's synthetic image. `sigma` is a Gaussian width in pixels. Neither is an AB magnitude, a maggie, or a Roman DN/s flux. `snr` is a detector signal-to-noise. `anomaly_score` is a detector statistic. RomanRecoverer copies it and does not turn it into a discovery.

Photometric bias is `NOT_APPLICABLE` for that pair of tables. If both catalogues also carry an `amplitude` column in the same uncalibrated unit, a relative peak ratio may be reported with status `RELATIVE_AMPLITUDE_ONLY`. That ratio is not a flux bias.

## Synthetic image model

The default generator writes `calibration_status = dimensionless_synthetic`.

* A Gaussian source has central value `peak_amplitude` and standard deviation `sigma` in pixels. Its analytic integral is `peak_amplitude * 2 * pi * sigma**2`, stored as `integrated_proxy`. The integral is in image units times square pixels. It is not a calibrated flux.
* An exponential source has central value `peak_amplitude` and scale length `h` pixels. Its analytic integral is `2 * pi * peak_amplitude * h**2`. On the truth table the `sigma` column holds `h` for that profile, and `profile` says `exponential`.
* Noise is Gaussian with the declared `noise_sigma`, unless `noise_sigma` is 0. Poisson draws are refused unless `image_unit` is `electron`.
* `gain` and `exposure_s` are stored when the configuration supplies them. They are not converted into magnitudes.
* Passing an AB magnitude, a zeropoint, or an image unit of `ab_mag` / `jansky` raises an error.

`snr` on a synthetic truth row is `peak_amplitude / noise_sigma` when the noise is positive. It is a proxy, not a detection SNR and not a Roman photometric SNR.

Seeds are split with `numpy.random.SeedSequence.spawn` into position, amplitude, and noise streams. The same seed repeats source ids and placements. A different seed changes the placements.

## Eligibility

An injection is eligible when:

1. its coordinates are finite
2. it is not masked (`mask` in `{0, false, no, unmasked}`)
3. if an image shape `(ny, nx)` is known, its center lies in the closed interval `[margin, n - 1 - margin]` on both axes after conversion to 0-based pixels

`x` is the column and `y` is the row. This is a cut on the center, not on the wings of the PSF. Without an image shape the edge cut is not applied, and the metrics say `edge_exclusion_applied: false`.

Ineligible injections stay in the match table. They do not enter the completeness denominator and they do not consume a detection.

Masked and edge cases can coincide (`masked_and_edge_or_outside`).

Zero peak amplitude is rejected. `allow_zero` records the row as ineligible instead. Negative and non-finite peaks, and non-positive widths, always raise.

## Completeness

```
completeness = N_recovered / N_eligible
```

`N_recovered` counts eligible injections that received a unique detection inside the match radius. `N_injected = N_eligible + N_ineligible`. `N_unmatched_truth` counts eligible injections without a match.

If `N_eligible` is 0 the ratio is undefined (`UNDEFINED_ZERO_DENOMINATOR`), not zero.

### Intervals

Eligible injections are treated as independent Bernoulli trials. Blending, correlated noise, and a shared detection threshold violate that, so the intervals are binomial uncertainties, not a survey covariance.

The interval to quote is the Wilson score interval at 95 percent, using the normal quantile `z = normsinv(0.975)`. Clopper–Pearson equal-tailed beta intervals are reported beside it. A percentile bootstrap of the observed Bernoulli outcomes is a third number, drawn with a seed that is not the injection seed. The bootstrap collapses when every trial has the same outcome; Wilson and Clopper–Pearson do not. Empty bins stay empty: counts are zero, completeness is null, status is `EMPTY_BIN`.

## Positions

Matching is Euclidean in the pixel frame. `dx = x_detection - x_truth` and `dy = y_detection - y_truth`, with `x` the column and `y` the row. Sky catalogues are not matched as raw degrees. Convert them with the image WCS first (`romanrecoverer.wcsutil`), using Astropy's high-level API, which is 0-based. FITS `CRPIX` is 1-based; pass `pixel_origin=1` for catalogues in that convention.

The synthetic generator can attach a demonstration TAN WCS with a plate scale of 0.11 arcsec/pixel. That scale is only for a round trip. It is not a Roman GWCS and it has no distortion.

## Precision and false positives

An unmatched detection is not a false positive.

* `unspecified` or `populated_background`: precision is `NOT_APPLICABLE`. Real sources that were never injected are a normal feature of a populated image.
* `truth_complete_synthetic`: the scene is *assumed* to contain only the injections and the declared noise. Unmatched detections are then counted as false positives under that assumption. The report says so. The assumption is not a proof.
* `--baseline`: detections that match the baseline catalogue, and not an injection, are pre-existing. Only detections that match neither are candidate false positives. The baseline match is one-to-one at the same radius unless `--baseline-radius-px` is set.

Precision is undefined when there are no recovered sources and no candidate false positives, including the empty-detection case.

## Photometry

Bias is computed only when `calibration_status` is `calibrated` and both sides have the same quantity, either magnitude or flux, in the same declared unit. Magnitude residual is measured minus injected. Flux residual is `(measured - injected) / injected`, and zero injected flux is dropped.

Magnitude bins are added only in that calibrated case. Peak amplitude is never placed on a magnitude axis.

## Stratification

Completeness is split by:

* amplitude, as a brightness proxy, with the unit the table declares
* `sigma`, as a size proxy
* crowding: the number of *other eligible* injections within `--crowd-radius-px` (default 5)
* `local_background` when that truth column exists
* truth `snr` when that column exists

Detected SNR is not a completeness axis. It does not exist for unrecovered injections, so binning on it would make the denominator depend on the outcome.

The synthetic generator fills `local_background` with the median of a 4-to-8 pixel annulus on the stamped image. That is a Numpy median, not a `photutils` background model and not a Roman sky subtraction. `photutils` is an optional extra and is not required for the catalogue benchmark.

## Thresholds and comparisons

RomanRecoverer does not choose a detection threshold. `romanrecoverer compare` reports the completeness difference and whether the recorded seeds match. A higher completeness on the same sample is not an unbiased reason to prefer that configuration. Tuning a threshold on the evaluation sample and then quoting that sample is descriptive only. The metrics file says this.

## Invalid data

Non-finite coordinates raise. Duplicate identifiers raise. A non-positive match radius raises. An unknown dataset assumption raises. The synthetic generator raises on a non-positive image shape, a non-positive PSF width, and a forbidden magnitude unit.
