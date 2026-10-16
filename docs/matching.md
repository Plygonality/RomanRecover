# Matching

Matching is a gated one-to-one assignment.

1. Coordinates are taken from the resolved columns. For an NGRST-style detection table those are `x_centroid` and `y_centroid`. For an NGRST-style truth table they are `x` and `y`. `x` is the column and `y` is the row.
2. Origin 1 catalogues are converted to origin 0 before the edge cut. The match itself uses the numbers in the catalogue after that conversion, so a FITS-pixel catalogue and a 0-based catalogue are not silently mixed.
3. Ineligible injections are removed from the assignment. A masked or edge source cannot take a detection away from an eligible one.
4. Pairwise Euclidean distances are computed with `scipy.spatial.distance.cdist`.
5. Pairs farther than the radius receive a cost larger than any in-radius pair. `scipy.optimize.linear_sum_assignment` then solves the rectangular assignment, which is the Hungarian algorithm. Each truth is paired with at most one detection, and each detection with at most one truth.
6. Assigned pairs whose true distance is still outside the radius are dropped. A distance equal to the radius is kept.
7. Equal distances are perturbed by `1e-9 * (truth_index * n_detections + detection_index) / (n_truth * n_detections)`. The perturbation is far below a pixel, so it does not pull a pair across the radius. It breaks ties toward the earlier detection row, then the earlier eligible truth row.

## Flags

* **Ambiguous.** An eligible truth has more than one detection inside the radius, before the assignment. The recovery count still increases by at most one.
* **Blended.** A detection has more than one eligible truth inside the radius. The match row for the truth that received that detection is marked blended. The other truths are unmatched if no detection remains.

One detection piled under several truth sources recovers one of them, not all of them. Several detections piled on one truth recover that truth once.

## Counts

| Count | Definition |
| --- | --- |
| `N_injected` | Truth rows. |
| `N_eligible` | Unmasked and, when the image shape is known, inside the center cut. |
| `N_recovered` | Eligible truths with a unique in-radius detection. |
| `N_unmatched_truth` | Eligible truths without one. |
| `N_unmatched_detected` | Detections that recovered nobody. |
| `N_ineligible` | Injected rows left out of the denominator. |

`N_recovered + N_unmatched_truth = N_eligible`.

`N_unmatched_detected` is not a false-positive count. See `docs/scientific-assumptions.md`.

## Outputs

`matches.csv` has one row per truth and one extra row per unmatched detection (`row_kind` is `truth` or `unmatched_detection`). Separations, `dx`, `dy`, the eligibility reason, and the ambiguity flags are on those rows.

The algorithm does not use `anomaly_score`.
