"""Diagnostic figures. Labels follow the metric status, including non-claims."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from romanrecoverer.schemas import STATUS_NOT_APPLICABLE


def _save(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def completeness_vs_quantity(strata: pd.DataFrame, stratum: str, path: Path, *, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    subset = strata[strata["stratum"] == stratum].copy()
    if subset.empty:
        ax.set_title(f"{stratum}: not available")
        ax.text(0.5, 0.5, "No bins were defined for this stratum.", ha="center", va="center")
        ax.set_axis_off()
        _save(fig, path)
        return
    x = np.arange(len(subset))
    completeness = subset["completeness"].to_numpy(dtype=float)
    lower = subset["wilson_lower"].to_numpy(dtype=float)
    upper = subset["wilson_upper"].to_numpy(dtype=float)
    yerr_low = np.where(
        np.isfinite(completeness) & np.isfinite(lower),
        np.maximum(0.0, completeness - lower),
        np.nan,
    )
    yerr_high = np.where(
        np.isfinite(completeness) & np.isfinite(upper),
        np.maximum(0.0, upper - completeness),
        np.nan,
    )
    yerr = np.vstack([yerr_low, yerr_high])
    ax.bar(x, np.nan_to_num(completeness, nan=0.0), color="#4C78A8", alpha=0.85)
    finite = np.isfinite(yerr).all(axis=0)
    if np.any(finite):
        ax.errorbar(
            x[finite],
            completeness[finite],
            yerr=yerr[:, finite],
            fmt="none",
            ecolor="black",
            capsize=3,
        )
    for index, row in enumerate(subset.itertuples(index=False)):
        label = f"{int(row.n_recovered)}/{int(row.n_eligible)}"
        if row.status != "DEFINED":
            label = f"{label}\n{row.status}"
        ax.text(index, 0.02, label, ha="center", va="bottom", fontsize=8, color="white")
    ax.set_xticks(x, list(subset["bin"]), rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(ylabel)
    ax.set_title(stratum.replace("_", " "))
    ax.axhline(1.0, color="0.7", lw=0.6)
    fig.tight_layout()
    _save(fig, path)


def position_residuals(matches: pd.DataFrame, path: Path, *, radius: float) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    recovered = matches[(matches["row_kind"] == "truth") & (matches["recovered"] == True)]  # noqa: E712
    ax.axhline(0, color="0.7", lw=0.6)
    ax.axvline(0, color="0.7", lw=0.6)
    circle = plt.Circle((0, 0), radius, fill=False, ls="--", color="0.4", label=f"radius {radius:g} px")
    ax.add_patch(circle)
    if recovered.empty:
        ax.set_title("No recovered sources")
    else:
        ax.scatter(recovered["dx"], recovered["dy"], s=28, c="#F58518", zorder=3)
        ax.set_title("Position residuals (detection minus truth)")
    ax.set_xlabel("dx (pixel, column)")
    ax.set_ylabel("dy (pixel, row)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    _save(fig, path)


def association_overlay(
    matches: pd.DataFrame,
    path: Path,
    *,
    image: np.ndarray | None = None,
    precision_status: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    if image is not None:
        finite = image[np.isfinite(image)]
        vmin, vmax = np.percentile(finite, [5, 99]) if finite.size else (None, None)
        ax.imshow(image, origin="lower", cmap="gray", vmin=vmin, vmax=vmax, zorder=0)
    truths = matches[matches["row_kind"] == "truth"]
    recovered = truths[truths["recovered"] == True]  # noqa: E712
    missed = truths[(truths["eligible"] == True) & (truths["recovered"] == False)]  # noqa: E712
    ineligible = truths[truths["eligible"] == False]  # noqa: E712
    unmatched = matches[matches["row_kind"] == "unmatched_detection"]
    ax.scatter(truths["truth_x"], truths["truth_y"], s=36, facecolors="none", edgecolors="#4C78A8", label="injected")
    if not recovered.empty:
        ax.plot(recovered["detection_x"], recovered["detection_y"], "+", color="#54A24B", label="recovered detection")
        for row in recovered.itertuples(index=False):
            ax.plot([row.truth_x, row.detection_x], [row.truth_y, row.detection_y], color="#54A24B", lw=0.7)
    if not missed.empty:
        ax.scatter(missed["truth_x"], missed["truth_y"], s=18, c="#E45756", label="unrecovered eligible")
    if not ineligible.empty:
        ax.scatter(ineligible["truth_x"], ineligible["truth_y"], s=18, c="0.5", marker="x", label="ineligible")
    if not unmatched.empty:
        label = "unmatched detection"
        if precision_status == STATUS_NOT_APPLICABLE:
            label = "unmatched detection (not a false positive)"
        ax.scatter(unmatched["detection_x"], unmatched["detection_y"], s=28, c="#F58518", marker="^", label=label)
    ax.set_xlabel("x (column)")
    ax.set_ylabel("y (row)")
    ax.set_title("Source associations")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    _save(fig, path)


def write_all_plots(
    matches: pd.DataFrame,
    strata: pd.DataFrame,
    out_dir: Path,
    *,
    radius: float,
    image: np.ndarray | None,
    precision_status: str,
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    paths = {
        "completeness_brightness": out_dir / "completeness_brightness.png",
        "completeness_crowding": out_dir / "completeness_crowding.png",
        "position_residuals": out_dir / "position_residuals.png",
        "associations": out_dir / "associations.png",
    }
    completeness_vs_quantity(
        strata,
        "brightness_proxy_amplitude",
        paths["completeness_brightness"],
        ylabel="completeness (Wilson 95% bars)",
    )
    completeness_vs_quantity(
        strata,
        "crowding",
        paths["completeness_crowding"],
        ylabel="completeness (Wilson 95% bars)",
    )
    position_residuals(matches, paths["position_residuals"], radius=radius)
    association_overlay(matches, paths["associations"], image=image, precision_status=precision_status)
    return paths
