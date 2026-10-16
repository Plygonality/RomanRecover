"""Write the machine-readable products and a self-contained HTML summary."""

from __future__ import annotations

import base64
import html
from pathlib import Path
from typing import Any

import pandas as pd

from romanrecoverer.provenance import build_manifest, file_record, software_versions, utc_now, write_json
from romanrecoverer.schemas import REPORT_SCHEMA, SCHEMA_VERSION


def _fmt(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_products(
    out_dir: str | Path,
    *,
    metrics: dict[str, Any],
    matches: pd.DataFrame,
    strata: pd.DataFrame,
    plot_paths: dict[str, Path],
    inputs: list[dict[str, Any]],
    extra_outputs: list[Path] | None = None,
) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.json"
    matches_path = out / "matches.csv"
    strata_path = out / "strata.csv"
    report_path = out / "report.html"
    provenance_path = out / "provenance.json"
    manifest_path = out / "manifest.json"

    metrics = dict(metrics)
    metrics["software"] = software_versions()
    metrics["created_utc"] = utc_now()
    write_json(metrics_path, metrics)
    matches.to_csv(matches_path, index=False)
    strata.to_csv(strata_path, index=False)
    write_json(
        provenance_path,
        {
            "schema_name": "romanrecoverer.provenance",
            "schema_version": SCHEMA_VERSION,
            "created_utc": metrics["created_utc"],
            "software": metrics["software"],
            "inputs": inputs,
            "mode": metrics.get("mode"),
            "calibration_status": metrics.get("calibration_status"),
            "immutable_inputs": True,
        },
    )
    html_text = render_html(metrics, strata, plot_paths)
    report_path.write_text(html_text, encoding="utf-8")

    outputs = [metrics_path, matches_path, strata_path, report_path, provenance_path, *plot_paths.values()]
    if extra_outputs:
        outputs.extend(extra_outputs)
    output_records = [file_record(path, role=path.name) for path in outputs if path.is_file()]
    manifest = build_manifest(inputs=inputs, outputs=output_records)
    write_json(manifest_path, manifest)
    return {
        "metrics": metrics_path,
        "matches": matches_path,
        "strata": strata_path,
        "report": report_path,
        "provenance": provenance_path,
        "manifest": manifest_path,
    }


def render_html(metrics: dict[str, Any], strata: pd.DataFrame, plot_paths: dict[str, Path]) -> str:
    counts = metrics["counts"]
    precision = metrics["precision"]
    photometry = metrics["photometry"]
    position = metrics["position"]
    intervals = metrics["confidence_intervals"]
    rows = []
    for record in strata.to_dict(orient="records"):
        rows.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(_fmt(record.get(column)))}</td>"
                for column in (
                    "stratum",
                    "bin",
                    "n_eligible",
                    "n_recovered",
                    "completeness",
                    "status",
                    "wilson_lower",
                    "wilson_upper",
                )
            )
            + "</tr>"
        )
    figures = []
    for name, path in plot_paths.items():
        if not path.is_file():
            continue
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        figures.append(
            f"<figure><img alt='{html.escape(name)}' src='data:image/png;base64,{encoded}'/>"
            f"<figcaption>{html.escape(name)}</figcaption></figure>"
        )
    title = f"RomanRecoverer {metrics.get('mode', '')}"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{html.escape(title)}</title>
<style>
body {{ font-family: Georgia, serif; margin: 2rem; color: #1c1c1c; background: #fafafa; }}
h1 {{ font-weight: 500; }}
.banner {{ padding: 0.8rem 1rem; background: #fff4d6; border: 1px solid #e0c36a; }}
.caveat {{ padding: 0.8rem 1rem; background: #f3f6fb; border: 1px solid #c5d0e0; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.92rem; }}
th, td {{ border-bottom: 1px solid #ddd; text-align: left; padding: 0.3rem 0.4rem; }}
figure {{ margin: 1rem 0; }}
img {{ max-width: 100%; height: auto; background: white; }}
code {{ font-family: ui-monospace, monospace; }}
</style>
</head>
<body>
<h1>RomanRecoverer evaluation</h1>
<p class="banner"><strong>Mode:</strong> {html.escape(str(metrics.get("mode")))}. {html.escape(str(metrics.get("mode_note")))}</p>
<p class="caveat">{html.escape(str(metrics.get("false_positive_policy")))} {html.escape(str(metrics.get("anomaly_note")))}</p>
<h2>Counts</h2>
<ul>
<li>Injected: {counts["n_injected"]}</li>
<li>Eligible: {counts["n_eligible"]}</li>
<li>Recovered: {counts["n_recovered"]}</li>
<li>Unmatched eligible truth: {counts["n_unmatched_truth"]}</li>
<li>Unmatched detections: {counts["n_unmatched_detected"]}</li>
<li>Ineligible: {counts["n_ineligible"]}</li>
<li>Ambiguous truths: {counts["n_ambiguous_truth"]}</li>
</ul>
<h2>Completeness</h2>
<p>Value: <strong>{html.escape(_fmt(metrics.get("completeness")))}</strong> ({html.escape(str(metrics.get("completeness_status")))}).</p>
<p>Wilson 95%: {_fmt(intervals["wilson"]["lower"])} – {_fmt(intervals["wilson"]["upper"])}.
Clopper–Pearson 95%: {_fmt(intervals["clopper_pearson"]["lower"])} – {_fmt(intervals["clopper_pearson"]["upper"])}.
Bootstrap: {_fmt(intervals["bootstrap"]["lower"])} – {_fmt(intervals["bootstrap"]["upper"])}.</p>
<p>{html.escape(intervals["assumptions"])}</p>
<h2>Precision</h2>
<p>Status: {html.escape(str(precision.get("status")))}. Value: {html.escape(_fmt(precision.get("value")))}.
False-positive count: {html.escape(_fmt(precision.get("false_positive_count")))}.
Unmatched detections labelled as false positives: {html.escape(str(precision.get("labelled_unmatched_as_false_positive")))}.</p>
<p>{html.escape(str(precision.get("reason") or precision.get("assumption")))}</p>
<h2>Photometry</h2>
<p>Status: {html.escape(str(photometry.get("status")))}. {html.escape(str(photometry.get("reason") or photometry.get("note")))}</p>
<h2>Positions</h2>
<p>Status: {html.escape(str(position.get("status")))}.
Median dx: {_fmt(position.get("median_dx"))} px.
Median dy: {_fmt(position.get("median_dy"))} px.
RMS separation: {_fmt(position.get("rms_separation"))} px.</p>
<h2>Strata</h2>
<table>
<thead><tr><th>Stratum</th><th>Bin</th><th>Eligible</th><th>Recovered</th><th>Completeness</th><th>Status</th><th>Wilson low</th><th>Wilson high</th></tr></thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
<h2>Figures</h2>
{''.join(figures)}
<p>Schema {html.escape(REPORT_SCHEMA)} {html.escape(SCHEMA_VERSION)}. Calibration status: {html.escape(str(metrics.get("calibration_status")))}. Coordinate frame: {html.escape(str(metrics.get("coordinate_frame")))}, origin {html.escape(str(metrics.get("pixel_origin")))}, {html.escape(str(metrics.get("axis_order")))}.</p>
<p>{html.escape(str(metrics.get("threshold_policy")))}</p>
</body>
</html>
"""
