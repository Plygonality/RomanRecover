# Design

RomanRecoverer is a catalogue benchmark with a synthetic scene generator and a separate Roman I-Sim adapter.

```
src/romanrecoverer/
  catalogs.py            column contracts, including NGRST names
  eligibility.py         mask and edge cut
  match.py               gated Hungarian assignment
  metrics.py             completeness, intervals, precision, photometry
  stratify.py            bins and empty-bin handling
  wcsutil.py             Astropy high-level WCS, origin and axis order
  inject.py              dimensionless stamps only
  image_io.py            small npz/FITS bundles, not survey products
  evaluate.py            catalogue workflow
  plots.py / reports.py  figures, HTML, metrics JSON, manifest
  provenance.py          hashes and versions
  detector_adapters/
    ngrst.py             column contract and verified ngrst-discover command
    subprocess_runner.py explicit command templates, no shell
    romanisim_l2.py      pinned L2 call, or SKIPPED_DEPENDENCY
  cli.py
```

Workflow A (`evaluate`) imports none of romanisim and none of `ngrst`. Workflow B (`synth`) writes a truth CSV and an `.npz` image. Workflow C (`inject_l2`) imports romanisim only inside the call, and only after the version and signature checks.

Inputs are hashed. The tools write into `--out` and do not rewrite the caller's files. A detector subprocess is allowed to touch the path it was given; the input hash is compared afterwards and a change is recorded.

Large FITS, ASDF, and NPZ observation products are gitignored. The committed catalogues under `examples/` are small hand-built demonstration tables, not survey extracts and not toolkit outputs.

Exit codes: 0 success, 2 invalid data, 3 detector process failed, 4 missing science dependency.
