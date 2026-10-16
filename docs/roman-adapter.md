# Roman I-Sim L2 adapter

Roman I-Sim is the high-fidelity WFI simulator. STIPS is in maintenance mode and is not used. The Roman User Documentation says STIPS receives only critical bug fixes and points new work at Roman I-Sim:

https://roman-docs.stsci.edu/simulation-tools/additional-simulation-tools/stips-space-telescope-imaging-product-simulator

The adapter is pinned to **romanisim 0.15.0** (PyPI release, also the version named in the Roman User Documentation page dated against the 5 August 2026 release). The call was read from the tag, not invented:

https://github.com/spacetelescope/romanisim/blob/v0.15.0/romanisim/image.py

```text
romanisim.image.inject_sources_into_l2(
    model,
    cat,
    x=None,
    y=None,
    psf=None,
    seed=50,
    rng=None,
    gain=None,
    psftype="epsf",
)
```

`model` is a `roman_datamodels.datamodels.ImageModel`. `cat` is an astropy Table. The function returns a deep copy with the new slope; the v0.15.0 source copies the model before replacing pixels.

## What v0.15.0 actually does

From that function body:

* If `x` or `y` is omitted, pixel positions come from `model.meta.wcs.numerical_inverse` on `cat['ra']` and `cat['dec']`.
* The filter is `model.meta.instrument.optical_element`.
* The table is converted with `catalog.table_to_catalog(cat, [filter_name])`. The catalog documentation requires `ra`, `dec`, `type` (`PSF` or `SER`), `n`, `half_light_radius` (arcsec), `pa`, `ba`, and one flux column per bandpass. Fluxes are maggies: AB magnitude = `-2.5 log10(flux)`.
* The SCA index is `int(model.meta.instrument.detector[-2:])`, so the detector name has to end in two digits.
* When `psf` is omitted, `romanisim.psf.make_psf` is called with `model.meta.exposure.start_time`.
* The ramp uses `model.meta.exposure.read_pattern`.
* If `gain` is omitted, v0.15.0 uses `romanisim.models.parameters.reference_data['gain']` (electron/DN). `image.py` imports `parameters` from `romanisim.models`. The source then divides resultant electrons by that gain when adding them to a ramp built from the L2 rate image.

The latest readthedocs page for the same function describes gain differently: it says gain sets the electron Poisson noise while the image photometric calibration sets the total flux. That prose and the v0.15.0 source do not say the same thing. The adapter records `inspect.signature` and `__version__` at runtime. If the installed version is not 0.15.0, it refuses to call unless `accept_unpinned=True`, and even then it only passes parameter names that exist on the live function.

## L2 shortcuts

The image documentation and the L2 documentation state the shortcuts directly:

* L2 files do not record which pixels contain cosmic-ray hits. Injected flux in those pixels is refit as if the hit were absent. No new cosmic rays are added.
* Nonlinearity and saturation are not included.
* The ramp is refit without persistence. Persistence already present in the original slope remains.

If those effects matter, the Roman I-Sim documentation says to inject at L1 instead. This package does not implement an L1 path.

## Filters checked against the v0.15.0 map

`romanisim/models/bandpass.py` at that tag maps Roman imaging names `F062`, `F087`, `F106`, `F129`, `F146`, `F158`, `F184`, and `K213` onto the GalSim names (`R062`, `Z087`, `Y106`, `J129`, `W146`, `H158`, `F184`, `K213`). `GRISM` and `PRISM` are separate aliases. The adapter refuses an optical element outside the imaging list when the installed version is exactly 0.15.0. A newer install is not second-guessed beyond the live signature, and it still will not run without `accept_unpinned`.

## Failure behaviour

| Condition | Status |
| --- | --- |
| `romanisim` cannot be imported | `SKIPPED_DEPENDENCY` |
| The function or the `model`/`cat` parameters are missing | `INCOMPATIBLE_API` |
| Installed version is not 0.15.0 and unpinning was not requested | `UNPINNED_VERSION` |
| Gain was omitted and `reference_data['gain']` cannot be read | `MISSING_REFERENCE_DATA` |
| The object is not an `ImageModel`, or WCS, filter, detector, read pattern, start time, or catalog columns are missing | a data error, before any stamp is written |
| The romanisim call raises | the original exception is wrapped and re-raised |

There is no path from a failed L2 injection to `romanrecoverer.inject`. A Gaussian stamp is a different workflow, selected only by `romanrecoverer synth`, and its products are labelled `synthetic_demonstration`.

Reference files and CRDS are not downloaded by this package. Building a PSF inside romanisim may still require them. That is a property of romanisim, and a missing gain or a failed call stops the adapter.

## What is not verified here

romanisim 0.15.0 was not installed in the environment that produced v0.1.0. The tests call `probe_romanisim()`. When the import fails they expect `SKIPPED_DEPENDENCY` and check that `inject_l2` raises that status. They do not replace the function with a mock and then call the result a Roman validation. Catalog-column checks and the `ImageModel` name check are pure and do not pretend to have run the simulator.

Installing the extra later is:

```bash
pip install -e ".[roman]"
romanrecoverer doctor --require romanisim
```

An end-to-end L2 injection still needs a real Level-2 `ImageModel`, a WCS, a filter, a read pattern, and the reference files romanisim asks for. Until that run exists, no report from this repository is a romanisim result.
