# Fe heat-capacity fitting

From the repository root, with the package and ESPEI installed:

```sh
python libreCalphad/databases/LC-steels_validation/FE/fit_cpm.py \
  --datasets /path/to/ESPEI-datasets/datasets \
  --phase BCC_A2 \
  --output-dir /path/to/fit-results
```

Choose `BCC_A2`, `FCC_A1`, `LIQUID`, or `GAS`. The command reads the phase settings from `FE-params.json` (override with `--params`) and selects matching Fe CPM datasets. It writes a fitted-parameters JSON file and a comparison plot to the output directory. Review the fit before using its parameters in the ESPEI workflow; the command does not update the database or reference states.

## Combined BCC assessment script

`FE-fit-bcc.py` is a separate, combined workflow. Its `FE-params.json` input,
`_fitted_params.json` output, `_fit_diagnostics.json` output, and plots resolve
relative to the script directory, regardless of the working directory. The
dataset directory still comes from `system.datasets` in `run_param_gen.yaml`.

The script updates `FE.BCC_A2` in `_fitted_params.json`, preserving other entries.
It separately updates `FE.BCC_A2` in `_fit_diagnostics.json` with optimizer
success/status/message, iteration and evaluation counts, the optimizer parameter
vector, the dimensionless relative-residual objective, and dataset references.
Both files are written atomically **before plotting**, so a plotting failure does
not lose the fit or its diagnostics. Fits that exhaust the iteration limit are
still saved, and the existing downstream assessment behavior is unchanged.

After saving and plotting, this script continues to update external reference-state
files, regenerate databases, calculate properties, and adjust multi-phase offsets
in `FE-params.json`. It is **not** a fit-only command. The saved Cp fit is a snapshot
before those subsequent offset adjustments, not a final offset-calibrated assessment.

TODOs:
- Separate Cp fitting/persistence from the downstream assessment operations.
- Define acceptance criteria and an approval mechanism before promoting fitted
  results to assessment inputs or databases. Diagnostics currently record solver
  status; they do not constitute approval.

## Models and fitting conventions

To use numerical Debye fitting, supply a separate `--params` file with the phase's lattice entry replaced by `"debye": {"theta": [250, [200, 400], "fit"]}`. Use `[300, "fix"]` to hold theta fixed, or omit the bounds to use `(1e-6, 2500)` K. Select only one of `einstein`, `holzapfel`, and `debye`; theta and any custom lower bound must be positive. Debye fits can be plotted and saved as fitted-parameter JSON, but Gibbs reference-state generation and database parameter export are not implemented and explicitly reject these models. A Debye theta must not be substituted directly for pycalphad's Einstein theta.

Exact Debye Gibbs energy, enthalpy, and entropy are available numerically and as SymPy expressions through `libreCalphad.models.energy._debye_gibbs`, `_debye_enthalpy`, and `_debye_entropy`. These include zero-point energy `9R*theta/8` and use `S(0)=0`; enthalpy/entropy reference offsets must be added separately. Their integral expressions cannot be compiled by the current pycalphad/SymEngine backend, so these helpers do not enable database export. See the [energy expressions and usage](../../../../README.md#debye-gibbs-energy-enthalpy-and-entropy).

The fitting objective is the norm of the relative Cp residuals. An observed Cp of exactly zero is an exact constraint: a matching prediction contributes zero, while a nonzero prediction receives an infinite objective. Nonfinite observed or predicted values also receive an infinite objective rather than being ignored. No epsilon or weighting floor is added, so the relative residuals for nonzero observations are unchanged. Experimental values rounded to zero should not be treated as exact zero-Cp constraints.

For a phase with a `melt` model, the selected lattice (Einstein, Holzapfel, or numerical Debye) and magnetic contributions remain active above `T_melt`; the other solid contributions are replaced by the melt correction. When `liquid_Cp` is specified, the fit fixes the melt expression's `a` parameter to `liquid_Cp - 3R` when a lattice contribution is present, so the total heat capacity approaches `liquid_Cp`.

The generated custom Gibbs reference state retains pycalphad's Einstein and Xiong solid contributions above `T_melt` and matches Gibbs energy and entropy at the boundary.

The Xiong magnetic heat capacity follows `Cp = -T * d²G/dT²` for the Gibbs energy in equations (12)–(14) of Xiong et al. (2012), doi:10.1016/j.calphad.2012.07.002. Earlier versions included an extra `T/Tcrit` multiplier in the magnetic heat capacity. Fits made with that expression should be repeated against the experimental CPM datasets and reviewed before updating database parameters; existing fitted parameters and databases are not automatically changed.
