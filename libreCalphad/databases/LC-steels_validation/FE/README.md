# Fe heat-capacity fitting

From the repository root, with the package and ESPEI installed:

```sh
python libreCalphad/databases/LC-steels_validation/FE/fit_cpm.py \
  --datasets /path/to/ESPEI-datasets/datasets \
  --phase BCC_A2 \
  --output-dir /path/to/fit-results
```

Choose `BCC_A2`, `FCC_A1`, `LIQUID`, or `GAS`. The command reads the phase settings from `FE-params.json` (override with `--params`) and selects matching Fe CPM datasets. It writes a fitted-parameters JSON file and a comparison plot to the output directory. Review the fit before using its parameters in the ESPEI workflow; the command does not update the database or reference states.

For a phase with a `melt` model, the fit uses the solid heat capacity through `T_melt` and the melt expression above it. When `liquid_Cp` is specified, it fixes the melt expression's `a` parameter (its high-temperature limit), rather than fitting `a` separately.

The generated custom Gibbs reference state cancels pycalphad's Einstein and Xiong solid contributions above `T_melt` and matches Gibbs energy and entropy at the boundary. The Xiong cancellation assumes its fitted structure factor matches the phase model and its critical temperature is below `T_melt`.
