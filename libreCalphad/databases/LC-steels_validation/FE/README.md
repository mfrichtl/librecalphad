# Fe heat-capacity fitting

From the repository root, with the package and ESPEI installed:

```sh
python libreCalphad/databases/LC-steels_validation/FE/fit_cpm.py \
  --datasets /path/to/ESPEI-datasets/datasets \
  --phase BCC_A2 \
  --output-dir /path/to/fit-results
```

Choose `BCC_A2`, `FCC_A1`, `LIQUID`, or `GAS`. The command reads the phase settings from `FE-params.json` (override with `--params`) and selects matching Fe CPM datasets. It writes a fitted-parameters JSON file and a comparison plot to the output directory. Review the fit before using its parameters in the ESPEI workflow; the command does not update the database or reference states.

For a phase with a `melt` model, the Einstein (or Holzapfel) and magnetic contributions remain active above `T_melt`; the other solid contributions are replaced by the melt correction. When `liquid_Cp` is specified, the fit fixes the melt expression's `a` parameter to `liquid_Cp - 3R` when an Einstein or Holzapfel contribution is present, so the total heat capacity approaches `liquid_Cp`.

The generated custom Gibbs reference state retains pycalphad's Einstein and Xiong solid contributions above `T_melt` and matches Gibbs energy and entropy at the boundary.

The Xiong magnetic heat capacity follows `Cp = -T * d²G/dT²` for the Gibbs energy in equations (12)–(14) of Xiong et al. (2012), doi:10.1016/j.calphad.2012.07.002. Earlier versions included an extra `T/Tcrit` multiplier in the magnetic heat capacity. Fits made with that expression should be repeated against the experimental CPM datasets and reviewed before updating database parameters; existing fitted parameters and databases are not automatically changed.
