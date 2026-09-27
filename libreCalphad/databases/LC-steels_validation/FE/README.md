# Fe heat-capacity fitting

From the repository root, with the package and ESPEI installed:

```sh
python libreCalphad/databases/LC-steels_validation/FE/fit_cpm.py \
  --datasets /path/to/ESPEI-datasets/datasets \
  --phase BCC_A2 \
  --output-dir /path/to/fit-results
```

Choose `BCC_A2`, `FCC_A1`, `LIQUID`, or `GAS`. The command reads the phase settings from `FE-params.json` (override with `--params`) and selects matching Fe CPM datasets. It writes a fitted-parameters JSON file and a comparison plot to the output directory. Review the fit before using its parameters in the ESPEI workflow; the command does not update the database or reference states.
