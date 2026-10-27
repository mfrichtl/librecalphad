# About
Welcome to the LibreCalphad project. The purpose is to create a set of open CALPHAD databases and property models for use with open-source modeling software, such as [pycalphad](https://pycalphad.org/docs/latest/#). It is currently (and probably always will be) a work in progress. User discretion is advised. I make no assertions as to the accuracy of the results and highly recommend you validate them versus other CALPHAD software or reliable references.  If you want to help make this software better, please do. See the contributing section below for more information.

# Steel Database
- The steel database file (mf-steel.tdb) is decent, but there are too many opportunities for improvement to list. Known errors are documented in the frontmatter, though other issues probably exist.
- The mobility and molar volume databases (mf-mobility.tdb and mf-volume.tdb) are designed to be used in conjunction with mf-steel.tdb. Therefore, things like phase definitions are not contained in these support databases, which will likely cause errors if you try to use them standalone.

The rest of the files don't really do much for now. More will be added to this at a later date.

# Property Models
A martensite-start temperature model for steels is currently available. It still needs much work, but is a good start for collaboration. See the README in it's folder or the notebook in the examples folder for more information.

# Heat-capacity fitting

Numerical Debye fitting is available through the standard Cp fitting interface:

```python
from libreCalphad.models.heat_capacity import fit_heat_capacity
from libreCalphad.plotting import plot_heat_capacity_from_models

models = {"debye": {"theta": [250, [200, 400], "fit"]}}
fits, fitted_models = fit_heat_capacity(cpm_datasets, models)
fig, ax = plot_heat_capacity_from_models(fitted_models, cpm_datasets)
```

Here `cpm_datasets` is a list of ESPEI CPM datasets. Fitting also accepts a DataFrame
with `temperature`, `Cp`, and `reference` columns. Debye temperatures are in kelvin:
use `[300, "fix"]` for a fixed value or `[250, "fit"]` for default bounds
`(1e-6, 2500)`. Custom bounds must be positive, ordered, and contain the initial
value. Select only one lattice model: `einstein`, `holzapfel`, or `debye`.

Numerical Debye currently supports Cp fitting and plotting, **not Gibbs reference
states or database export**; those export functions reject Debye models explicitly.
For the existing pycalphad export workflow, fit an Einstein model separately rather
than reusing a Debye temperature as an Einstein temperature. See the
[Fe fitting workflow](libreCalphad/databases/LC-steels_validation/FE/README.md) for
experimental-data selection and output review.

# Contributing
Do you want to help improve this software? Great! Unsolicited pull requests are gladly appreciated! If you find an issue with a system, but do not know how to help fix it, open an issue and provide as much information as you can. Images of current behavior and references to expected behavior will help tremendously. Please check that this particular issue isn't documented elsewhere.

To install the package and run its Python tests locally (Python 3.11–3.13):

```sh
python -m pip install '.[test]'
python -m pytest -q libreCalphad/tests
```

NumPy is constrained to `<2.5` because pycalphad 0.11.2 assigns directly to array
shapes, which emits deprecation warnings with NumPy 2.5. This keeps the plotting
regression tests' warnings-as-errors checks intact until pycalphad supports the
newer API. To apply this constraint to an existing environment, run
`python -m pip install 'numpy<2.5'` before running the tests.
