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

Numerical Debye supports Cp fitting and plotting, plus the exact energy functions
below, but **not pycalphad Gibbs reference-state or database export**. Those export
functions reject Debye models explicitly. For the existing pycalphad export
workflow, fit an Einstein model separately rather than reusing a Debye temperature
as an Einstein temperature. See the
[Fe fitting workflow](libreCalphad/databases/LC-steels_validation/FE/README.md) for
experimental-data selection and output review.

## Debye Gibbs energy, enthalpy, and entropy

For a constant Debye temperature, define $y=\theta_D/T$ and

$$D_3(y)=\frac{3}{y^3}\int_0^y\frac{x^3}{e^x-1}\,dx.$$

The corresponding molar lattice energies are

$$G_D=\frac{9}{8}R\theta_D+RT\left[3\ln(1-e^{-y})-D_3(y)\right],$$
$$H_D=\frac{9}{8}R\theta_D+3RTD_3(y),\qquad
S_D=R\left[4D_3(y)-3\ln(1-e^{-y})\right].$$

These satisfy $G_D=H_D-TS_D$ and $C_p=-T\,d^2G_D/dT^2$ for the same fixed-theta
lattice model used by `_debye_Cp`. No thermal-expansion or separate $C_p-C_v$
correction is included. The convention includes zero-point energy
$H_D(0)=G_D(0)=9R\theta_D/8$ and $S_D(0)=0$; add $H_0-TS_0$ to Gibbs energy for
other enthalpy/entropy reference offsets.

```python
from libreCalphad.models import energy

gibbs = energy._debye_gibbs([0, 300, 1500], theta=300)  # J/mol
enthalpy = energy._debye_enthalpy([0, 300, 1500], theta=300)  # J/mol
entropy = energy._debye_entropy([0, 300, 1500], theta=300)  # J/mol/K
gibbs_expression = energy._debye_gibbs(None, theta=300, ret_expr=True)
```

Numeric evaluation uses quadrature of the exact Debye integral. All three helpers
accept scalars or temperature arrays (finite and nonnegative), with positive finite
theta. `ret_expr=True` ignores the temperature argument and returns an exact SymPy
expression in `T`, with the zero-temperature limit handled explicitly; symbolic
positive theta is also supported. These integral expressions can be differentiated
and evaluated in SymPy, but cannot be compiled by the current pycalphad/SymEngine
LLVM backend. The export safeguards therefore remain in place.

## Magnetic energy reference

The standalone `_xiong_gibbs`, `_xiong_entropy`, and `_xiong_enthalpy` helpers in
`libreCalphad.models.energy` use the magnetic ground state as their reference:
the magnetic contributions satisfy `G(0) = H(0) = S(0) = 0`. They convert the
ordering energy in [Xiong et al. (2012), equations (12)–(14)](https://doi.org/10.1016/j.calphad.2012.07.002)
using `G_ground(T) = G_Xiong(T) - G_Xiong(0) - T*G_Xiong'(0)`.

With `F` denoting the appropriate `_g_int` series, Gibbs energy is
`-R*T*log(beta+1)*F/D` below and at `Tc`. Only the branch above `Tc` adds the
disordering term `-R*log(beta+1)*(T - 0.38438376*Tc/(p*D))`. Entropy is calculated
from the full derivative `S = -dG/dT`, and enthalpy from `H = G + T*S`. Numerical
and symbolic Gibbs evaluation use the same gas constant. For symbolic Gibbs,
`ret_expr=True` returns the expression for the branch selected by `T_arr`, not
a complete piecewise expression.

This corrects the previous low-temperature Gibbs offset and constant entropy;
it does not change the magnetic Cp expression or existing fitted parameters.
Published rounded coefficients leave very small residual mismatches at `Tc`
(about `1.1e-4` J/mol in G and `1.6e-9` J/mol/K in S for beta=2, p=0.37,
Tc=1000 K). The standalone ground-state reference must not be added to pycalphad's
built-in magnetic contribution without accounting for its different reference.
No database or exported reference-state changes are made by these helpers.

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
