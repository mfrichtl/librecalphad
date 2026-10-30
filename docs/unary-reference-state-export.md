# Exporting fitted unary endmembers through ESPEI

## Scope and responsibilities

Unary endmembers are implemented as ESPEI reference states using the companion
[librecalphad-refstates repository](https://github.com/mfrichtl/librecalphad-refstates).
They are not exported merely by copying all fitted numbers into database
parameters. The existing architecture has two coordinated channels:

| Channel | Owner | Responsibility |
|---|---|---|
| Fitting inputs and results | librecalphad | Fit experimental CPM with `libreCalphad/models/heat_capacity.py`; review fitted models and diagnostics. |
| Reference-state registry | librecalphad-refstates, `LCRefstates.json` | Store approved model dictionaries and stable-phase symbol aliases. |
| Gibbs expression builder | librecalphad, `libreCalphad/models/energy.py` | `create_espei_custom_refstate_stable` builds the non-native Gibbs terms used for both stable and other phase entries. |
| ESPEI plugin | librecalphad-refstates, `refstate.py` and `setup.py` | Register `LCRefState` and construct the three dictionaries consumed by ESPEI. |
| Native model parameters | librecalphad, `libreCalphad/databases/db_utils.py` | `upsert_db_param_from_models` supplies native pycalphad contributions in a seed database. |
| SER metadata | librecalphad-refstates, `LCSERparams.json` | Supply reference phase, molar mass, and reference H/S metadata. |

This document describes the implemented data flow and recommended review process.
It does **not** imply that the currently known Cp export mismatches have been fixed
or that existing assessment outputs have been accepted.

The [endmember audit report API](endmember-export-audit.md) is the reusable
reporting foundation for this workflow. It supports compound as well as unary
endmember identities, but currently assembles supplied findings only; it does not
yet perform the registry, database, or thermodynamic comparisons described here.

## Plugin contract and key mapping

`setup.py` registers the entry point `LCRefState = refstate` in the
`espei.reference_states` group. At import time, `refstate.py` reads the two JSON
files beside the installed module and creates:

- `LCRefStateStable`: element name to stable reference Gibbs expression.
- `LCRefState`: `(element, phase)` to the phase's reference Gibbs expression.
- `LCRefStateSER`: element name to SER metadata.

For Fe, the intended registry layout is:

| `LCRefstates.json` key | Value | Plugin destination |
|---|---|---|
| `FE` | Approved BCC model dictionary | `LCRefStateStable["FE"]` |
| `FE-BCC_A2` | String `"GHSERFE"` | `LCRefState[("FE", "BCC_A2")]`, referring to the stable function |
| `FE-FCC_A1` | Approved FCC model dictionary | `LCRefState[("FE", "FCC_A1")]` |
| `FE-LIQUID` | Approved liquid model dictionary | `LCRefState[("FE", "LIQUID")]` |
| `FE-GAS` | Approved gas model dictionary, when exported | `LCRefState[("FE", "GAS")]` |

These are required mappings for the phases being exported, not an assertion that
the current registry contains every row. A phase entry is the reference Gibbs
expression ESPEI should assign to that phase's unary endmember. It is **not** an
automatic difference from BCC: ESPEI does not add `GHSERFE` to a model dictionary's
expression unless that expression explicitly contains the symbol. The stable-phase
alias prevents duplicating the stable expression.

The fit artifact has a different shape: `_fitted_params.json` contains models at
`["FE"]["BCC_A2"]`, `["FE"]["FCC_A1"]`, etc. Select and review each model before
mapping it into the plugin registry; do not copy that outer nesting into
`LCRefstates.json`.

`upsert_custom_refstate_json` is the existing registry-writing helper. With
`phase=None`, it writes the bare element key; with `phase="FCC_A1"`, it writes
`FE-FCC_A1`. It also stores fit provenance as `Cp_fits.x` and `Cp_fits.RSE` when
given the fit result. It does not create the stable-phase alias, update SER
metadata, export native parameters, or validate the resulting database.

## Which parameters go where

| Model or fields | Current channel and required interpretation |
|---|---|
| `einstein.theta` | Native `THETA=ln(theta)`, not theta itself. Do not also add the full Einstein Gibbs term to the reference state. |
| `xiong.beta`, `Tc`, `Tn` | Native `BMAGN`, `TC`, `NT`. Verify that the correct magnetic model is enabled by the phase hints and that stale parameters do not activate another branch. |
| `xiong.p` | Currently skipped by the native exporter. pycalphad uses a phase-wide structure-factor hint; this is an unresolved export limitation, not optional fit metadata. |
| `two-state.expression` and coefficients | Native `GD`, after substitution by coefficient name. `GD` is the Gibbs difference between states, not their enthalpy difference. |
| `symbolic`, `symbolic_LT`, etc. | Reference-state Gibbs terms obtained by integrating Cp, honoring temperature bounds. |
| `melt.T_melt`, `a`, `b`, `c` | Piecewise reference-state Gibbs correction. The total Cp must follow the fitter's melt masking, not simply add all models above melting. |
| `offset.HM`/`SM` or `enthalpy`/`entropy` | Reference-state term `H0-T*S0`, counted once with an explicit reference convention. |
| `melt.solid_enthalpy`, `solid_entropy` | Legacy affine terms that cancel in the current continuity-matched melt construction. Do not introduce them as extra offsets without defining their intended role. |
| `melt.liquid_Cp` | A fitting target used to determine the melt constant, not an extra additive database contribution. |
| `fit`/`fix`, bounds, optimizer indices, `symbols`, `variable_values`, `Cp_fits.*`, `HM_increment`/`SM_increment` | Instructions, derived values, or provenance; not independent thermodynamic contributions. Preserve provenance, but derive expressions from named coefficients. |

The reference-state builder deliberately omits native Einstein, magnetic, and
two-state Gibbs terms. Therefore, successfully loading the plugin does not prove
that all fitted physics has been exported. Conversely, adding these full terms to
the reference state without accounting for the native model would double count
them. Any future endmember-specific compatibility correction belongs in this
coordinated reference-state/native-parameter design, not in an unrelated parallel
unary export path.

Keep the current Debye export rejection. Numerical Debye energies do not make
their integral expressions compilable by the installed pycalphad backend, and a
Debye theta cannot be used as an Einstein `THETA` parameter. Do not assume other
unsupported lattice models are exported just because a registry accepts their keys.

## SER metadata and reference conventions

`LCSERparams.json` contains entries such as:

```text
FE: {phase, mass, H298, S298}
```

These values describe the reference convention used by the database and ESPEI.
They are not synonymous with `offset.HM` and `offset.SM`, and changing them does
not itself add `H0-T*S0` to Gibbs energy. Do not replace them with the current
model's calculated H/S merely to make a discrepancy disappear. First establish
the intended convention and compare measurements on the same basis; DH/DS
increments cannot determine an arbitrary absolute offset.

The standalone Xiong energy helpers use a magnetic ground-state reference that
differs from pycalphad's native magnetic reference. Resolve that affine reference
difference explicitly before combining expressions or calibrating offsets.

## Recommended staged workflow

1. **Fit and review.** Fit experimental CPM in librecalphad. Retain the full model
   dictionary, input data references, and optimizer diagnostics separately from
   assessment inputs. Saving an unconverged fit is not approval to promote it.
2. **Stage a coordinated export.** After approval, map the relevant unary models
   into `LCRefstates.json`, maintain the stable-phase alias, and export native
   parameters for the exact unary constituent arrays in a disposable seed
   database. Preserve unrelated elements, phases, and registry entries. Update
   SER metadata only when its convention has been established and approved.
3. **Start a fresh generator process.** Finish registry writes before importing
   the plugin or ESPEI in the process used for generation. Select
   `generate_parameters.ref_state: LCRefState` in the ESPEI configuration.
4. **Check the seed, not just the plugin.** Inspect existing reference symbols,
   unary parameters, SER metadata, phase hints, and normalization. Use a seed
   with the intended phase definitions and native parameters, not blindly a
   previously generated assessment. Do not discard unrelated seed content or
   overwrite production databases to force a refresh.
5. **Generate into separate output.** Use ESPEI to assign the plugin reference
   expressions to unary endmembers and perform the requested parameter
   generation. Keep this output separate from source inputs until reviewed.
6. **Validate before promotion.** Compare the same fitted Cp model with the
   resulting pure-endmember pycalphad Cp before and after XML save/reload. Check
   individual contributions, magnetic and melt boundaries (including exact and
   one-sided evaluations), symbolic bounds, and a declared positive-temperature
   domain. Then validate G/H/S reference conventions and offset calibration
   separately. Repeated export should not duplicate contributions or alter
   unrelated endmembers. Unary validation is not validation of alloy mixing.

The installed plugin currently reads JSON from its own source/module directory;
it has no staging-directory command-line option. A disposable installation or
future explicit staging support must be used if a trial needs a separate registry.
Do not assume changing the working directory redirects these resources.

### Two independent stale-data hazards

**Import-time caching:** `refstate.py` builds its dictionaries when imported.
ESPEI then binds those dictionaries into `espei.refdata`. `reload(espei)` does
not rebuild the plugin dictionaries or update an already generated database.
A fresh generation process is the preferred reproducible approach. If an
interactive refresh is implemented, it must refresh both the plugin and ESPEI's
reference-data bindings before rebuilding models; reloading one module is not
a sufficient export protocol.

**Seed preservation:** in ESPEI 0.9.1, `initialize_database` uses `setdefault`
for `GHSER` symbols and preserves existing SER rows and phase definitions.
`phase_fit` skips an existing unary `G` parameter. Thus even a fresh process can
retain old reference data from its supplied database. Missing SER metadata can
also fall back to SGTE91. Explicitly verify that the generated database contains
the intended plugin functions, unary references, and SER values. Changing JSON
and observing a successful generation call is not enough.

When a unary reference is inserted from the plugin, ESPEI scales the per-mole
reference expression by the number of non-vacancy atoms in the endmember formula.
Keep the native-parameter units and pycalphad site-ratio normalization consistent;
do not multiply all parameters by the total number of sites including vacancies.

## Current limitations to resolve incrementally

These are observed implementation issues, not instructions to silently modify
existing fits, dependencies, or phase definitions:

- Fitted magnetic `p` is not transferred. Changing a shared phase hint for one
  element can affect other unary and alloy descriptions; decide on a tested,
  endmember-safe treatment before implementation.
- The installed pycalphad 0.11.2 low-temperature Xiong series differs from the Cp
  fitter's series. Matching parameter values alone does not remove that mismatch.
- Native two-state Cp remains active above `T_melt`, whereas the Cp fitter
  replaces non-lattice/non-magnetic solid contributions there. A correction must
  also preserve the intended G/S boundary matching.
- The fitter uses `R=8.314472`; pycalphad 0.11.2 uses `R=8.3145`. Set and test the
  numerical convention rather than silently changing existing fitted curves.
- The plugin relies on the Gibbs builder in librecalphad, so changing that code
  can change generated references even if the JSON is unchanged. Record both
  repository revisions and the ESPEI/pycalphad versions with validation results.
- The existing combined FE scripts still mix fitting, registry/database writes,
  validation, and offset adjustment. Their `reload(espei)` calls and source-file
  overwrites are not the recommended staged workflow. Separation and fit
  acceptance rules remain TODOs; no complete new exporter is implemented here.

## Installation and read-only discovery check

Use the same Python environment for librecalphad, ESPEI, pycalphad, and the plugin.
The companion's current `setup.py` declares a single `refstate.py` module but no
JSON resource packaging. An editable checkout is the supported development
workflow; do not assume a non-editable wheel includes the JSON files.

```sh
python -m pip install -e /path/to/librecalphad-refstates
```

In a fresh process, this check discovers the installed plugin and verifies ESPEI's
dictionary bindings without changing either repository or generating a database:

```python
from importlib.metadata import entry_points
import espei.refdata

plugins = [ep for ep in entry_points(group="espei.reference_states")
           if ep.name == "LCRefState"]
assert len(plugins) == 1, "Expected exactly one LCRefState plugin"
plugin = plugins[0].load()
for name in ("LCRefState", "LCRefStateStable", "LCRefStateSER"):
    assert getattr(espei.refdata, name) is getattr(plugin, name)
print("Plugin source:", plugin.__file__)
print("Stable elements:", sorted(plugin.LCRefStateStable))
print("Unary phases:", sorted(plugin.LCRefState))
```

Verify the printed source is the intended checkout. This is only a discovery
check: it does not establish Cp agreement, complete registry coverage, suitable
SER conventions, or fit acceptance.
