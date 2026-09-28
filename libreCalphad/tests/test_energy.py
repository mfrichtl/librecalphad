from importlib import resources as impresources
import json
from libreCalphad.databases.db_utils import load_database, upsert_db_param_from_models
import libreCalphad.models.energy as en
import numpy as np
import pandas as pd
from pycalphad import Model, variables as v
import pytest
import symengine as se


def test_symbolic_gibbs():
    T, a = se.symbols("T a")
    Cp_f = a * T
    temp_bounds = (0, 1000)
    gibbs = en._symbolic_gibbs(100, [2], Cp_f, temp_bounds, ret_expr=True)
    assert np.isclose(gibbs.subs([T, a], [100, 2]), -10000)
    gibbs = en._symbolic_gibbs(100, [2], Cp_f, temp_bounds)
    assert np.isclose(gibbs, -10000)
    gibbs = en._symbolic_gibbs([0, 100], [2], Cp_f, temp_bounds)
    assert np.isclose(np.sum(gibbs), -10000)


def test_create_espei_custom_refstate_stable_offset():
    model_dict = {
        "offset": {"enthalpy": [477.094, "fix"], "entropy": [9.77906, "fix"]}
    }
    gibbs = en.create_espei_custom_refstate_stable(model_dict)
    assert np.isclose(
        float(gibbs.subs(se.Symbol("T"), 300)), 477.094 - 9.77906 * 300
    )


def test_melt_reference_state_matches_fitted_capacity_and_is_smooth():
    models = {
        "einstein": {"theta": [300, "fix"]},
        "xiong": {"beta": [2, "fix"], "p": [0.37, "fix"], "Tc": [1000, "fix"]},
        "symbolic_LT": {
            "expression": "a*T",
            "temp_bounds": [0, 1811],
            "a": [0.005, "fix"],
        },
        "melt": {
            "T_melt": [1811, "fix"],
            "a": [46, "fix"],
            "b": [1e20, "fix"],
            "c": [-1e39, "fix"],
        },
    }
    T = se.Symbol("T")
    builtin = en._built_in_solid_gibbs_above_melt(models, 1811)
    total = en.create_espei_custom_refstate_stable(models) + builtin
    capacity = -T * total.diff(T, 2)
    for temperature in (1900, 3000, 9000):
        expected = 46 + 1e20 * temperature**-6 - 1e39 * temperature**-12
        assert np.isclose(float(capacity.subs({T: temperature})), expected, rtol=1e-5)
    assert np.isclose(
        float(total.subs({T: 1811 - 1e-4})),
        float(total.subs({T: 1811 + 1e-4})),
        atol=1,
    )
    assert np.isclose(
        float(total.diff(T).subs({T: 1811 - 1e-4})),
        float(total.diff(T).subs({T: 1811 + 1e-4})),
        atol=0.01,
    )


def test_melt_reference_state_rejects_magnetic_transition_above_melting():
    models = {
        "xiong": {"beta": [2], "p": [0.37], "Tc": [2000]},
        "offset": {"enthalpy": [100, "fix"], "entropy": [1, "fix"]},
        "melt": {"T_melt": [1811], "a": [46], "b": [0], "c": [0]},
    }
    with pytest.raises(ValueError, match="critical temperature must be below T_melt"):
        en.create_espei_custom_refstate_stable(models)


def test_melt_cancellation_matches_pycalphad_fe_builtin_terms():
    database = load_database("LC-steels-input.xml")
    species = {item.name: item for item in database.species}
    models = {
        "einstein": {"theta": [300]},
        "xiong": {"beta": [2], "p": [0.37], "Tc": [1000]},
    }
    database = upsert_db_param_from_models(
        database, models, "BCC_A2", ((species["FE"],), (species["VA"],))
    )
    phase = Model(database, ["FE", "VA"], "BCC_A2")
    builtins = phase.einstein_energy(database) + phase.magnetic_energy(database)
    cancellation = en._built_in_solid_gibbs_above_melt(models, 1811)
    for temperature in (1900, 3000, 9000):
        conditions = {
            v.T: temperature,
            v.SiteFraction("BCC_A2", 0, "FE"): 1,
            v.SiteFraction("BCC_A2", 1, "VA"): 1,
        }
        assert np.isclose(
            float(builtins.subs(conditions)),
            float(cancellation.subs({v.T: temperature})),
            atol=1e-6,
        )
