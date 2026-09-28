from importlib import resources as impresources
import json
from libreCalphad.databases.db_utils import load_database, upsert_db_param_from_models
import libreCalphad.models.energy as en
import libreCalphad.models.heat_capacity as hc
import numpy as np
import pandas as pd
from pycalphad import Model, variables as v
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
        "symbolic_LT": {
            "expression": "a*T",
            "temp_bounds": [0, 1811],
            "a": [0.005, "fix"],
        },
        "melt": {
            "T_melt": [1811, "fix"],
            "a": [46 - 3 * hc.R, "fix"],
            "b": [1e20, "fix"],
            "c": [-1e39, "fix"],
        },
    }
    T = se.Symbol("T")
    custom = en.create_espei_custom_refstate_stable(models)
    capacity = -T * custom.diff(T, 2)
    for temperature in (1900, 3000, 9000):
        expected = 46 - 3 * hc.R + 1e20 * temperature**-6 - 1e39 * temperature**-12
        assert np.isclose(float(capacity.subs({T: temperature})), expected, rtol=1e-5)
    assert np.isclose(
        float(custom.subs({T: 1811 - 1e-4})),
        float(custom.subs({T: 1811 + 1e-4})),
        atol=1,
    )
    assert np.isclose(
        float(custom.diff(T).subs({T: 1811 - 1e-4})),
        float(custom.diff(T).subs({T: 1811 + 1e-4})),
        atol=0.01,
    )


def test_melt_reference_keeps_pycalphad_einstein_and_magnetic_heat_capacity():
    models = {
        "einstein": {"theta": [300]},
        "xiong": {"beta": [2], "p": [0.37], "Tc": [1000]},
        "melt": {
            "T_melt": [1811],
            "a": [46 - 3 * hc.R],
            "b": [1e20],
            "c": [0],
        },
    }
    database = load_database("LC-steels-input.xml")
    species = {item.name: item for item in database.species}
    upsert_db_param_from_models(
        database, models, "BCC_A2", ((species["FE"],), (species["VA"],))
    )
    phase = Model(database, ["FE", "VA"], "BCC_A2")
    builtins = phase.einstein_energy(database) + phase.magnetic_energy(database)
    builtin_capacity = -v.T * builtins.diff(v.T, 2)
    custom = en.create_espei_custom_refstate_stable(models)
    custom_capacity = -v.T * custom.diff(v.T, 2)
    for temperature in (1900, 3000, 9000):
        conditions = {
            v.T: temperature,
            v.SiteFraction("BCC_A2", 0, "FE"): 1,
            v.SiteFraction("BCC_A2", 1, "VA"): 1,
        }
        actual = float(builtin_capacity.subs(conditions)) + float(
            custom_capacity.subs({v.T: temperature})
        )
        tau = temperature / 1000
        denominator = 0.33471979 + 0.49649686 * (1 / 0.37 - 1)
        magnetic_capacity = (
            2
            * float(v.R)
            * np.log(3)
            / denominator
            * (tau**-7 + tau**-21 / 3 + tau**-35 / 5 + tau**-49 / 7)
        )
        expected = (
            hc._einstein_Cp(np.array([temperature]), theta=300)[0]
            + magnetic_capacity
            + hc._melt_Cp(temperature, T_melt=1811, a=46 - 3 * hc.R, b=1e20)
        )
        assert np.isclose(actual, expected, atol=0.001)
