from importlib import resources as impresources
import json
from libreCalphad.databases.db_utils import load_database, upsert_db_param_from_models
import libreCalphad.models.energy as en
import libreCalphad.models.heat_capacity as hc
import numpy as np
import pandas as pd
from pycalphad import Model, variables as v
import pytest
import symengine as se


@pytest.mark.parametrize("scalar_type", [int, float, np.int64, np.float32, np.float64])
@pytest.mark.parametrize("temperature,expected", [(0, 0), (300, 25.95741206200213)])
def test_einstein_entropy_reference_value(scalar_type, temperature, expected):
    entropy = en._einstein_entropy(scalar_type(temperature), theta=300)

    assert np.isscalar(entropy)
    assert entropy == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    "temperatures",
    [[], [0, 50, 300], (0, 50, 300), np.array([[0, 50], [300, 1500]])],
)
def test_einstein_entropy_arrays(temperatures):
    temperature_array = np.asarray(temperatures)
    expected = np.array(
        [en._einstein_entropy(float(temp), theta=300) for temp in temperature_array.flat]
    ).reshape(temperature_array.shape)

    entropy = en._einstein_entropy(temperatures, theta=300)

    assert isinstance(entropy, np.ndarray)
    assert entropy.shape == temperature_array.shape
    np.testing.assert_allclose(entropy, expected)


@pytest.mark.parametrize("theta", [100, 300])
def test_einstein_entropy_matches_gibbs_and_heat_capacity(theta):
    T = se.Symbol("T")
    gibbs = 3 * en.R * theta / 2 + 3 * en.R * T * se.log(1 - se.exp(-theta / T))
    entropy_from_gibbs = -gibbs.diff(T)
    for temp in theta * np.array([0.03, 0.2, 1.0, 5.0]):
        entropy = en._einstein_entropy(temp, theta)
        assert entropy == pytest.approx(
            float(entropy_from_gibbs.subs({T: temp})), rel=1e-10, abs=1e-14
        )
        step = temp * 1e-5
        capacity = temp * (
            en._einstein_entropy(temp + step, theta)
            - en._einstein_entropy(temp - step, theta)
        ) / (2 * step)
        assert capacity == pytest.approx(hc._einstein_Cp(temp, theta), rel=1e-7)


def test_einstein_entropy_zero_and_low_temperature_limits():
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        entropy = en._einstein_entropy([0, 1e-6, 0.3, 3, 30], theta=300)

    assert np.all(np.isfinite(entropy))
    assert entropy[0] == entropy[1] == entropy[2] == 0
    assert np.all(np.diff(entropy) >= 0)
    # For theta/T >> 1, S approaches 3R*(theta/T + 1)*exp(-theta/T).
    assert entropy[3] == pytest.approx(3 * en.R * 101 * np.exp(-100), rel=1e-12, abs=0)


def test_einstein_entropy_high_temperature_limit():
    temperature, theta = 3e14, 300
    entropy = en._einstein_entropy(temperature, theta)

    assert entropy == pytest.approx(
        3 * en.R * (1 + np.log(temperature / theta)), rel=1e-12
    )


@pytest.mark.parametrize("theta", [0, -300, np.nan, np.inf])
def test_einstein_entropy_rejects_invalid_theta(theta):
    with pytest.raises(ValueError, match="theta must be finite and positive"):
        en._einstein_entropy(300, theta)


@pytest.mark.parametrize("temperature", [-1, np.nan, np.inf])
def test_einstein_entropy_rejects_invalid_temperatures(temperature):
    with pytest.raises(ValueError, match="temperatures must be finite and nonnegative"):
        en._einstein_entropy([300, temperature], theta=300)


@pytest.mark.parametrize(
    "beta,p,Tc", [(2, 0.37, 1000), (0.7, 0.25, 67), (0, 0.37, 1000)]
)
@pytest.mark.parametrize("ratio", [0, 0.3, 1, 1.5])
def test_xiong_gibbs_scalar_matches_symbolic(monkeypatch, beta, p, Tc, ratio):
    # Isolate branch consistency from the existing differing gas constants.
    monkeypatch.setattr(en, "R", float(v.R))
    temperature = float(ratio * Tc)
    expression = en._xiong_gibbs(temperature, beta, p, Tc, ret_expr=True)
    expected = float(expression.subs({v.T: temperature}))

    actual = en._xiong_gibbs(temperature, beta, p, Tc)

    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("container", [list, np.array])
def test_xiong_gibbs_array_matches_symbolic(monkeypatch, container):
    monkeypatch.setattr(en, "R", float(v.R))
    temperatures = [0, 300, 1000, 1500]
    expected = [
        float(en._xiong_gibbs(temp, 2, 0.37, 1000, ret_expr=True).subs({v.T: temp}))
        for temp in temperatures
    ]

    actual = en._xiong_gibbs(container(temperatures), beta=2, p=0.37, Tc=1000)

    assert isinstance(actual, np.ndarray)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_xiong_gibbs_zero_temperature_includes_disordering_offset():
    beta, p, Tc = 2, 0.37, 1000
    denominator = 0.33471979 + 0.49649686 * (1 / p - 1)
    expected = en.R * np.log(beta + 1) * 0.38438376 * Tc / (p * denominator)

    assert en._xiong_gibbs(0, beta, p, Tc) == pytest.approx(expected, rel=1e-12)
    np.testing.assert_allclose(
        en._xiong_gibbs([0], beta, p, Tc), [expected], rtol=1e-12
    )


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


def test_create_espei_custom_refstate_rejects_numerical_debye():
    models = {"debye": {"theta": [300, "fix"]}}

    with pytest.raises(NotImplementedError, match="Debye Gibbs.*not implemented"):
        en.create_espei_custom_refstate_stable(models)


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
