from importlib import resources as impresources
import json
from copy import deepcopy
import libreCalphad.models.heat_capacity as hc
import numpy as np
import pandas as pd
import pytest
import symengine as se


@pytest.mark.filterwarnings("error")
def test_calc_RSE_preserves_nonzero_relative_residuals():
    observed = np.array([10.0, 20.0, -30.0, 1e-20])
    predicted = np.array([11.0, 18.0, -27.0, 1.2e-20])
    expected = np.sqrt(np.sum(np.square((predicted - observed) / observed)))

    assert hc._calc_RSE(predicted, observed, []) == expected


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    "predicted,observed,expected",
    [
        ([0.0, 0.0], [0.0, 0.0], 0.0),
        ([0.0, 11.0, 18.0], [0.0, 10.0, 20.0], np.sqrt(0.02)),
    ],
)
def test_calc_RSE_matching_zeros(predicted, observed, expected):
    result = hc._calc_RSE(np.array(predicted), np.array(observed), [])

    assert result == pytest.approx(expected)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("prediction", [1.0, -1.0, 1e-20])
def test_calc_RSE_penalizes_nonzero_predictions_at_zero(prediction):
    predicted = np.array([10.0, prediction])
    observed = np.array([10.0, 0.0])

    assert hc._calc_RSE(predicted, observed, []) == np.inf


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("invalid_array", ["predicted", "observed", "both"])
@pytest.mark.parametrize("other_value", [0.0, 10.0])
def test_calc_RSE_rejects_nonfinite_values(invalid_value, invalid_array, other_value):
    predicted = np.array([other_value, 20.0])
    observed = np.array([other_value, 20.0])
    if invalid_array in ("predicted", "both"):
        predicted[0] = invalid_value
    if invalid_array in ("observed", "both"):
        observed[0] = invalid_value

    assert hc._calc_RSE(predicted, observed, []) == np.inf


def test_fit_einstein_Cp():
    data_file = (
        impresources.files("libreCalphad.tests.test_heat_capacity_files")
        / "test_einstein_data.json"
    )
    with open(data_file, "r") as f:
        test_data = [json.load(f)]
    model_dict = {"einstein": {"theta": [250, "fit"]}}
    fits, model_dict = hc.fit_heat_capacity(test_data, model_dict)
    assert np.isclose(model_dict["einstein"]["theta"][0], 300)


def test_repeated_fit_does_not_change_input_models():
    data_file = (
        impresources.files("libreCalphad.tests.test_heat_capacity_files")
        / "test_einstein_data.json"
    )
    with open(data_file, "r") as f:
        test_data = [json.load(f)]
    models = {"einstein": {"theta": [250, "fit"]}}

    _, first_fit = hc.fit_heat_capacity(test_data, models)
    _, second_fit = hc.fit_heat_capacity(test_data, models)

    assert models == {"einstein": {"theta": [250, "fit"]}}
    assert np.isclose(first_fit["einstein"]["theta"][0], 300)
    assert np.isclose(second_fit["einstein"]["theta"][0], 300)


def test_fit_einstein_Cp_with_bounds():
    data_file = (
        impresources.files("libreCalphad.tests.test_heat_capacity_files")
        / "test_einstein_data.json"
    )
    with open(data_file, "r") as f:
        test_data = [json.load(f)]
    model_dict = {"einstein": {"theta": [250, [200, 400], "fit"]}}
    fits, model_dict = hc.fit_heat_capacity(test_data, model_dict)
    assert np.isclose(model_dict["einstein"]["theta"][0], 300)


def test_fix_einstein_Cp():
    data_file = (
        impresources.files("libreCalphad.tests.test_heat_capacity_files")
        / "test_einstein_data.json"
    )
    with open(data_file, "r") as f:
        test_data = [json.load(f)]
    model_dict = {"einstein": {"theta": [300, "fix"]}}
    fits, model_dict = hc.fit_heat_capacity(test_data, model_dict)
    assert np.isclose(fits, 0, atol=1e-5)


def test_fit_holzapfel_Cp():
    data_file = (
        impresources.files("libreCalphad.tests.test_heat_capacity_files")
        / "test_holzapfel_data.json"
    )
    with open(data_file, "r") as f:
        test_data = [json.load(f)]
    model_dict = {"holzapfel": {"theta": [250, "fit"]}}
    fits, model_dict = hc.fit_heat_capacity(test_data, model_dict)
    assert np.isclose(model_dict["holzapfel"]["theta"][0], 300)


def test_fix_holzapfel_Cp():
    data_file = (
        impresources.files("libreCalphad.tests.test_heat_capacity_files")
        / "test_holzapfel_data.json"
    )
    with open(data_file, "r") as f:
        test_data = [json.load(f)]
    model_dict = {"holzapfel": {"theta": [300, "fix"]}}
    fits, model_dict = hc.fit_heat_capacity(test_data, model_dict)
    assert np.isclose(fits, 0, atol=1e-5)


def test_melt_replaces_solid_and_approaches_liquid_capacity():
    temps = np.array([500.0, 1000.0, 2000.0, 1e6])
    solid = hc._holzapfel_debye_Cp(temps, thetaD=300)
    magnetic = hc._xiong_Cp(temps, beta=2, p=0.37, Tc=800)
    correction = hc._melt_Cp(
        temps[2:], T_melt=1000, a=46 - 3 * hc.R, b=1e19, c=0
    )
    liquid = solid[2:] + magnetic[2:] + correction
    data = pd.DataFrame(
        {
            "temperature": temps,
            "Cp": [*(solid[:2] + magnetic[:2]), *liquid],
            "reference": "test",
        }
    )
    models = {
        "holzapfel": {"theta": [300, "fix"]},
        "xiong": {"beta": [2, "fix"], "p": [0.37, "fix"], "Tc": [800, "fix"]},
        "melt": {
            "T_melt": [1000, "fix"],
            "liquid_Cp": [46, "fix"],
            "a": [20, "fit"],
            "b": [1e19, "fix"],
            "c": [0, "fix"],
        },
    }
    fixed_models = deepcopy(models)
    fixed_models["melt"]["a"] = [46 - 3 * hc.R, "fix"]
    args = {
        "temperature_array": temps,
        "Cp_array": data["Cp"].values,
        "models": fixed_models,
    }
    assert np.isclose(hc._fit_heat_capacity([], args), 0)
    assert np.isclose(liquid[-1], 46, atol=0.01)
    _, fitted = hc.fit_heat_capacity(data, models)
    assert fitted["melt"]["a"] == [46 - 3 * hc.R, "fix"]
    assert models["melt"]["a"] == [20, "fit"]


def _xiong_gibbs_heat_capacity(temperatures, beta, p, critical_temperature):
    # Xiong et al. (2012), equations (12)-(14), doi:10.1016/j.calphad.2012.07.002.
    temperature = se.Symbol("T")
    tau = temperature / critical_temperature
    denominator = 0.33471979 + 0.49649686 * (1 / p - 1)
    low_temperature_g = 1 - (
        0.38438376 / (p * tau)
        + 0.63570895
        * (1 / p - 1)
        * (tau**3 / 6 + tau**9 / 135 + tau**15 / 600 + tau**21 / 1617)
    ) / denominator
    high_temperature_g = -(
        tau**-7 / 21 + tau**-21 / 630 + tau**-35 / 2975 + tau**-49 / 8232
    ) / denominator
    capacities = []
    for g in (low_temperature_g, high_temperature_g):
        gibbs = hc.R * temperature * se.log(beta + 1) * g
        capacities.append(-temperature * gibbs.diff(temperature, 2))
    return np.array(
        [
            0.0
            if temp == 0
            else float(
                capacities[int(temp > critical_temperature)].subs(temperature, temp)
            )
            for temp in temperatures
        ]
    )


@pytest.fixture
def xiong_data():
    temperatures = np.linspace(300, 1500, 200)
    return pd.DataFrame(
        {
            "temperature": temperatures,
            "Cp": _xiong_gibbs_heat_capacity(temperatures, 2, 0.25, 1000),
            "reference": "Xiong2012",
        }
    )


@pytest.mark.parametrize(
    "critical_temperature_key,beta,p,critical_temperature",
    [
        ("Tc", 2.22, 0.37, 1043),
        ("Tn", 0.7, 0.25, 67),
        ("Tc", 0, 0.37, 1043),
        ("Tn", 0, 0.25, 67),
    ],
)
def test_xiong_Cp_matches_gibbs_derivative(
    critical_temperature_key, beta, p, critical_temperature
):
    temperatures = np.concatenate(
        (
            [0],
            critical_temperature * np.geomspace(0.01, 10, 60),
            critical_temperature * np.array([0.99999, 1, 1.00001]),
        )
    )
    parameters = {"beta": beta, "p": p, critical_temperature_key: critical_temperature}
    expected = _xiong_gibbs_heat_capacity(temperatures, beta, p, critical_temperature)
    np.testing.assert_allclose(
        hc._xiong_Cp(temperatures, **parameters), expected, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        [hc._xiong_Cp(float(temp), **parameters) for temp in temperatures],
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_fit_xiong_Cp(xiong_data):
    model_dict = {
        "xiong": {"beta": [1.8, "fit"], "p": [0.20, "fit"], "Tc": [1300, "fit"]}
    }
    fits, model_dict = hc.fit_heat_capacity(xiong_data, model_dict)
    assert np.isclose(model_dict["xiong"]["beta"][0], 2, rtol=0.05)
    assert np.isclose(model_dict["xiong"]["p"][0], 0.25, rtol=1e-2)
    assert np.isclose(model_dict["xiong"]["Tc"][0], 1000, rtol=1)


def test_fix_xiong_Cp(xiong_data):
    model_dict = {
        "xiong": {"beta": [2, "fix"], "p": [0.25, "fix"], "Tc": [1000, "fix"]}
    }
    fits, model_dict = hc.fit_heat_capacity(xiong_data, model_dict)
    assert np.isclose(fits, 0)


def test_fit_bcm_Cp():
    data_file = (
        impresources.files("libreCalphad.tests.test_heat_capacity_files")
        / "test_bcm_data.json"
    )
    with open(data_file, "r") as f:
        test_data = [json.load(f)]
    model_dict = {
        "bcm": {
            "beta_1": [0.1, "fit"],
            "beta_2": [0.2, "fit"],
            "tau": [450, "fit"],
            "gamma": [20, "fit"],
            "T_melt": [1000, "fix"],
        }
    }

    fits, model_dict = hc.fit_heat_capacity(test_data, model_dict)
    assert np.isclose(model_dict["bcm"]["beta_1"][0], 0.0242859)
    assert np.isclose(model_dict["bcm"]["beta_2"][0], 0.1995009)
    assert np.isclose(model_dict["bcm"]["tau"][0], 700.0)
    assert np.isclose(model_dict["bcm"]["gamma"][0], 8.6405188)
    assert np.isclose(fits.fun, 1.328524, atol=1e-5)


def test_fix_bcm_Cp():
    data_file = (
        impresources.files("libreCalphad.tests.test_heat_capacity_files")
        / "test_bcm_data.json"
    )
    with open(data_file, "r") as f:
        test_data = [json.load(f)]
    model_dict = {
        "bcm": {
            "beta_1": [0.025, "fix"],
            "beta_2": [0.3, "fix"],
            "tau": [750, "fix"],
            "gamma": [50, "fix"],
            "T_melt": [1000, "fix"],
        }
    }
    fits, model_dict = hc.fit_heat_capacity(test_data, model_dict)
    assert np.isclose(fits, 0)


def test_symbolic_Cp_calc():
    # test with linear Cp
    temps = np.linspace(0, 1000)
    T, a = se.symbols("T a")
    Cp_func = a * T
    variable_values = [0.25]
    # Test single value
    Cp_0 = hc._symbolic_Cp(temps[0], variable_values, Cp_func, (0, np.inf))
    assert np.isclose(Cp_0, 0)
    # test array
    Cp_array = hc._symbolic_Cp(temps, variable_values, Cp_func, (0, np.inf))
    assert np.isclose(Cp_array[-1], 250)


def test_fit_symbolic_Cp_symengine_expression():
    temps = np.linspace(0, 1000)
    T, a = se.symbols("T a")
    Cp_func = a * T
    Cp_array = [np.float64(Cp_func.subs([T, a], [temp, 0.25])) for temp in temps]
    model_dict = {"symbolic": {"expression": Cp_func, "a": [0.25, "fix"]}}
    data_df = pd.DataFrame({"temperature": temps, "Cp": Cp_array, "reference": "test"})
    fits, model_dict = hc.fit_heat_capacity(data_df, model_dict)
    assert np.isclose(model_dict["symbolic"][str(a)][0], 0.25)


def test_print_all_fixed_params():
    data_file = (
        impresources.files("libreCalphad.tests.test_heat_capacity_files")
        / "test_einstein_data.json"
    )
    with open(data_file, "r") as f:
        test_data = [json.load(f)]
    model_dict = {"einstein": {"theta": [300, "fix"]}}
    fits, model_dict = hc.fit_heat_capacity(test_data, model_dict, verbose=True)


def test_fix_twostate_Cp():
    T = se.symbols("T")
    data_file = (
        impresources.files("libreCalphad.tests.test_heat_capacity_files")
        / "test_twostate_data.json"
    )
    with open(data_file, "r") as f:
        test_data = [json.load(f)]
    model_dict = {
        "two-state": {
            "expression": "a + b*T + c*T*log(T)",
            "a": [15000, "fix"],
            "b": [-5, "fix"],
            "c": [-1, "fix"],
        }
    }
    fits, model_dict = hc.fit_heat_capacity(test_data, model_dict)
    assert np.isclose(fits, 0, atol=1e-7)


def test_fit_twostate_Cp_no_bounds():
    T = se.symbols("T")
    data_file = (
        impresources.files("libreCalphad.tests.test_heat_capacity_files")
        / "test_twostate_data.json"
    )
    with open(data_file, "r") as f:
        test_data = [json.load(f)]
    model_dict = {
        "two-state": {
            "expression": "a + b*T + c*T*log(T)",
            "a": [10000, "fit"],
            "b": [-2, "fit"],
            "c": [-2, "fit"],
        }
    }
    fits, model_dict = hc.fit_heat_capacity(test_data, model_dict)
    assert np.isclose(model_dict["two-state"]["a"][0], 15000, atol=1e-4)
    assert np.isclose(model_dict["two-state"]["b"][0], -5, atol=1e-4)
    assert np.isclose(model_dict["two-state"]["c"][0], -1, atol=1e-4)
    assert np.isclose(fits.fun, 0, atol=1e-6)


def test_fit_twostate_Cp_with_bounds():
    T = se.symbols("T")
    data_file = (
        impresources.files("libreCalphad.tests.test_heat_capacity_files")
        / "test_twostate_data.json"
    )
    with open(data_file, "r") as f:
        test_data = [json.load(f)]
    model_dict = {
        "two-state": {
            "expression": "a + b*T + c*T*log(T)",
            "a": [10000, [7500, 17500], "fit"],
            "b": [-2, [-10, 0], "fit"],
            "c": [-2, [-5, 5], "fit"],
        }
    }
    fits, model_dict = hc.fit_heat_capacity(test_data, model_dict)
    assert np.isclose(model_dict["two-state"]["a"][0], 15000, atol=1e-4)
    assert np.isclose(model_dict["two-state"]["b"][0], -5, atol=1e-4)
    assert np.isclose(model_dict["two-state"]["c"][0], -1, atol=1e-4)
    assert np.isclose(fits.fun, 0, atol=1e-6)
