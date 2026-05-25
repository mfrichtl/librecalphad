from importlib import resources as impresources
import json
import libreCalphad.models.heat_capacity as hc
import numpy as np
import pandas as pd
import symengine as se


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


def test_fit_xiong_Cp():
    data_file = (
        impresources.files("libreCalphad.tests.test_heat_capacity_files")
        / "test_xiong_data.json"
    )
    with open(data_file, "r") as f:
        test_data = [json.load(f)]
    model_dict = {
        "xiong": {"beta": [1.8, "fit"], "p": [0.20, "fit"], "Tc": [1300, "fit"]}
    }
    fits, model_dict = hc.fit_heat_capacity(test_data, model_dict)
    assert np.isclose(model_dict["xiong"]["beta"][0], 2, rtol=0.05)
    assert np.isclose(model_dict["xiong"]["p"][0], 0.25, rtol=1e-2)
    assert np.isclose(model_dict["xiong"]["Tc"][0], 1000, rtol=1)


def test_fix_xiong_Cp():
    data_file = (
        impresources.files("libreCalphad.tests.test_heat_capacity_files")
        / "test_xiong_data.json"
    )
    with open(data_file, "r") as f:
        test_data = [json.load(f)]
    model_dict = {
        "xiong": {"beta": [2, "fix"], "p": [0.25, "fix"], "Tc": [1000, "fix"]}
    }
    fits, model_dict = hc.fit_heat_capacity(test_data, model_dict)
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
