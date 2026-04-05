import espei
from importlib import resources as impresources
import json
import libreCalphad.models.segmented_regression as sr
import numpy as np
import pandas as pd


def test_fit_einstein_Cp():
    data_file = impresources.files("libreCalphad.tests") / "test_einstein_data.json"
    with open(data_file, "r") as f:
        test_data = [json.load(f)]
    model_dict = {"einstein": {"theta": [250, "fit"]}}
    fits, model_dict = sr.fit_segmented_regression(test_data, model_dict)
    assert np.isclose(model_dict["einstein"]["theta"][0], 300)


def test_fix_einstein_Cp():
    data_file = impresources.files("libreCalphad.tests") / "test_einstein_data.json"
    with open(data_file, "r") as f:
        test_data = [json.load(f)]
    model_dict = {"einstein": {"theta": [300, "fix"]}}
    fits, model_dict = sr.fit_segmented_regression(test_data, model_dict)
    assert np.isclose(fits, 0, atol=1e-5)


def test_fit_holzapfel_Cp():
    data_file = impresources.files("libreCalphad.tests") / "test_holzapfel_data.json"
    with open(data_file, "r") as f:
        test_data = [json.load(f)]
    model_dict = {"holzapfel": {"theta": [250, "fit"]}}
    fits, model_dict = sr.fit_segmented_regression(test_data, model_dict)
    assert np.isclose(model_dict["holzapfel"]["theta"][0], 300)


def test_fix_holzapfel_Cp():
    data_file = impresources.files("libreCalphad.tests") / "test_holzapfel_data.json"
    with open(data_file, "r") as f:
        test_data = [json.load(f)]
    model_dict = {"holzapfel": {"theta": [300, "fix"]}}
    fits, model_dict = sr.fit_segmented_regression(test_data, model_dict)
    assert np.isclose(fits, 0, atol=1e-5)


def test_fit_xiong_Cp():
    data_file = impresources.files("libreCalphad.tests") / "test_xiong_data.json"
    with open(data_file, "r") as f:
        test_data = [json.load(f)]
    model_dict = {
        "xiong": {"beta": [1.8, "fit"], "p": [0.20, "fit"], "Tc": [1300, "fit"]}
    }
    fits, model_dict = sr.fit_segmented_regression(test_data, model_dict)
    assert np.isclose(model_dict["xiong"]["beta"][0], 2, rtol=0.05)
    assert np.isclose(model_dict["xiong"]["p"][0], 0.25, rtol=1e-2)
    assert np.isclose(model_dict["xiong"]["Tc"][0], 1000, rtol=1)


def test_fix_xiong_Cp():
    data_file = impresources.files("libreCalphad.tests") / "test_xiong_data.json"
    with open(data_file, "r") as f:
        test_data = [json.load(f)]
    model_dict = {
        "xiong": {"beta": [2, "fix"], "p": [0.25, "fix"], "Tc": [1000, "fix"]}
    }
    fits, model_dict = sr.fit_segmented_regression(test_data, model_dict)
    assert np.isclose(fits, 0)


def test_fit_bcm_Cp():
    data_file = impresources.files("libreCalphad.tests") / "test_bcm_data.json"
    with open(data_file, "r") as f:
        test_data = [json.load(f)]
    model_dict = {
        "bcm": {
            "beta_1": [0.1, "fit"],
            "beta_2": [0.2, "fit"],
            "tau": [450, "fit"],
            "gamma": [20, "fit"],
        }
    }

    fits, model_dict = sr.fit_segmented_regression(test_data, model_dict)
    assert np.isclose(model_dict["bcm"]["beta_1"][0], 0.025)
    assert np.isclose(model_dict["bcm"]["beta_2"][0], 0.3)
    assert np.isclose(model_dict["bcm"]["tau"][0], 750)
    assert np.isclose(model_dict["bcm"]["gamma"][0], 50)
    assert np.isclose(fits.fun, 0, atol=1e-5)


def test_fix_bcm_Cp():
    data_file = impresources.files("libreCalphad.tests") / "test_bcm_data.json"
    with open(data_file, "r") as f:
        test_data = [json.load(f)]
    model_dict = {
        "bcm": {
            "beta_1": [0.025, "fix"],
            "beta_2": [0.3, "fix"],
            "tau": [750, "fix"],
            "gamma": [50, "fix"],
        }
    }

    fits, model_dict = sr.fit_segmented_regression(test_data, model_dict)
    assert np.isclose(fits, 0)
