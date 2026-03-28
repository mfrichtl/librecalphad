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
    fits = sr.fit_segmented_regression(test_data, model_dict)
    assert np.isclose(fits.x[0], 300), (
        "Did not correctly calculate Einstein temperature."
    )
