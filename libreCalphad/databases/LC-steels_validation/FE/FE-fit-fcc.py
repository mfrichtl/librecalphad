from espei.datasets import load_datasets, recursive_glob
from importlib import reload
import importlib.resources as impresources
import json
from libreCalphad.databases.db_utils import load_database
from libreCalphad.models.energy import upsert_custom_refstate_json
from libreCalphad.models.heat_capacity import (
    _bent_cable_Cp,
    _debye_Cp,
    _einstein_Cp,
    _holzapfel_debye_Cp,
    _xiong_Cp,
    _twostate_Cp,
    _linear_Cp,
    _melt_Cp,
    fit_heat_capacity,
)
from libreCalphad.models.plotting import plot_heat_capacity_from_models
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from pycalphad import Workspace, as_property, calculate, variables as v
from pycalphad.mapping import BinaryStrategy, plot_binary
from pycalphad.property_framework.metaproperties import IsolatedPhase
from scipy.optimize import curve_fit
import symengine as se
from tinydb import where
import yaml


param_gen_file = impresources.files("libreCalphad.databases") / "run_param_gen.yaml"
with open(param_gen_file, "r") as f:
    dataset_folder = yaml.safe_load(f)["system"]["datasets"]
datasets = load_datasets(recursive_glob(dataset_folder))
components = ["FE", "VA"]
phase = ["FCC_A1"]

query = (
    (where("phases") == phase)
    & (where("components") == components)
    & (where("output") == "CPM")
)
search_results = datasets.search(query)

R = 8.314472
a, T = se.symbols("a T")
# magnetic model constants
beta_Fe = 0.7  # magnetic moment per atom Fe
structure_factor = 0.28
Tn_Fe = 67  # K, Curie/Neel temperature for Fe
theta_Fe = 309  # K, Einstein temperature from Chen & Sundman
T_melt = 1811
model_dict = {
    "einstein": {"theta": [300, "fit"]},
    "xiong": {
        "beta": [beta_Fe, "fix"],
        "p": [structure_factor, "fix"],
        "Tn": [Tn_Fe, "fix"],
    },
    "symbolic": {
        "expression": a * T,
        "param_bounds": {"a": (-np.inf, np.inf)},
        "temp_bounds": (0, T_melt),
    },
    "melt": {
        "T_melt": [1811, "fix"],
        "liquid_Cp": [46, "fix"],
        "a": [-21, "fit"],
        "b": [9e18, "fit"],
        "c": [-1.5e37, "fit"],
        "solid_enthalpy": [-27190.7, "fix"],
        "solid_entropy": [-144.759, "fix"],
    },
    "offset": {
        "enthalpy": [-2905.15, "fix"],
        "entropy": [8.529, "fix"],
    },
    "two-state": {"dE": [[9.02352375e3, -2.4952226], [T**0, T**1], "fit"]},
}
min_fits, model_dict = fit_heat_capacity(search_results, model_dict)

fig, ax = plot_heat_capacity_from_models(model_dict, datasets, phase, components)
fig.savefig("FE-FCC-CPM_fits.png")
plt.close()
# initialize a dataframe for the model

refstate_file = impresources.files("refstate") / "LCRefstates.json"

# Add the fitted refstate results
# Not passing the phase as a kwarg makes libreCalphad assume this is a stable phase Gibbs energy expression
upsert_custom_refstate_json(
    refstate_file, "FE", min_fits, phase="FCC_A1", model_dict=model_dict
)
# Need to add the lattice stability for FE also, can directly write it to the json
with open(refstate_file, "r") as f:
    refstate_dict = json.load(f)

if os.path.exists("./_fitted_params.json"):
    with open("./_fitted_params.json", "r") as f:
        params_json = json.load(f)
else:
    params_json = {"FE": {}}
params_json["FE"].update({phase[0]: model_dict})
with open("./_fitted_params.json", "w") as f:
    json.dump(params_json, f, indent=4)

import espei


# Gibbs energies
# df_model["two-state-gibbs"] = _twostate_gibbs(df_model["temperature"], [de0, de1])
#
# fig, ax = plt.subplots()
# ax.plot(df_model["temperature"], df_model["two-state-gibbs"], label="Two-State")
# ax.legend()
# fig.tight_layout()
# fig.savefig("./FE-Gibbs-FCC_A1.png")
