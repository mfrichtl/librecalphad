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
import matplotlib.pyplot as plt
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
T = se.symbols("T")
# magnetic model constants
beta_Fe = 0.7  # magnetic moment per atom Fe
structure_factor = 0.28
Tn_Fe = 67  # K, Curie/Neel temperature for Fe
theta_Fe = 309  # K, Einstein temperature from Chen & Sundman
model_dict = {
    "einstein": {"theta": [300, "fit"]},
    "xiong": {
        "beta": [beta_Fe, "fix"],
        "p": [structure_factor, "fix"],
        "Tn": [Tn_Fe, "fix"],
    },
    "linear": {"alpha": [0.01, "fit"], "T_melt": [1811, "fix"]},
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
theta_Fe = model_dict["einstein"]["theta"][0]
beta_Fe = model_dict["xiong"]["beta"][0]
structure_factor = model_dict["xiong"]["p"][0]
Tn_Fe = model_dict["xiong"]["Tn"][0]
alpha = model_dict["linear"]["alpha"][0]
T_melt = model_dict["linear"]["T_melt"][0]
melt_a = model_dict["melt"]["a"][0]
melt_b = model_dict["melt"]["b"][0]
melt_c = model_dict["melt"]["c"][0]
de0 = model_dict["two-state"]["dE"][0][0]
de1 = model_dict["two-state"]["dE"][0][1]

# H298 = calc_enthalpy(298.15, min_fits, xiong_params=xiong_params)
# S298 = calc_entropy(298.15, min_fits, xiong_params=xiong_params)
# initialize a dataframe for the model
df_model = pd.DataFrame()
df_model["temperature"] = np.linspace(1, 3000, num=1000)

print("Optimized model parameters:")
for model, values in model_dict.items():
    print(f"Model: {model}")
    if model == "two-state":
        out_str = "dE: "
        for i in range(len(values["dE"][0])):
            out_str += f"{values['dE'][0][i] * values['dE'][1][i]} "
        print(out_str)
    else:
        for param_name, param_value in values.items():
            print(f"{param_name}: {param_value[0]:4f}")
print(f"RSE: {min_fits.fun:.4f} J/mol/K")
# print(f"H298: {H298}")
# print(f"S298: {S298}")

# Save it to libreCalphad's custom refstate json file

refstate_file = impresources.files("refstate") / "LCRefstates.json"

# Add the fitted refstate results
# Not passing the phase as a kwarg makes libreCalphad assume this is a stable phase Gibbs energy expression
upsert_custom_refstate_json(
    refstate_file, "FE", min_fits, phase="FCC_A1", model_dict=model_dict
)

# Need to add the lattice stability for FE also, can directly write it to the json
with open(refstate_file, "r") as f:
    refstate_dict = json.load(f)

# and update the SER
ser_file = impresources.files("refstate") / "LCSERparams.json"
# with open(ser_file, "r") as f:
#     ser_dict = json.load(f)
# ser_dict["FE"] = {"phase": "BCC_A2", "mass": 55.847, "H298": H298, "S298": S298}
# with open(ser_file, "w") as f:
#     json.dump(ser_dict, f, indent=True)
# prepare arrays for plotting
# df_model["debye_model"] = _holzapfel_debye_Cp(df_model["temperature"], *min_fits.x[:1])

import espei

df_model["einstein_model"] = _einstein_Cp(df_model["temperature"], theta_Fe)
df_model["magnetic_model"] = _xiong_Cp(
    df_model["temperature"], beta_Fe, structure_factor, Tn_Fe
)
df_model["linear"] = _linear_Cp(df_model["temperature"], alpha, T_melt)
df_model["melt"] = _melt_Cp(df_model["temperature"], T_melt, melt_a, melt_b, melt_c)
df_model["two-state"] = _twostate_Cp(df_model["temperature"], [de0, de1], [T**0, T**1])
df_model["cumulative_model"] = (
    df_model["einstein_model"]
    + df_model["magnetic_model"]
    + df_model["linear"]
    + df_model["melt"]
    + df_model["two-state"]
)


fig, ax = plt.subplots()
for dataset in search_results:
    ax.scatter(
        dataset["conditions"]["T"],
        np.array(dataset["values"]).squeeze(),
        label=dataset["reference"],
    )
ax.plot(df_model["temperature"], df_model["einstein_model"], label="Einstein")
ax.plot(df_model["temperature"], df_model["magnetic_model"], label="Magnetic")
ax.plot(df_model["temperature"], df_model["linear"], label="Linear")
ax.plot(df_model["temperature"], df_model["melt"], label="Melt")
ax.plot(df_model["temperature"], df_model["two-state"], label="Two-state")
ax.plot(df_model["temperature"], df_model["cumulative_model"], label="Cumulative")

ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Heat Capacity (J/mol-K-atom)")
ax.legend()
fig.tight_layout()
fig.savefig("./FE-Cp-FCC_A1.png")

# fig, ax = plt.subplots()
# ax.plot(df_model["temperature"], df_model["two-state"])
# fig.tight_layout()
# fig.savefig("./FE-Cp-twostate-FCC_A1.png")

# Gibbs energies
# df_model["two-state-gibbs"] = _twostate_gibbs(df_model["temperature"], [de0, de1])
#
# fig, ax = plt.subplots()
# ax.plot(df_model["temperature"], df_model["two-state-gibbs"], label="Two-State")
# ax.legend()
# fig.tight_layout()
# fig.savefig("./FE-Gibbs-FCC_A1.png")
