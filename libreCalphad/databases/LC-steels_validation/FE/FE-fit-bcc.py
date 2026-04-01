from espei.datasets import load_datasets, recursive_glob
import importlib.resources as impresources
import json
from libreCalphad.databases.db_utils import load_database
from libreCalphad.models.segmented_regression import (
    _bent_cable_Cp,
    _debye_Cp,
    _einstein_Cp,
    _holzapfel_debye_Cp,
    _xiong_Cp,
    calc_enthalpy,
    calc_entropy,
    create_espei_custom_refstate_stable,
    fit_segmented_regression,
    get_segmented_regression_Cp,
    upsert_custom_refstate_json,
)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pycalphad import Workspace, as_property, calculate, variables as v
from pycalphad.mapping import BinaryStrategy, plot_binary
from pycalphad.property_framework.metaproperties import IsolatedPhase
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
# magnetic model constants
beta_Fe = 2.22  # magnetic moment per atom Fe
struct_fact_bcc = 0.37
Tc_Fe = 1043  # K, Curie temperature for Fe
theta_Fe = 309  # K, Einstein temperature from Chen & Sundman
model_dict = {
    "einstein": {"theta": [300, "fit"]},
    "xiong": {"beta": [2.22, "fit"], "p": [0.37, "fit"], "Tc": [1043, "fit"]},
    "bcm": {
        "beta_1": [0.01, "fit"],
        "beta_2": [0.04, "fit"],
        "tau": [1750, "fit"],
        "gamma": [50, "fit"],
    },
}
min_fits, model_dict = fit_segmented_regression(search_results, model_dict)
theta_Fe = min_fits.x[model_dict["einstein"]["theta"][-1]]
beta_Fe = min_fits.x[model_dict["xiong"]["beta"][-1]]
struct_fact_bcc = min_fits.x[model_dict["xiong"]["p"][-1]]
Tc_Fe = min_fits.x[model_dict["xiong"]["Tc"][-1]]
beta_1 = min_fits.x[model_dict["bcm"]["beta_1"][-1]]
beta_2 = min_fits.x[model_dict["bcm"]["beta_2"][-1]]
tau = min_fits.x[model_dict["bcm"]["tau"][-1]]
gamma = min_fits.x[model_dict["bcm"]["gamma"][-1]]

# H298 = calc_enthalpy(298.15, min_fits, xiong_params=xiong_params)
# S298 = calc_entropy(298.15, min_fits, xiong_params=xiong_params)
# initialize a dataframe for the model
df_model = pd.DataFrame()
df_model["temperature"] = np.linspace(1, 2000, num=1000)

print("Optimized model parameters:")
print(f"Theta_E={theta_Fe:4f}")
print("Xiong model parameters:")
print(f"beta={beta_Fe}")
print(f"structure factor={struct_fact_bcc}")
print(f"Tc={Tc_Fe}")
print("Bent cable model")
print(f"beta_1={beta_1:3f}")
print(f"beta_2={beta_2:3f}")
print(f"tau={tau:3f}")
print(f"gamma={gamma:3f}")
print(f"RSE: {min_fits.fun:.4f} J/mol/K")
# print(f"H298: {H298}")
# print(f"S298: {S298}")

# Save it to libreCalphad's custom refstate json file

refstate_file = impresources.files("refstate") / "LCRefstates.json"

# Add the fitted refstate results
# Not passing the phase as a kwarg makes libreCalphad assume this is a stable phase Gibbs energy expression
upsert_custom_refstate_json(refstate_file, "FE", min_fits, model_dict=model_dict)

# Need to add the lattice stability for FE also, can directly write it to the json
with open(refstate_file, "r") as f:
    refstate_dict = json.load(f)
refstate_dict["FE-BCC_A2"] = "GHSERFE"
with open(refstate_file, "w") as f:
    json.dump(refstate_dict, f, indent=True)

# and update the SER
ser_file = impresources.files("refstate") / "LCSERparams.json"
with open(ser_file, "r") as f:
    ser_dict = json.load(f)
ser_dict["FE"] = {"phase": "BCC_A2", "mass": 55.847, "H298": H298, "S298": S298}
with open(ser_file, "w") as f:
    json.dump(ser_dict, f, indent=True)
# prepare arrays for plotting
# df_model["debye_model"] = _holzapfel_debye_Cp(df_model["temperature"], *min_fits.x[:1])
df_model["einstein_model"] = _einstein_Cp(df_model["temperature"], theta_Fe)
df_model["magnetic_model"] = _xiong_Cp(
    df_model["temperature"], beta_Fe, struct_fact_bcc, Tc_Fe
)
df_model["bent_cable_model"] = _bent_cable_Cp(
    df_model["temperature"], beta_1, beta_2, tau, gamma
)
df_model["cumulative_model"] = (
    df_model["einstein_model"]
    + df_model["magnetic_model"]
    + df_model["bent_cable_model"]
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
ax.plot(df_model["temperature"], df_model["bent_cable_model"], label="Bent Cable")
ax.plot(df_model["temperature"], df_model["cumulative_model"], label="Cumulative")
ax.legend()

ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Heat Capacity (J/mol-K-atom)")
fig.tight_layout()
fig.savefig("./FE-Cp-BCC_A2.png")

# < 300 K
fig, ax = plt.subplots()
for dataset in search_results:
    ax.scatter(
        dataset["conditions"]["T"],
        np.array(dataset["values"]).squeeze(),
        label=dataset["reference"],
    )

ax.plot(df_model["temperature"], df_model["einstein_model"], label="Einstein")
ax.plot(df_model["temperature"], df_model["magnetic_model"], label="Magnetic")
ax.plot(df_model["temperature"], df_model["bent_cable_model"], label="Bent Cable")
ax.plot(df_model["temperature"], df_model["cumulative_model"], label="Cumulative")
ax.legend()

ax.set_xlim((0, 300))
ax.set_xlabel("Temperature (K)")
ax.set_ylim((0, 30))
ax.set_ylabel("Heat Capacity (J/mol-K-atom)")
fig.tight_layout()
fig.savefig("./FE-Cp-BCC_A2-300K.png")

# Close to 0 K
fig, ax = plt.subplots()
for dataset in search_results:
    ax.scatter(
        dataset["conditions"]["T"],
        np.array(dataset["values"]).squeeze(),
        label=dataset["reference"],
    )
ax.plot(df_model["temperature"], df_model["einstein_model"], label="Einstein")
ax.plot(df_model["temperature"], df_model["magnetic_model"], label="Magnetic")
ax.plot(df_model["temperature"], df_model["bent_cable_model"], label="Bent Cable")
ax.plot(df_model["temperature"], df_model["cumulative_model"], label="Cumulative")
ax.legend()

ax.set_xlim((1e0, 1e2))
ax.set_xscale("log")
ax.set_xlabel("Temperature (K)")
ax.set_ylim((1e-3, 1e1))
ax.set_yscale("log")
ax.set_ylabel("Heat Capacity (J/mol-K-atom)")
fig.tight_layout()
fig.savefig("Fe-Cp-BCC_A2-0K.png")
