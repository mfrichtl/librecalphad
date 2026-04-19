from espei.datasets import load_datasets, recursive_glob
import importlib.resources as impresources
import json
from libreCalphad.databases.db_utils import load_database
from libreCalphad.models.segmented_regression import (
    _linear_Cp,
    _debye_Cp,
    _einstein_Cp,
    _holzapfel_debye_Cp,
    _xiong_Cp,
    _melt_Cp,
    calc_enthalpy,
    calc_entropy,
    create_espei_custom_refstate_stable,
    fit_segmented_regression,
    get_segmented_regression_Cp,
    upsert_custom_refstate_json,
)
import matplotlib.pyplot as plt
import numpy as np
import os
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
phase = ["BCC_A2"]
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
    "xiong": {"beta": [2.22, "fix"], "p": [0.37, "fix"], "Tc": [1043, "fix"]},
    "linear": {"alpha": [0.01, "fit"], "T_melt": [1811, "fix"]},
    "melt": {
        "T_melt": [1811, "fix"],
        "liquid_Cp": [46, "fix"],
        "a": [-21, "fit"],
        "b": [9e18, "fit"],
        "c": [-1.5e37, "fit"],
        "solid_enthalpy": [-27442.8, "fix"],
        "solid_entropy": [-153.941, "fix"],
    },
    "offset": {"enthalpy": [-1130.419885, "fix"], "entropy": [8.055879, "fix"]},
}
min_fits, model_dict = fit_segmented_regression(search_results, model_dict)
theta_Fe = model_dict["einstein"]["theta"][0]
beta_Fe = model_dict["xiong"]["beta"][0]
struct_fact_bcc = model_dict["xiong"]["p"][0]
Tc_Fe = model_dict["xiong"]["Tc"][0]
# beta_1 = model_dict["bcm"]["beta_1"][0]
# beta_2 = model_dict["bcm"]["beta_2"][0]
# tau = model_dict["bcm"]["tau"][0]
# gamma = model_dict["bcm"]["gamma"][0]
# T_melt = model_dict["bcm"]["T_melt"][0]
alpha = model_dict["linear"]["alpha"][0]
T_melt = model_dict["linear"]["T_melt"][0]
melt_a = model_dict["melt"]["a"][0]
melt_b = model_dict["melt"]["b"][0]
melt_c = model_dict["melt"]["c"][0]

if os.path.exists("./_fitted_params.json"):
    with open("./_fitted_params.json", "r") as f:
        params_json = json.load(f)
else:
    params_json = {}
params_json["FE"].update({phase[0]: model_dict})
with open("./_fitted_params.json", "w") as f:
    json.dump(params_json, f, indent=4)


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
# ser_dict["FE"] = {"phase": "BCC_A2", "mass": 55.847, "H298": H298, "S298": S298}
with open(ser_file, "w") as f:
    json.dump(ser_dict, f, indent=True)
# prepare arrays for plotting
# df_model["debye_model"] = _holzapfel_debye_Cp(df_model["temperature"], *min_fits.x[:1])
df_model["einstein_model"] = _einstein_Cp(df_model["temperature"], theta_Fe)
df_model["magnetic_model"] = _xiong_Cp(
    df_model["temperature"], beta_Fe, struct_fact_bcc, Tc_Fe
)
# df_model["bent_cable_model"] = _bent_cable_Cp(
#     df_model["temperature"],
#     beta_1,
#     beta_2,
#     tau,
#     gamma,
#     T_melt,
# )
df_model["linear"] = _linear_Cp(df_model["temperature"], alpha, T_melt)
df_model["melt"] = _melt_Cp(df_model["temperature"], T_melt, melt_a, melt_b, melt_c)
df_model["cumulative_model"] = (
    df_model["einstein_model"]
    + df_model["magnetic_model"]
    + df_model["linear"]
    + df_model["melt"]
)

fig, ax = plt.subplots(figsize=(8, 6))
for dataset in search_results:
    ax.scatter(
        dataset["conditions"]["T"],
        np.array(dataset["values"]).squeeze(),
        label=dataset["reference"],
    )
ax.plot(df_model["temperature"], df_model["linear"], label="Linear")
ax.plot(df_model["temperature"], df_model["cumulative_model"], label="Cumulative")
ax.plot(df_model["temperature"], df_model["einstein_model"], label="Einstein")
ax.plot(df_model["temperature"], df_model["magnetic_model"], label="Magnetic")
ax.plot(df_model["temperature"], df_model["melt"], label="Melt")
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

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
ax.plot(df_model["temperature"], df_model["linear"], label="Linear")
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
ax.plot(df_model["temperature"], df_model["linear"], label="Linear")
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
