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

R = 8.314472
# magnetic model constants
beta_Fe = 0.7  # magnetic moment per atom Fe
structure_factor = 0.28
Tc_Fe = 67  # K, Curie/Neel temperature for Fe
theta_Fe = 309  # K, Einstein temperature from Chen & Sundman
xiong_params = {"beta": beta_Fe, "p": structure_factor, "Tc": Tc_Fe}
sr_df, min_fits = fit_segmented_regression(
    dataset_folder,
    ["FE", "VA"],
    phase,
    xiong_params=xiong_params,
    models={
        "einstein",
    },
)
H298 = calc_enthalpy(298.15, min_fits, xiong_params=xiong_params)
S298 = calc_entropy(298.15, min_fits, xiong_params=xiong_params)
# initialize a dataframe for the model
df_model = pd.DataFrame()
df_model["temperature"] = np.linspace(1, np.max(sr_df["temperature"]), num=1000)

print("Optimized model parameters:")
print(f"Theta_(D/E)={min_fits.x[0]:4f}")
print(f"beta_1={min_fits.x[1]:3f}")
print(f"beta_2={min_fits.x[2]:3f}")
print(f"tau={min_fits.x[3]:3f}")
print(f"gamma={min_fits.x[4]:3f}")
print(f"RSE: {min_fits.fun:.4f} J/mol/K")
print(f"H298: {H298}")
print(f"S298: {S298}")

# Save it to libreCalphad's custom refstate json file

refstate_file = impresources.files("refstate") / "LCRefstates.json"

# Add the fitted refstate results
# Not passing the phase as a kwarg makes libreCalphad assume this is a stable phase Gibbs energy expression
upsert_custom_refstate_json(
    refstate_file, "FE", min_fits, phase="FCC_A1", xiong_params=xiong_params
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
df_model["einstein_model"] = _einstein_Cp(df_model["temperature"], theta_Fe)
df_model["magnetic_model"] = _xiong_Cp(
    df_model["temperature"], beta_Fe, structure_factor, Tc_Fe
)
df_model["bent_cable_model"] = _bent_cable_Cp(df_model["temperature"], *min_fits.x[1:])
df_model["cumulative_model"] = get_segmented_regression_Cp(
    df_model["temperature"], min_fits, xiong_params=xiong_params, use_einstein=True
)

fig, ax = plt.subplots()
for reference in sr_df["reference"].unique():
    sr_sub = sr_df.query("reference == @reference")
    ax.scatter(sr_sub["temperature"], sr_sub["Cp"], label=reference)
# ax.plot(df_model["temperature"], df_model["debye_model"], label="Debye")
ax.plot(df_model["temperature"], df_model["einstein_model"], label="Einstein")
ax.plot(df_model["temperature"], df_model["magnetic_model"], label="Magnetic")
ax.plot(df_model["temperature"], df_model["bent_cable_model"], label="Bent Cable")
# ax.plot(
#     df_model["temperature"],
#     df_model["debye_model"] + df_model["bent_cable_model"],
#     label="Debye+BCM",
# )
ax.plot(df_model["temperature"], df_model["cumulative_model"], label="Cumulative")
ax.legend()

ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Heat Capacity (J/mol-K-atom)")
fig.tight_layout()
fig.savefig("./FE-Cp-FCC_A1.png")

# < 300 K
fig, ax = plt.subplots()
for reference in sr_df["reference"].unique():
    sr_sub = sr_df.query("reference == @reference")
    ax.scatter(sr_sub["temperature"], sr_sub["Cp"], label=reference)
# ax.plot(df_model["temperature"], df_model["debye_model"], label="Debye")
ax.plot(df_model["temperature"], df_model["einstein_model"], label="Einstein")
ax.plot(df_model["temperature"], df_model["magnetic_model"], label="Magnetic")
ax.plot(df_model["temperature"], df_model["bent_cable_model"], label="Bent Cable")
# ax.plot(
#     df_model["temperature"],
#     df_model["debye_model"] + df_model["bent_cable_model"],
#     label="Debye+BCM",
# )
ax.plot(df_model["temperature"], df_model["cumulative_model"], label="Cumulative")
ax.legend()

ax.set_xlim((0, 300))
ax.set_xlabel("Temperature (K)")
ax.set_ylim((0, 30))
ax.set_ylabel("Heat Capacity (J/mol-K-atom)")
fig.tight_layout()
fig.savefig("./FE-Cp-FCC_A1-300K.png")

# Close to 0 K
fig, ax = plt.subplots()
for reference in sr_df["reference"].unique():
    sr_sub = sr_df.query("reference == @reference")
    ax.scatter(sr_sub["temperature"], sr_sub["Cp"], label=reference)
# ax.plot(df_model["temperature"], df_model["debye_model"], label="Debye")
ax.plot(df_model["temperature"], df_model["einstein_model"], label="Einstein")
ax.plot(df_model["temperature"], df_model["magnetic_model"], label="Magnetic")
ax.plot(df_model["temperature"], df_model["bent_cable_model"], label="Bent Cable")
# ax.plot(
#     df_model["temperature"],
#     df_model["debye_model"] + df_model["bent_cable_model"],
#     label="Debye+BCM",
# )
ax.plot(df_model["temperature"], df_model["cumulative_model"], label="Cumulative")
ax.legend()

ax.set_xlim((1e0, 1e2))
ax.set_xscale("log")
ax.set_xlabel("Temperature (K)")
ax.set_ylim((1e-3, 1e1))
ax.set_yscale("log")
ax.set_ylabel("Heat Capacity (J/mol-K-atom)")
fig.tight_layout()
fig.savefig("Fe-Cp-FCC_A1-0K.png")
