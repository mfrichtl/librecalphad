from espei.datasets import load_datasets, recursive_glob
from importlib import reload
import importlib.resources as impresources
import json
from libreCalphad.databases.db_utils import load_database, upsert_db_param_from_models
from libreCalphad.models.energy import (
    calculate_offset,
    calculate_transition_energies,
    upsert_custom_refstate_json,
)
from libreCalphad.models.heat_capacity import fit_heat_capacity
from libreCalphad.plotting import (
    plot_calculated_gibbs_energies,
    plot_calculated_heat_capacity,
    plot_heat_capacity_from_models,
)
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from pycalphad import calculate, Database, variables as v
import seaborn as sns
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

params_file = "./FE-params.json"
with open(params_file, "r") as f:
    model_dict = json.load(f)[phase[0]]
min_fits, model_dict = fit_heat_capacity(search_results, model_dict, verbose=True)
T_melt = model_dict["melt"]["T_melt"][0]

fig, ax = plot_heat_capacity_from_models(model_dict, datasets, phase, components)
fig.savefig("FE-FCC-CPM_fits.png")
plt.close()

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

# Now let's update the input dbf
input_db_file = impresources.files("libreCalphad.databases") / "LC-steels-input.xml"
input_dbf = Database(input_db_file)
# backup in case something goes wrong, it can delete all the parameters in the file
input_db_bak = impresources.files("libreCalphad.databases") / "LC-steels-input.bak.xml"
input_dbf.to_file(input_db_bak, if_exists="overwrite")

species_dict = {sp.name: sp for sp in input_dbf.species}
constituent_array = ((species_dict["FE"],), (species_dict["VA"],))
input_dbf = upsert_db_param_from_models(
    input_dbf, model_dict, phase[0], constituent_array
)
input_dbf.to_file(input_db_file, if_exists="overwrite")


phase_model_file = impresources.files("libreCalphad.databases") / "phase_models.json"
with open(phase_model_file, "r") as f:
    phase_models = json.load(f)

reload(espei)
dbf = espei.generate_parameters(
    phase_models, datasets, "LCRefState", "linear", dbf=input_dbf
)
db_file = impresources.files("libreCalphad.databases") / "LC-steels-thermo.xml"
dbf.to_file(db_file, if_exists="overwrite")

# now let's calculate some stuffs
H298 = (
    calculate(dbf, components, "BCC_A2", T=298.15, P=101325, N=1, output="HM")
    .HM.squeeze()
    .values
)
S298 = (
    calculate(dbf, components, "BCC_A2", T=298.15, P=101325, N=1, output="SM")
    .SM.squeeze()
    .values
)

melt_calc_HM = calculate(
    dbf, components, phase, T=T_melt - 0.01, P=101325, N=1, output="HM"
)
melt_calc_SM = calculate(
    dbf, components, phase, T=T_melt - 0.01, P=101325, N=1, output="SM"
)
print(f"Melt HM={melt_calc_HM.HM.squeeze().values}")
print(f"Melt SM={melt_calc_SM.SM.squeeze().values}")

melt_HM_min = calculate_transition_energies(dbf, components, phase, T_melt, "HM")
print(f"Melt enthalpy offset required = {melt_HM_min.x[0]} J/mol-formula")
melt_SM_min = calculate_transition_energies(dbf, components, phase, T_melt, "SM")
print(f"Melt entropy offset required = {-melt_SM_min.x[0]} J/mol-K-formula")

# Plot CPM
query = (
    (where("phases") == phase)
    & (where("components") == components)
    & (where("output") == "CPM")
)
search_results = datasets.search(query)

search_results = datasets.search(query)
phases = ["BCC_A2", "FCC_A1", "GAS", "LIQUID"]
fig, ax = plt.subplots(figsize=(8, 6))
plot_calculated_heat_capacity(
    dbf, components, phases, {v.T: (0, 2500, 2), v.P: 101325, v.N: 1}, fig=fig, ax=ax
)
fig.savefig("CPM-CALC-all_phases.png")
plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
plot_calculated_heat_capacity(
    dbf,
    components,
    phase,
    {v.T: (0, 2500, 2), v.P: 101325, v.N: 1},
    datasets=search_results,
    fig=fig,
    ax=ax,
)
plt.savefig(f"./CPM-CALC-FE-{phase[0]}.png")
plt.close()

# Enthalpy plotting

query = (
    (where("phases") == phase)
    & (where("components") == components)
    & (where("output") == "HM")
)
search_results = datasets.search(query)


enthalpy_df = pd.DataFrame()
for result in search_results:
    res_dict = {
        "T": result["conditions"]["T"],
        "HM_meas": np.array(result["values"]).squeeze(),
        "reference": result["reference"],
    }
    enthalpy_calc_arr = np.array([])
    for temp in result["conditions"]["T"]:
        enthalpy_calc = calculate(
            dbf, components, phase, T=temp, P=101325, N=1, output="HM"
        )
        enthalpy_calc_arr = np.append(
            enthalpy_calc_arr, enthalpy_calc.HM.squeeze().values
        )
    res_dict["HM_calc"] = enthalpy_calc_arr

    if enthalpy_df.empty:
        enthalpy_df = pd.DataFrame(res_dict)
    else:
        enthalpy_df = pd.concat(
            [enthalpy_df, pd.DataFrame(res_dict)], ignore_index=True
        )

query = (
    (where("phases") == phase)
    & (where("components") == components)
    & (where("output") == "DH")
)
search_results = datasets.search(query)

for result in search_results:
    res_dict = {
        "T": result["conditions"]["T"],
        "HM_meas": np.array(result["values"]).squeeze() - H298,
        "reference": result["reference"],
    }
    enthalpy_calc_arr = np.array([])
    for temp in result["conditions"]["T"]:
        enthalpy_calc = calculate(
            dbf, components, phase, T=temp, P=101325, N=1, output="HM"
        )
        enthalpy_calc_arr = np.append(
            enthalpy_calc_arr, enthalpy_calc.HM.squeeze().values - H298
        )
    res_dict["HM_calc"] = enthalpy_calc_arr

    if enthalpy_df.empty:
        enthalpy_df = pd.DataFrame(res_dict)
    else:
        enthalpy_df = pd.concat(
            [enthalpy_df, pd.DataFrame(res_dict)], ignore_index=True
        )

enthalpy_df["HM_error"] = enthalpy_df["HM_meas"] - enthalpy_df["HM_calc"]
enthalpy_df = enthalpy_df.sort_values("T")

hm_fits = calculate_offset(enthalpy_df["T"], enthalpy_df["HM_error"], order=0)
print(
    f"Enthalpy calculation is {hm_fits[0]} J/mol-formula different than measurements."
)


fig, ax = plt.subplots()
calc_res = calculate(
    dbf,
    components,
    phase,
    T=(0.5, np.max(enthalpy_df["T"]) + 200, 2),
    P=101325,
    N=1,
    output="HM",
)
ax.plot(calc_res.T, calc_res.HM.squeeze())
sns.scatterplot(enthalpy_df, x="T", y="HM_meas", hue="reference", ax=ax)
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Enthalpy (J/mol-formula)")
ax.legend()
fig.tight_layout()
plt.savefig(f"./HM-CALC-FE-{phase[0]}.png")
plt.close()

parity_array = np.linspace(
    np.min(enthalpy_df["HM_meas"]) - 100, np.max(enthalpy_df["HM_meas"]) + 100
)
fig, ax = plt.subplots()
sns.scatterplot(enthalpy_df, x="HM_meas", y="HM_calc", hue="reference", ax=ax)
ax.plot(parity_array, parity_array)
ax.set_xlabel("Measured Enthalpy (J/mol-formula)")
ax.set_ylabel("Calculated Enthalpy (J/mol-formula)")
ax.legend()
fig.tight_layout()
plt.savefig(f"./HM-parity_plot-FE-{phase[0]}.png")
plt.close()

fig, ax = plt.subplots()
sns.scatterplot(enthalpy_df, x="T", y="HM_error", hue="reference", ax=ax)
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Enthalpy Error (J/mol-formula)")
ax.legend()
fig.tight_layout()
plt.savefig(f"./HM-error_plot-FE-{phase[0]}.png")
plt.close()

# Entropy plotting

entropy_df = pd.DataFrame()
query = (
    (where("phases") == phase)
    & (where("components") == components)
    & (where("output") == "SM")
)
search_results = datasets.search(query)

fig, ax = plt.subplots()
calc_res = calculate(
    dbf, components, phase, T=(0.5, 2000, 2), P=101325, N=1, output="SM"
)
ax.plot(calc_res.T, calc_res.SM.squeeze())
for result in search_results:
    res_dict = {
        "T": result["conditions"]["T"],
        "SM_meas": np.array(result["values"]).squeeze(),
        "reference": result["reference"],
    }
    entropy_calc_arr = np.array([])
    for temp in result["conditions"]["T"]:
        entropy_calc = calculate(
            dbf, components, phase, T=temp, P=101325, N=1, output="SM"
        )
        entropy_calc_arr = np.append(entropy_calc_arr, entropy_calc.SM.squeeze().values)
    res_dict["SM_calc"] = entropy_calc_arr

    if entropy_df.empty:
        entropy_df = pd.DataFrame(res_dict)
    else:
        entropy_df = pd.concat([entropy_df, pd.DataFrame(res_dict)], ignore_index=True)

query = (
    (where("phases") == phase)
    & (where("components") == components)
    & (where("output") == "DS")
)
search_results = datasets.search(query)

for result in search_results:
    res_dict = {
        "T": result["conditions"]["T"],
        "SM_meas": np.array(result["values"]).squeeze() + S298,
        "reference": result["reference"],
    }
    entropy_calc_arr = np.array([])
    for temp in result["conditions"]["T"]:
        entropy_calc = calculate(
            dbf, components, phase, T=temp, P=101325, N=1, output="SM"
        )
        entropy_calc_arr = np.append(entropy_calc_arr, entropy_calc.SM.squeeze().values)
    res_dict["SM_calc"] = entropy_calc_arr

    if entropy_df.empty:
        entropy_df = pd.DataFrame(res_dict)
    else:
        entropy_df = pd.concat([entropy_df, pd.DataFrame(res_dict)], ignore_index=True)


entropy_df["SM_error"] = entropy_df["SM_meas"] - entropy_df["SM_calc"]
entropy_df = entropy_df.sort_values("T")

sm_fits = calculate_offset(entropy_df["T"], entropy_df["SM_error"], order=0)
print(
    f"Entropy calculation is {sm_fits[0]} J/mol-K-formula different than measurements."
)

fig, ax = plt.subplots()
sns.scatterplot(entropy_df, x="T", y="SM_meas", hue="reference", ax=ax)
ax.plot(calc_res.T, calc_res.SM.squeeze())
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Entropy (J/mol-formula-K)")
ax.legend()
fig.tight_layout()
plt.savefig(f"./SM-CALC-FE-{phase[0]}.png")

fig, ax = plt.subplots()
sns.scatterplot(entropy_df, x="T", y="SM_error", hue="reference", ax=ax)
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Enthalpy Error (J/mol-K-formula)")
ax.legend()
fig.tight_layout()
plt.savefig(f"./SM-error_plot-FE-{phase[0]}.png")
plt.close()

# Gibbs plotting
query = (
    (where("phases") == phase)
    & (where("components") == components)
    & (where("output") == "GM")
)
search_results = datasets.search(query)
calc_res = calculate(
    dbf, components, phase, T=(0.5, 2000, 2), P=101325, N=1, output="GM"
)

fig, ax = plt.subplots()
for result in search_results:
    ax.scatter(
        result["conditions"]["T"],
        np.array(result["values"]).squeeze(),
        label=result["reference"],
    )
ax.plot(calc_res.T, calc_res.GM.squeeze())
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Gibbs energy (J/mol-formula)")
ax.legend()
fig.tight_layout()
fig.savefig(f"./GM-CALC-{phase[0]}.png")


# BCC_A2 - FCC_A1
dg_phases = ["BCC_A2", "FCC_A1"]
# DG
query = (
    (where("phases") == dg_phases)
    & (where("components") == components)
    & (where("output") == "DG-BCC_A2-FCC_A1")
)
search_results = datasets.search(query)
inv_query = (
    (where("phases") == dg_phases)
    & (where("components") == components)
    & (where("output") == "DG-FCC_A1-BCC_A2")
)
inv_search_results = datasets.search(inv_query)

fig, ax = plt.subplots()
bcc_calc = calculate(
    dbf,
    components,
    "BCC_A2",
    T=(0.5, 2000, 2),
    P=101325,
    N=1,
)
fcc_calc = calculate(
    dbf,
    components,
    "FCC_A1",
    T=(0.5, 2000, 2),
    P=101325,
    N=1,
)

ax.plot(bcc_calc.T, (fcc_calc.GM.squeeze() - bcc_calc.GM.squeeze()))
for result in search_results:
    ax.scatter(
        result["conditions"]["T"],
        np.array(result["values"]).squeeze(),
        label=result["reference"],
    )
for result in inv_search_results:
    ax.scatter(
        result["conditions"]["T"],
        -np.array(result["values"]).squeeze(),
        label=result["reference"],
    )
ax.set_xlabel("Temperature (K)")
ax.set_ylabel(r"$\Delta G^{\mathrm{\alpha} \rightarrow \mathrm{\gamma}}$ (J/mol)")
ax.legend()
fig.tight_layout()
plt.savefig(f"./DG-BCC_A2-FCC_A1.png")
plt.close()

# DH data
query = (
    (where("phases") == dg_phases)
    & (where("components") == components)
    & (where("output") == "DH-BCC_A2-FCC_A1")
)
search_results = datasets.search(query)

inv_query = (
    (where("phases") == dg_phases)
    & (where("components") == components)
    & (where("output") == "DH-FCC_A1-BCC_A2")
)
inv_search_results = datasets.search(inv_query)

fig, ax = plt.subplots()
bcc_calc = calculate(
    dbf, components, "BCC_A2", T=(0.5, 2000, 2), P=101325, N=1, output="enthalpy"
)
fcc_calc = calculate(
    dbf, components, "FCC_A1", T=(0.5, 2000, 2), P=101325, N=1, output="enthalpy"
)

ax.plot(bcc_calc.T, (fcc_calc.enthalpy.squeeze() - bcc_calc.enthalpy.squeeze()))
for result in search_results:
    ax.scatter(
        result["conditions"]["T"],
        np.array(result["values"]).squeeze(),
        label=result["reference"],
    )
for result in inv_search_results:
    ax.scatter(
        result["conditions"]["T"],
        -np.array(result["values"]).squeeze(),
        label=result["reference"],
    )

ax.set_xlabel("Temperature (K)")
ax.set_ylabel(r"$\Delta H^{\mathrm{\alpha} \rightarrow \mathrm{\gamma}}$ (J/mol)")
ax.legend()
fig.tight_layout()
plt.savefig(f"./DH-BCC_A2-FCC_A1.png")
plt.close()

fig, ax = plt.subplots(figsize=(6, 4))
conditions = {v.T: (0.5, 4000, 5), v.P: 101325, v.N: 1}
plot_calculated_gibbs_energies(
    dbf, ["FE", "VA"], phases, conditions, print_transition_temps=True, fig=fig, ax=ax
)
fig.savefig("GM-CALC-all_phases.png")
plt.close()
