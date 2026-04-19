from espei.datasets import load_datasets, recursive_glob
import importlib.resources as impresources
from libreCalphad.databases.db_utils import load_database
from libreCalphad.models.segmented_regression import (
    calculate_offset,
    calculate_transition_energies,
)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pycalphad import calculate
from scipy.optimize import curve_fit, minimize
import seaborn as sns
from tinydb import where
import yaml


dbf = load_database("LC-steels-thermo.tdb")
param_gen_file = impresources.files("libreCalphad.databases") / "run_param_gen.yaml"
with open(param_gen_file, "r") as f:
    dataset_folder = yaml.safe_load(f)["system"]["datasets"]
datasets = load_datasets(recursive_glob(dataset_folder))
components = ["FE", "VA"]
phase = ["FCC_A1"]
T_melt = 1811
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
query = (
    (where("phases") == phase)
    & (where("components") == components)
    & (where("output") == "CPM")
)
search_results = datasets.search(query)

fig, ax = plt.subplots()
cpm_res = calculate(
    dbf, components, phase, T=(0.5, 2000, 2), P=101325, N=1, output="heat_capacity"
)
ax.plot(cpm_res.T, cpm_res.heat_capacity.squeeze())
for result in search_results:
    ax.scatter(
        result["conditions"]["T"],
        np.array(result["values"]).squeeze(),
        label=result["reference"],
    )
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Isobaric Heat Capacity (J/mol-formula-K)")
ax.legend()
fig.tight_layout()
plt.savefig(f"./CPM-CALC-FE-{phase[0]}.png")
plt.close()

# Enthalpy plotting

enthalpy_df = pd.DataFrame()
query = (
    (where("phases") == phase)
    & (where("components") == components)
    & (where("output") == "HM")
)
search_results = datasets.search(query)
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
enthalpy_df["HM_error"] = enthalpy_df["HM_meas"] - enthalpy_df["HM_calc"]
hm_fits = calculate_offset(enthalpy_df["HM_calc"], enthalpy_df["HM_meas"])
print(
    f"Enthalpy calculation is {hm_fits[0][0]} J/mol-formula different than measurements."
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
entropy_df["SM_error"] = entropy_df["SM_meas"] - entropy_df["SM_calc"]
sm_fits = calculate_offset(entropy_df["SM_calc"], entropy_df["SM_meas"])
print(
    f"Entropy calculation is {sm_fits[0][0]} J/mol-K-formula different than measurements."
)
fig, ax = plt.subplots()
sns.scatterplot(entropy_df, x="T", y="SM_meas", hue="reference", ax=ax)
ax.plot(calc_res.T, calc_res.SM.squeeze())
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Entropy (J/mol-formula-K)")
ax.legend()
fig.tight_layout()
plt.savefig(f"./SM-CALC-FE-{phase[0]}.png")

parity_array = np.linspace(
    np.min(entropy_df["SM_meas"]) - 10, np.max(entropy_df["SM_meas"]) + 10
)
fig, ax = plt.subplots()
sns.scatterplot(entropy_df, x="SM_meas", y="SM_calc", hue="reference", ax=ax)
ax.plot(parity_array, parity_array)
ax.set_xlabel("Measured Entropy (J/mol-K-formula)")
ax.set_ylabel("Calculated Entropy (J/mol-K-formula)")
ax.legend()
fig.tight_layout()
plt.savefig(f"./SM-parity_plot-FE-{phase[0]}.png")
plt.close()

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
ax.set_ylabel("Gibbs Energy (J/mol-formula)")
ax.legend()
fig.tight_layout()
fig.savefig(f"./GM-CALC-{phase[0]}.png")
