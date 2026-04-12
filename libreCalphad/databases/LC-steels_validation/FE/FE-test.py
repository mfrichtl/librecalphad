from espei.datasets import load_datasets, recursive_glob
import importlib.resources as impresources
from libreCalphad.databases.db_utils import load_database
import matplotlib.pyplot as plt
import numpy as np
from pycalphad import calculate, equilibrium, variables as v
from tinydb import where
import yaml

dbf = load_database("LC-steels-thermo.tdb")
# dbf = load_database("LC-steels-input.tdb")
param_gen_file = impresources.files("libreCalphad.databases") / "run_param_gen.yaml"
with open(param_gen_file, "r") as f:
    dataset_folder = yaml.safe_load(f)["system"]["datasets"]
datasets = load_datasets(recursive_glob(dataset_folder))
components = ["FE", "VA"]
phases = ["BCC_A2", "FCC_A1", "LIQUID"]
temps = (0.5, 2000, 2)
conditions = {v.T: temps, v.N: 1, v.P: 101325}

# CPM data

fig, ax = plt.subplots(figsize=(8, 6))
eq_res = equilibrium(dbf, components, phases, conditions, output="heat_capacity")
for phase in phases:
    cpm_res = calculate(
        dbf, components, phase, T=temps, P=101325, N=1, output="heat_capacity"
    )
    ax.plot(cpm_res.T, cpm_res.heat_capacity.squeeze(), label=phase)

ax.plot(eq_res.T, eq_res.heat_capacity.squeeze(), label="equilibrium")
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Heat Capacity (J/mol-formula-K)")
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
fig.tight_layout()
fig.savefig("./all_heat_capacity.png")
plt.close()

# Gibbs energies

fig, ax = plt.subplots(figsize=(8, 6))
eq_res = equilibrium(dbf, components, phases, conditions)
for phase in phases:
    cpm_res = calculate(dbf, components, phase, T=temps, P=101325, N=1)
    ax.plot(cpm_res.T, cpm_res.GM.squeeze(), label=phase)

ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Gibbs Energy (J/mol-formula)")
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
fig.tight_layout()
fig.savefig("./all_gibbs.png")
plt.close()

phases = ["BCC_A2", "FCC_A1"]
# DG
query = (
    (where("phases") == phases)
    & (where("components") == components)
    & (where("output") == "DG-BCC_A2-FCC_A1")
)
search_results = datasets.search(query)
inv_query = (
    (where("phases") == phases)
    & (where("components") == components)
    & (where("output") == "DG-FCC_A1-BCC_A2")
)
inv_search_results = datasets.search(inv_query)

fig, ax = plt.subplots()
DG_BCC_A2 = calculate(
    dbf,
    components,
    "BCC_A2",
    T=(0.5, 2000, 2),
    P=101325,
    N=1,
)
DG_FCC_A1 = calculate(
    dbf,
    components,
    "FCC_A1",
    T=(0.5, 2000, 2),
    P=101325,
    N=1,
)

ax.plot(DG_BCC_A2.T, (DG_FCC_A1.GM.squeeze() - DG_BCC_A2.GM.squeeze()))
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
    (where("phases") == phases)
    & (where("components") == components)
    & (where("output") == "DH-BCC_A2-FCC_A1")
)
search_results = datasets.search(query)

inv_query = (
    (where("phases") == phases)
    & (where("components") == components)
    & (where("output") == "DH-FCC_A1-BCC_A2")
)
inv_search_results = datasets.search(inv_query)

fig, ax = plt.subplots()
H_BCC_A2 = calculate(
    dbf, components, "BCC_A2", T=(0.5, 2000, 2), P=101325, N=1, output="enthalpy"
)
H_FCC_A1 = calculate(
    dbf, components, "FCC_A1", T=(0.5, 2000, 2), P=101325, N=1, output="enthalpy"
)

ax.plot(H_BCC_A2.T, (H_FCC_A1.enthalpy.squeeze() - H_BCC_A2.enthalpy.squeeze()))
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
