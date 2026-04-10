from espei.datasets import load_datasets, recursive_glob
import importlib.resources as impresources
from libreCalphad.databases.db_utils import load_database
import matplotlib.pyplot as plt
import numpy as np
from pycalphad import calculate
from tinydb import where
import yaml

dbf = load_database("LC-steels-thermo.tdb")
# dbf = load_database("LC-steels-input.tdb")
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

query = (
    (where("phases") == phase)
    & (where("components") == components)
    & (where("output") == "DH")
)
search_results = datasets.search(query)

fig, ax = plt.subplots()
calc_res = calculate(
    dbf, components, phase, T=(0.5, 2000, 2), P=101325, N=1, output="HM"
)
H298 = (
    calculate(dbf, components, "BCC_A2", T=298.15, P=101325, N=1, output="HM")
    .HM.squeeze()
    .values
)
ax.plot(calc_res.T, calc_res.HM.squeeze() - H298)
for result in search_results:
    ax.scatter(
        result["conditions"]["T"],
        np.array(result["values"]).squeeze(),
        label=result["reference"],
    )
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Enthalpy (J/mol-formula)")
ax.legend()
fig.tight_layout()
plt.savefig(f"./HM-CALC-FE-{phase[0]}.png")
plt.close()

# Entropy plotting

query = (
    (where("phases") == phase)
    & (where("components") == components)
    & (where("output") == "DS")
)
search_results = datasets.search(query)

fig, ax = plt.subplots()
calc_res = calculate(
    dbf, components, phase, T=(0.5, 2000, 2), P=101325, N=1, output="SM"
)
S298 = (
    calculate(dbf, components, "BCC_A2", T=298.15, P=101325, N=1, output="SM")
    .SM.squeeze()
    .values
)
ax.plot(calc_res.T, calc_res.SM.squeeze() - S298)
for result in search_results:
    ax.scatter(
        result["conditions"]["T"],
        np.array(result["values"]).squeeze(),
        label=result["reference"],
    )
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Entropy (J/mol-formula-K)")
ax.legend()
fig.tight_layout()
plt.savefig(f"./SM-CALC-FE-{phase[0]}.png")

# Gibbs plotting
calc_res = calculate(
    dbf, components, phase, T=(0.5, 2000, 2), P=101325, N=1, output="GM"
)

fig, ax = plt.subplots()
ax.plot(calc_res.T, calc_res.GM.squeeze())
fig.tight_layout()
fig.savefig(f"./GM-CALC-{phase[0]}.png")
