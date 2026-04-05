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
phase = ["BCC_A2", "FCC_A1"]

query = (
    (where("phases") == phase)
    & (where("components") == components)
    & (where("output") == "DH-BCC_A2-FCC_A1")
)
search_results = datasets.search(query)

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
ax.set_xlabel("Temperature (K)")
ax.set_ylabel(r"$\Delta H^{\mathrm{\alpha} \rightarrow \mathrm{\gamma}}$ (J/mol)")
ax.legend()
fig.tight_layout()
plt.savefig(f"./DH-BCC_A2-FCC_A1.png")
plt.close()

# # Enthalpy plotting
#
# query = (
#     (where("phases") == phase)
#     & (where("components") == components)
#     & (where("output") == "HM")
# )
# search_results = datasets.search(query)
#
# fig, ax = plt.subplots()
# calc_res = calculate(
#     dbf, components, phase, T=(0.5, 2000, 2), P=101325, N=1, output="HM"
# )
# H298 = (
#     calculate(dbf, components, phase, T=298.15, P=101325, N=1, output="HM")
#     .HM.squeeze()
#     .values
# )
# ax.plot(calc_res.T, calc_res.HM.squeeze())
# for result in search_results:
#     ax.scatter(
#         result["conditions"]["T"],
#         np.array(result["values"]).squeeze() + H298,
#         label=result["reference"],
#     )
# ax.set_xlabel("Temperature (K)")
# ax.set_ylabel("Enthalpy (J/mol-formula)")
# ax.legend()
# fig.tight_layout()
# plt.savefig(f"./HM-CALC-FE-{phase[0]}.png")
# plt.close()
#
# # Entropy plotting
#
# query = (
#     (where("phases") == phase)
#     & (where("components") == components)
#     & (where("output") == "SM")
# )
# search_results = datasets.search(query)
#
# fig, ax = plt.subplots()
# calc_res = calculate(
#     dbf, components, phase, T=(0.5, 2000, 2), P=101325, N=1, output="SM"
# )
# S298 = (
#     calculate(dbf, components, phase, T=298.15, P=101325, N=1, output="SM")
#     .SM.squeeze()
#     .values
# )
# ax.plot(calc_res.T, calc_res.SM.squeeze())
# for result in search_results:
#     ax.scatter(
#         result["conditions"]["T"],
#         np.array(result["values"]).squeeze() + S298,
#         label=result["reference"],
#     )
# ax.set_xlabel("Temperature (K)")
# ax.set_ylabel("Entropy (J/mol-formula-K)")
# ax.legend()
# fig.tight_layout()
# plt.savefig(f"./SM-CALC-FE-{phase[0]}.png")
