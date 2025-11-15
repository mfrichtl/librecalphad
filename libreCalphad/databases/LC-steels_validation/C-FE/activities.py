from datetime import date
from espei.datasets import load_datasets, recursive_glob
from espei.plot import dataplot, plot_endmember, plot_interaction
from libreCalphad.databases.db_utils import load_database
from libreCalphad.models.thermodynamics import calculate_energy_from_activity
import matplotlib.pyplot as plt
import numpy as np
from pycalphad import Workspace, variables as v
from pycalphad.mapping import BinaryStrategy, plot_binary
from pycalphad.property_framework.metaproperties import IsolatedPhase
from scipy.optimize import curve_fit
import seaborn as sns
import yaml


def _lin_fit(x, m, b):
    return m * x + b


db = load_database("LC-steels-thermo.tdb")
disabled_phases = ["CEMENTITE_D011"]  # stable first
phases = [phase for phase in list(db.phases.keys()) if phase not in disabled_phases]
with open("../../run_param_gen.yaml", "r") as f:
    dataset_folder = yaml.safe_load(f)["system"]["datasets"]
# datasets = load_datasets(recursive_glob(dataset_folder))

activity_df = calculate_energy_from_activity(
    dataset_folder, ["C", "FE", "VA"], ["BCC_A2"]
)
fig, ax = plt.subplots()
for temperature in activity_df["temperature"].unique():
    temp_df = activity_df.query("temperature == @temperature")
    ax.scatter(temp_df["concentration"], temp_df["activity"], label=temperature)
ax.legend()
plt.savefig("./BCC_A2_activity_vs_concentration.png")

fig, ax = plt.subplots()
fig_HM, ax_HM = plt.subplots()
activity_df["inv_temp"] = activity_df["temperature"] ** -1
activity_df["ln_coefficient"] = np.log(activity_df["activity_coefficient"])
conc_steps = np.arange(0, np.max(activity_df["concentration"]), step=0.0002)
x_vals = np.linspace(0.0007, np.max(activity_df["inv_temp"]))
# for step_index in np.arange(len(conc_steps))[1:]:
#     avg_conc = (conc_steps[step_index] - conc_steps[step_index - 1]) / 2 + conc_steps[
#         step_index - 1
#     ]
#     step_df = activity_df.query(
#         f"concentration >= {conc_steps[step_index - 1]} & concentration < {conc_steps[step_index]}"
#     )
#     if step_df.empty:
#         continue
#     fits = curve_fit(
#         _lin_fit, xdata=step_df["inv_temp"], ydata=step_df["ln_coefficient"]
#     )
#     print(f"X_C={avg_conc}, HM_MIX={fits[0][0]}")
#     ax.scatter(
#         step_df["inv_temp"] * 1000,
#         step_df["ln_coefficient"],
#         label=conc_steps[step_index],
#     )
#     ax.plot(
#         x_vals * 1000,
#         [_lin_fit(x, *fits[0]) for x in x_vals],
#         label=conc_steps[step_index],
#     )
#     ax_HM.scatter(avg_conc, fits[0][0])
sns.scatterplot(
    data=activity_df, x="inv_temp", y="ln_coefficient", hue="concentration", ax=ax
)
fits = curve_fit(
    _lin_fit, xdata=activity_df["inv_temp"], ydata=activity_df["ln_coefficient"]
)
ax.plot(x_vals, [_lin_fit(x, *fits[0]) for x in x_vals])
ax.legend()
fig.tight_layout()
fig.savefig("./BCC_A2_inverse-temp_vs_log-activity-coefficient.png")
print("max X_C: ", np.max(activity_df["concentration"]))
print("HM_MIX: ", fits[0][0])
