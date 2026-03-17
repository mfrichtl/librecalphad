from datetime import date
from espei.datasets import load_datasets, recursive_glob
from espei.plot import dataplot, plot_endmember, plot_interaction
from libreCalphad.databases.db_utils import load_database
from libreCalphad.models.thermodynamics import calculate_energy_from_activity
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pycalphad import Workspace, variables as v
from pycalphad.mapping import BinaryStrategy, plot_binary
from pycalphad.property_framework.metaproperties import IsolatedPhase
from scipy.optimize import curve_fit
from tinydb import Query
import yaml


def _lin_fit(x, m, b):
    return m * x + b


df = pd.DataFrame()
db = load_database("LC-steels-thermo.tdb")
disabled_phases = ["CEMENTITE_D011"]  # stable first
phases = [phase for phase in list(db.phases.keys()) if phase not in disabled_phases]
with open("../../run_param_gen.yaml", "r") as f:
    dataset_folder = yaml.safe_load(f)["system"]["datasets"]
datasets = load_datasets(recursive_glob(dataset_folder))
query = datasets.search(
    Query().components == ["C", "FE", "VA"] and Query().output == "ACR_C"
)

for data in query:
    data_dict = {
        "phase": data["phases"],
        "T": [data["conditions"]["T"]],
        "X_C": data["conditions"]["X_C"],
        "ACR_C": np.squeeze(data["values"]),
    }
    n = len(data_dict["X_C"])
    data_dict["phase"] = data_dict["phase"] * n
    data_dict["T"] = data_dict["T"] * n
    try:
        if df.empty:
            df = pd.DataFrame(data_dict)
        else:
            df = pd.concat([df, pd.DataFrame(data_dict)], ignore_index=True)
    except:
        breakpoint()

for phase in df["phase"].unique():
    fig, ax = plt.subplots()
    phase_df = df.query("phase == @phase")
    ax.scatter(phase_df["X_C"], phase_df["ACR_C"])
    fig.savefig(f"./{phase}-ACR_C.png")
