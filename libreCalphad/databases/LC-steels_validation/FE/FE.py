from espei.datasets import load_datasets, recursive_glob
from libreCalphad.databases.db_utils import load_database
import matplotlib.pyplot as plt
import numpy as np
from pycalphad import Workspace, as_property, calculate, variables as v
from pycalphad.mapping import BinaryStrategy, plot_binary
from pycalphad.property_framework.metaproperties import IsolatedPhase
from tinydb import where
import yaml

dbf = load_database("LC-steels-thermo.tdb")
with open("../../run_param_gen.yaml", "r") as f:
    dataset_folder = yaml.safe_load(f)["system"]["datasets"]
datasets = load_datasets(recursive_glob(dataset_folder))
components = ["FE", "VA"]
query = (
    (where("phases") == ["BCC_A2"])
    & (where("components") == components)
    & (where("output") == "CPM")
)
search_results = datasets.search(query)
# CPM plotting
fig, ax = plt.subplots()
for result in search_results:
    ax.scatter(
        result["conditions"]["T"],
        np.array(result["values"]).squeeze(),
        label=result["reference"],
    )
ax.legend()
ax.set_xlabel("Tempurature (K)")
ax.set_ylabel("Heat Capacity (J/mol-K-atom)")
fig.tight_layout()
fig.savefig("./FE-CPM.png")
