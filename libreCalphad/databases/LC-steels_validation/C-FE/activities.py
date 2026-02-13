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
from tinydb import Query
import yaml


def _lin_fit(x, m, b):
    return m * x + b


db = load_database("LC-steels-thermo.tdb")
disabled_phases = ["CEMENTITE_D011"]  # stable first
phases = [phase for phase in list(db.phases.keys()) if phase not in disabled_phases]
with open("../../run_param_gen.yaml", "r") as f:
    dataset_folder = yaml.safe_load(f)["system"]["datasets"]
datasets = load_datasets(recursive_glob(dataset_folder))
query = datasets.search(
    Query().components == ["C", "FE", "VA"] & Query().phases == ["FCC_A1"]
)

breakpoint()
