"""
This script is used for developing a martensite-start temperature prediction algorithm using
pycalphad and a thermodynamic database developed for steels. It calculates the driving
force for the transformation and uses experimental data to predict the martensite start
temperature.
"""
# TODO: Update to LC_martensite_start

from espei import generate_parameters
from espei.datasets import load_datasets, recursive_glob
from espei.parameter_selection.fitting_steps import AbstractLinearPropertyStep
from espei.parameter_selection.fitting_descriptions import ModelFittingDescription
import json
from libreCalphad.databases.db_utils import load_database
from libreCalphad.models.utilities import (
    convert_conditions,
    DG,
    get_components_from_conditions,
    parse_composition,
    trim_conditions,
)
from libreCalphad.plotting import step_plot
from collections import defaultdict
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pycalphad import equilibrium, Model, variables as v
from pycalphad.io.tdb import write_tdb
from scipy.optimize import Bounds, curve_fit, minimize
import sys
import time
import tinydb

# Optional imports for training new models. Running this code without these will not work, but
# these try-except clauses allow tests to pass successfully if the optional dependencies
# are not installed.
try:
    from pandarallel import pandarallel
except ImportError:
    pandarallel = None

try:
    import plotly.graph_objects as go
except ImportError:
    plotly = None

try:
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.metrics import ConfusionMatrixDisplay
except ImportError:
    sklearn = None

try:
    import seaborn as sns

    sns.set_style("whitegrid")
    sns.set_context("paper")
except ImportError:
    seaborn = None

exp_data_dir = "./experimental_data/"
figure_dir = "./figures/"
model_param_dir = "./model_params/"


def train_espei():
    # Setup the pycalphad model for the MS temperature parameters
    class MartensiteStartModel(Model):
        def build_phase(self, dbe):
            super().build_phase(dbe)
            phase = dbe.phases[self.phase_name]
            param_search = dbe.search
            for prop in ["MSL", "MSP", "MSE"]:
                prop_param_query = (
                    (tinydb.where("phase_name") == phase.name)
                    & (tinydb.where("parameter_type") == prop)
                    & (tinydb.where("constituent_array").test(self._array_validity))
                )
                prop_val = self.redlich_kister_sum(phase, param_search, prop_param_query).subs(
                    dbe.symbols
                )
                setattr(self, prop, prop_val)

    # Now define the fitting descriptions
    class StepMartensiteStartLath(AbstractLinearPropertyStep):
        parameter_name = "MSL"
        data_types_read = "MSL"

    class StepMartensiteStartPlate(AbstractLinearPropertyStep):
        parameter_name = "MSP"
        data_types_read = "MSP"

    class StepMartensiteStartEpsilon(AbstractLinearPropertyStep):
        parameter_name = "MSE"
        data_types_read = "MSE"

    martensite_start_fiting_description = ModelFittingDescription(
        [StepMartensiteStartLath, StepMartensiteStartPlate, StepMartensiteStartEpsilon],
        model=MartensiteStartModel,
    )
    datasets = load_datasets(recursive_glob(os.path.join(exp_data_dir, "training")))
    # generate phase_models.json based on the training data
    components = []
    phases = {}
    for root, _, files in os.walk(os.path.join(exp_data_dir, "training")):
        for file in files:
            if file.endswith(".json") and file != "phase_models.json":
                filename = os.path.join([root, file])
                with open(filename, "r") as f:
                    data = json.load(f)
                    for comp in data["components"]:
                        if comp not in components:
                            components.append(comp)
                    for phase in data["phases"]:
                        if phase not in phases.keys():
                            phases[phase] = {
                                "sublattice_model": data["solver"]["sublattice_configurations"],
                                "sublattice_site_ratios": data["solver"]["sublattice_site_ratios"],
                            }
                        else:
                            for sl in range(len(data["solver"]["sublattice_model"])):
                                for comp in data["solver"]["sublattice_configurations"][sl]:
                                    if comp not in phases[phase]["sublattice_models"][sl]:
                                        phases[phase]["sublattice_models"][sl].append(comp)
    phase_models = {"components": components, "phases": phases}
    with open(os.path.join([exp_data_dir, "training", "phase_models.json"]), "w") as f:
        json.dump(phase_models, f)
    dbf = generate_parameters(
        phase_models, datasets, "SGTE91", "linear", martensite_start_fiting_description
    )
    write_tdb(dbf, "./martensite_start.tdb")


figsize = (6.5, 4)
dbf = "mf-steel.tdb"
max_num_conds_DG = (
    12  # stack smashing errors seem to be associated with too many conditions in a computation
)
remove_list = [
    "FE",
    "GE",
]  # Remove these from component lists, Fe removed because it is the dependent component in all subsequent calculations
# do_not_remove_list = ['C', 'N', 'NB', 'O', 'TI', 'V', 'W']  # Don't remove the elements that can form stable high-temperature compounds that affect austenite composition
do_not_remove_list = ["C", "N"]
interstitials = ["B", "C", "H", "N", "O"]
disabled_phases = [
    "BCC_B2",
    "BCC_4SL",
    "FCC_L10",
    "FCC2_L10",
    "FCC_L12",
    "GAS",
    "HCP_L12",
    "KAPPA_A1",
    "KAPPA_E21",
    "IONIC_LIQ",
    "TAU2_ALFEMO_A2",
    "LAVES2_C14",
]  # disable some phases for calculations
nb_workers = 12  # number of cores you want to use for parallel computing
solute_threshold = 1e-12  # threshold below which I will discard a solute for predicting the Ms temperature, without it the calculations seem to stall.
mf_plate_baseline_fits = [2089]  # from my constant fit for Fe- alloys
mf_lath_baseline_fits = [1249]  # from my constant fit for the Fe- alloys
mf_epsilon_baseline_fits = [-75]
# PAGS fits using certain references. See code below for plots to find more information.
# TODO: Break these out separate functions so they can be fit dynamically so they don't need to be hard-coded.
mf_plate_PAGS_fits = [-57.66174052, 259.52648073]
mf_lath_PAGS_fits = [
    1.65754133,
    -38.90842902,
    300.39266453,
]  # 30 J/mol to offset and make the energy contribution at 100 um be ~0
mf_epsilon_PAGS_fits = [7.27409320e01, 1.81920225e-06, -1.65812954e-01, -4.98269462e02]


conc_arrays = {
    "lath-mf": {
        "Fe-C": np.linspace(0, 0.06, 4),
        "Fe-Co": np.linspace(0, 0.33, 7),
        "Fe-Cr": np.linspace(0, 0.15, 10),
        "Fe-Cu": np.linspace(0, 0.08, 3),
        "Fe-Mn": np.linspace(0, 0.15, 6),
        "Fe-N": np.linspace(0, 0.055, 3),
        "Fe-Ni": np.linspace(0, 0.25, 10),
        "C-Mn": [np.linspace(0, 0.03, 4), np.linspace(0, 0.05, 4)],
        "Al-Ni": [np.linspace(0, 0.05, 3), np.linspace(0.1, 0.2, 3)],
        "C-Cr": [np.linspace(0, 0.02, 3), np.linspace(0, 0.11, 3)],
        "C-Ni": [np.linspace(0, 0.025, 3), np.linspace(0.00, 0.2, 7)],
        "Co-Ni": [np.linspace(0, 0.35, 7), np.linspace(0.15, 0.3, 7)],
        "Cr-Ni": [np.linspace(0, 0.2, 6), np.linspace(0.07, 0.2, 5)],
        "Cu-Ni": [np.linspace(0, 0.05, 4), [0.16, 0.18, 0.2]],
        "Mo-Ni": [np.linspace(0, 0.04, 4), [0.16, 0.18, 0.2]],
        "Mn-Ni": [np.linspace(0, 0.05, 3), np.linspace(0.16, 0.2, 4)],
        "Ni-Si": [np.linspace(0.16, 0.2, 4), np.linspace(0, 0.05, 4)],
        "Ni-Ti": [[0.195], np.linspace(0, 0.03, 4)],
        "Ni-V": [np.linspace(0.15, 0.23, 3), np.linspace(0, 0.04, 2)],
        "Ni-W": [[0.14, 0.16], np.linspace(0, 0.04, 4)],
    },
    # not enough data
    #   'C-Ni': [np.linspace(0, 0.025, 2), np.linspace(0.05, 0.14, 4)], 'C-Co': [[0, 0.05], np.linspace(0.03, 0.1, 4)],
    #   'C-Mo': [np.linspace(0.01, 0.05, 3), np.linspace(0.005, 0.02, 3)]
    #   'C-Si': [np.linspace(0, 0.02, 3), np.linspace(0.00, 0.1, 3)]  2024-08-29: C-SI increased error
    #
    "plate-mf": {
        "Fe-C": np.linspace(0, 0.1, 6),
        "Fe-Co": np.linspace(0, 0.33, 10),
        "Fe-Cr": np.linspace(0, 0.15, 10),
        "Fe-Cu": np.linspace(0, 0.08, 3),
        "Fe-Mn": np.linspace(0, 0.14, 4),
        "Fe-N": np.linspace(0, 0.08, 4),
        "Fe-Ni": np.linspace(0, 0.25, 10),
        "Al-Ni": [np.linspace(0, 0.05, 3), np.linspace(0.2, 0.22, 3)],
        "C-Cr": [np.linspace(0, 0.01, 3), np.linspace(0, 0.1, 5)],
        "C-Mn": [np.linspace(0, 0.04, 4), np.linspace(0, 0.10, 4)],
        "Co-Ni": [np.linspace(0, 0.23, 7), np.linspace(0.15, 0.25, 4)],
        "Cr-N": [np.linspace(0.1, 0.15, 3), np.linspace(0, 0.03, 4)],
        # not enough data -- probably Ms too low to model without extension of systems to 0 K
        # 'Cr-Ni': [np.linspace(0, 0.05, 2), np.linspace(0.2, 0.23, 3)], 'Cu-Ni': [np.linspace(0, 0.05, 3), np.linspace(0.2, 0.23, 3)],
        # 'Mo-Ni': [np.linspace(0, 0.04, 4), [0.25]], 'Ni-Si': [np.linspace(0.2, 0.23, 3), np.linspace(0, 0.05, 3)], 'Ni-V': [np.linspace(0.2, 0.24, 3), np.linspace(0, 0.04, 2)],
        # 'Ni-W': [np.linspace(0.18, 0.24, 3), np.linspace(0, 0.04, 2)],
    },
    "epsilon-mf": {
        "Fe-Mn": np.linspace(0.1, 0.235, 10),
        "Al-Mn": [np.linspace(0, 0.05, 3), np.linspace(0.15, 0.17, 3)],
        "C-Mn": [np.linspace(0, 0.015, 4), np.linspace(0.12, 0.26, 6)],
        "Co-Mn": [np.linspace(0, 0.05, 3), np.linspace(0.1, 0.18, 4)],
        "Cr-Mn": [np.linspace(0, 0.05, 3), np.linspace(0.1, 0.2, 4)],
        "Cu-Mn": [np.linspace(0, 0.025, 4), np.linspace(0.16, 0.18, 3)],
        "Mn-Mo": [np.linspace(0.08, 0.18, 5), np.linspace(0, 0.04, 3)],
        "Mn-Nb": [np.linspace(0.16, 0.18, 3), np.linspace(0, 0.01, 4)],
        "Mn-Ni": [np.linspace(0.16, 0.18, 3), np.linspace(0, 0.03, 3)],
        "Mn-Si": [np.linspace(0.15, 0.26, 5), np.linspace(0, 0.15, 4)],
        "Mn-Ti": [[0.15, 0.176], np.linspace(0, 0.02, 4)],
        "Mn-V": [np.linspace(0.16, 0.19, 3), np.linspace(0, 0.03, 3)],
        "Mn-W": [[0.15, 0.178], np.linspace(0, 0.03, 3)],
    },
    "lath-storm": {
        "Fe-C": np.linspace(0, 0.04, 5),
        "Fe-Co": np.linspace(0, 0.3, 10),
        "Fe-Cr": np.linspace(0, 0.1, 10),
        "Fe-Cu": np.linspace(0, 0.1, 5),
        "Fe-Mn": np.linspace(0, 0.3, 10),
        "Fe-Ni": np.linspace(0, 0.25, 10),
        "C-Cr": [np.linspace(0, 0.02, 5), np.linspace(0, 0.1, 10)],
    },
    "plate-storm": {
        "Fe-C": np.linspace(0, 0.08, 5),
        "Fe-Co": np.linspace(0, 0.3, 10),
        "Fe-Cr": np.linspace(0, 0.1, 10),
        "Fe-Cu": np.linspace(0, 0.1, 5),
        "Fe-Mn": np.linspace(0, 0.3, 10),
        "Fe-Ni": np.linspace(0, 0.25, 10),
        "C-Cr": [np.linspace(0, 0.02, 5), np.linspace(0, 0.1, 10)],
    },
}
lath_orders = {
    "martensite_start": 0,
    "Fe-C": 0,
    "Fe-Co": 1,
    "Fe-Cr": 0,
    "Fe-Cu": 0,
    "Fe-Mn": 0,
    "Fe-N": 0,
    "Fe-Ni": 1,
    "Al-Ni": 0,
    "C-Cr": 0,
    "C-Mn": 0,
    "C-Ni": 0,
    "Co-Ni": 0,
    "Cr-Ni": 0,
    "Cu-Ni": 0,
    "Mn-Ni": 0,
    "Mo-Ni": 0,
    "Ni-Si": 0,
    "Ni-Ti": 0,
    "Ni-V": 0,
    "Ni-W": 0,
    # 'C-Si': 0,
}
plate_orders = {
    "martensite_start": 0,
    "Fe-C": 1,
    "Fe-Co": 1,
    "Fe-Cr": 0,
    "Fe-Cu": 0,
    "Fe-Mn": 1,
    "Fe-N": 0,
    "Fe-Ni": 1,
    "Al-Ni": 0,
    "C-Cr": 0,
    "C-Mn": 0,
    "Co-Ni": 1,
    "Cr-N": 0,
    # 'Cr-Ni': 0, 'Cu-Ni': 0, 'Mo-Ni': 0, 'Ni-Si': 0, 'Ni-V': 0, 'Ni-W': 0,
}
epsilon_orders = {
    "martensite_start": 0,
    "Fe-Mn": 1,
    "Al-Mn": 0,
    "C-Mn": 0,
    "Co-Mn": 0,
    "Cr-Mn": 0,
    "Cu-Mn": 0,
    "Mn-Mo": 0,
    "Mn-Nb": 0,
    "Mn-Ni": 0,
    "Mn-Si": 0,
    "Mn-Ti": 0,
    "Mn-V": 0,
    "Mn-W": 0,
}
#

# Ignore selected entries for fitting -- probably from some impurities or something

selected_ignore = {"plate": {"Fe-C": ["Grange1946", "Hanemann1932"]}, "lath": {}, "epsilon": {}}

included_components = {"lath": ["VA"], "plate": ["VA"], "epsilon": ["VA"]}
for term in lath_orders:
    if term == "martensite_start":
        continue
    else:
        for component in term.split("-"):
            if component.upper() not in included_components["lath"]:
                included_components["lath"].append(component.upper())
    assert term in conc_arrays["lath-mf"], f"Lath concentration array missing {term}."
for term in plate_orders:
    if term == "martensite_start":
        continue
    else:
        for component in term.split("-"):
            if component.upper() not in included_components["plate"]:
                included_components["plate"].append(component.upper())
    assert term in conc_arrays["plate-mf"], f"Plate concentration array missing {term}."
for term in epsilon_orders:
    if term == "martensite_start":
        continue
    else:
        for component in term.split("-"):
            if component.upper() not in included_components["epsilon"]:
                included_components["epsilon"].append(component.upper())
    assert term in conc_arrays["epsilon-mf"], f"Epsilon concentration array missing {term}."


def remove_min_component(components, conditions, dependent_element="FE"):
    n = 0
    removed = False
    components = np.array(
        components
    )  # need to make sure it's cast as an ndarray for the delete function to work properly.
    while not removed and n < len(conditions):
        min_element = list(conditions.keys())[
            list(conditions.values()).index(sorted(conditions.values())[n])
        ]
        component = str(min_element).split("_")[1]
        if component not in do_not_remove_list and component != dependent_element:
            components = np.delete(components, np.where(components == component))
            conditions.pop(min_element)
            removed = True
            print(f"Removing {component}.")
        n += 1
    return components, conditions


def get_components(row):
    components = ["VA"]
    solutes = [
        comp.upper() for comp in row.keys() if len(comp) < 3 and comp != "DG" and row[comp] > 0
    ]
    components = np.concatenate([solutes, components])
    return components


def get_conditions(row):
    conditions = {v.N: 1, v.P: 101325, v.T: row["martensite_start"]}
    components = get_components(row)
    # Remove components not included in the fitting
    # if row['type'] == 'lath':
    #     components = [comp for comp in components if comp in included_components['lath']]
    # elif row['type'] == 'plate':
    #     components = [comp for comp in components if comp in included_components['plate']]
    # elif row['type'] == 'epsilon':
    #     components = [comp for comp in components if comp in included_components['epsilon']]
    for comp in components:
        if comp != "FE" and comp != "VA":
            conditions.update({v.X(comp): row[comp.capitalize()]})
    components, conditions = trim_conditions(
        components,
        conditions,
        max_num_conditions=max_num_conds_DG,
        solute_threshold=solute_threshold,
        always_remove_list=remove_list,
        always_keep_list=do_not_remove_list,
    )
    row["components"] = components
    row["conditions"] = conditions

    return row


def calc_aus_comp(row, db):
    """
    Function to calculate the equilibrium austenite composition.

    Parameters : row : Pandas series
                    Pandas series that contains indices for the overall alloy composition and austenitizing temperature.

    Returns : row : Pandas series
                Modified row containing an index for the austenite composition and corresponding pycalphad conditions.
    """
    if pd.isnull(
        row["austenitization_temperature"]
    ):  # solutionizing tempeature not provided by publication
        row["austenitization_temperature"] = 1273.15  # assume 1000 C

    phases = [phase for phase in list(db.phases.keys()) if phase not in disabled_phases]
    soln_phases = []
    components = row["components"]
    conditions = row["conditions"]
    row["error_message"] = []
    conditions.update({v.T: row["austenitization_temperature"]})
    print(conditions, components)
    eq_energy = np.nan
    pdens = 500
    n = 0
    while pd.isnull(eq_energy) and n < 10:  # arbitrary cut off for now
        if n > 0 and pdens < 3000:
            pdens += 500
        if (
            pdens >= 3000
        ):  # dynamically remove components with the lowest atomic fraction if calculations are still failing
            components, conditions = remove_min_component(
                components, conditions, dependent_element="FE"
            )
            row["calc_error"] = True
            row["error_message"].append(
                f"Removed {n} components to calculate solutionizing equilibrium."
            )
        row["solutionizing_components"] = components
        row["solutionizing_conditions"] = conditions
        try:
            eq = equilibrium(db, components, phases, conditions, calc_opts={"pdens": pdens})
        except Exception as e:
            print(e)
            print(components)
            print(phases)
            print(conditions)
        eq_energy = eq.GM.squeeze().values
        print(f"Attempt: {n + 1}, Austenite calculated energy: {eq_energy} J/mol")
        print(conditions)
        n += 1

    soln_phases = [phase for phase in eq.Phase.squeeze().values if phase != ""]
    row["solutionizing_phases"] = soln_phases

    austenite_components = None
    austenite_composition = {}
    austenite_conditions = {v.P: 101325, v.N: 1}
    max_eq = eq.where(eq.NP == np.max(eq.NP))
    mat_NP = max_eq.NP.squeeze()[~pd.isnull(max_eq.NP.squeeze())]
    primary_phase = max_eq.Phase.squeeze()[
        ~pd.isnull(max_eq.Phase.squeeze())
    ]  # select the phase with the highest volume fraction
    if primary_phase.values != "FCC_A1":
        print(
            f"Austenite not identified as primary phase for {row['conditions']}, {row['reference']}."
        )
        row["calc_error"] = True
        row["error_message"].append(
            "Austenite not identified as primary phase during solutionizing treatment."
        )
    if mat_NP.values < 0.5:
        print(f"Cannot identify a primary phase during solutionizing treatment.")
        row["calc_error"] = True
        row["error_message"].append(f"Austenite phase NP < 0.5 during solutionizing treatment")
    if len(primary_phase.values) == 0:
        row["austenite_composition"] = austenite_composition
        row["austenite_components"] = austenite_components
        row["austenite_conditions"] = austenite_conditions
        row["calc_error"] = True
        row["error_message"].append(
            f"Cannot identify a primary phase during solutionizing treatment."
        )
        return row

    try:
        for component in max_eq.component.values:
            x_comp = max_eq.where(max_eq.component == component).X.squeeze().values
            x_comp = x_comp[~np.isnan(x_comp)][0]
            austenite_composition[component] = x_comp
            if component != "FE" and component != "VA":
                austenite_conditions.update({v.X(component): x_comp})
            if component == "FE" and x_comp < 0.5:
                print(f"Solution phase is Fe-poor with {conditions}")
                row["calc_error"] = True
                row["error_message"].append(f"Solution phase is Fe-poor.")
        austenite_components = np.append(max_eq.component.values, "VA")
        # Trim austenite conditions
        austenite_components, austenite_conditions = trim_conditions(
            austenite_components,
            austenite_conditions,
            max_num_conditions=max_num_conds_DG,
            solute_threshold=solute_threshold,
            always_remove_list=remove_list,
            always_keep_list=do_not_remove_list,
        )
        print(f"Overall alloy conditions: {conditions}")
        print(f"Austenite conditions: {austenite_conditions}")
        row["austenite_composition"] = austenite_composition
        row["austenite_components"] = austenite_components
        row["austenite_conditions"] = austenite_conditions

        return row

    except Exception as e:
        row["calc_error"] = True
        row["error_message"].append(
            f"Error processing solution phase composition for {row['conditions']}. Message: {getattr(e, 'message', str(e))}."
        )
        row["austenite_composition"] = austenite_composition
        row["austenite_components"] = austenite_components
        row["austenite_conditions"] = austenite_conditions
        return row


def get_dG(row, db):
    """
    Function to calculate the energy difference between FCC and BCC or HCP phases at the reported martensite start temperature.
    """
    conditions = row["austenite_conditions"].copy()
    components = row["austenite_components"].copy()
    if len(conditions) == 0:  # there was a calculation error with the austenite composition
        return np.nan
    conditions.update({v.T: row["martensite_start"]})
    phases = []
    if row["type"] == "epsilon":
        phases.append("HCP_A3")
    elif row["type"] == "lath" or row["type"] == "plate" or row["type"] == "alpha":
        phases.append("BCC_A2")

    if row["parent"] == "gamma":
        phases.append("FCC_A1")
    elif row["parent"] == "epsilon":
        phases.append("HCP_A3")

    assert len(phases) == 2, f"Cannot identify phases for {row}"

    energy = np.nan
    pdens = 500
    n = 0
    while pd.isnull(energy) and n < 10:
        if n > 0 and pdens < 3000:
            pdens += 500
        if (
            pdens >= 3000
        ):  # dynamically remove components with the lowest atomic fraction if calculations are still failing
            components, conditions = remove_min_component(
                components, conditions, dependent_element="FE"
            )
            row["calc_error"] = True
            row["error_message"].append(
                f"Removed {n} components while calculating martensite equilibrium."
            )
        row["martensite_components"] = components
        row["martensite_conditions"] = conditions
        energy = -DG(db, components, phases, conditions, calc_opts={"pdens": pdens})
        n += 1

    # Add grain-size contribution to DG
    if np.isnan(row["PAGS"]):
        pags_energy = 0
    elif row["type"] == "plate":
        pags_energy = mf_plate_PAGS_fits[0] * np.log(row["PAGS"]) + mf_plate_PAGS_fits[1]
    elif row["type"] == "lath":
        if row["PAGS"] >= 100:
            pags_energy = 0
        else:
            pags_energy = (
                mf_lath_PAGS_fits[0] * np.log(row["PAGS"] ** mf_lath_PAGS_fits[1])
                + mf_lath_PAGS_fits[2]
            )
    elif row["type"] == "epsilon":
        pags_energy = (
            mf_epsilon_PAGS_fits[0]
            * np.log(row["PAGS"] ** mf_epsilon_PAGS_fits[1]) ** mf_epsilon_PAGS_fits[2]
            + mf_epsilon_PAGS_fits[3]
        )
    else:
        pags_energy = 0

    print(components, conditions)
    print(
        f"Phases: {phases}, DG: {energy}, PAGS: {pags_energy}, Experimental Ms: {row['martensite_start']}"
    )
    row["DG_no_PAGS"] = energy
    row["PAGS_energy"] = pags_energy
    row["DG"] = energy + pags_energy
    return row


def calc_t0(row, db):
    def min_t0(T, db, components, phases, conditions):
        conditions.update({v.T: T})
        return np.abs(DG(db, components, phases, conditions, calc_opts={"pdens": 2000}))

    components = ["FE", "VA"]
    conditions = row["conditions"]

    for cond in list(conditions.keys()):
        cond = cond.__str__()
        if cond.startswith("X"):
            components = np.concatenate([components, [cond.split("_")[1]]])

    if row["type"] == "T0-lath":
        row["type"] = "T0-alpha"
    model_type = row["type"]

    if "epsilon" in model_type:
        phases = ["FCC_A1", "HCP_A3"]
    else:
        phases = ["FCC_A1", "BCC_A2"]
    temp_bounds = Bounds(1e-6, 1300)
    calc = minimize(
        min_t0,
        500,
        args=(db, components, phases, conditions),
        method="Nelder-Mead",
        bounds=temp_bounds,
    )
    t0_temp = calc.x[0]
    row["martensite_start"] = t0_temp

    return row


def predict_Ms(row, db, storm=False):
    # JSON-stored conditions are strings, need to turn them back to pycalphad variables
    conditions = convert_conditions(row["austenite_conditions"].copy())
    row["pred_martensite_conditions"] = conditions
    # conditions = convert_conditions(row['martensite_conditions'].copy())
    components = get_components_from_conditions(conditions, dependent_component="FE")
    # row['martensite_conditions'] = conditions
    ms_type = row["type"]
    if np.isnan(row["DG"]):
        return np.nan
    if row["new_DG"] == False:  # Don't re-calculate unless the DG model changed.
        print(f"DG unchanged for: {ms_type}, {conditions}")
        return row
    else:
        row["calc_error"] = False
        if "epsilon" in ms_type:
            phases = ["HCP_A3", "FCC_A1"]
        else:
            phases = ["BCC_A2", "FCC_A1"]
        pags = row["PAGS"]
        mf_fits = None
        temp_bounds = Bounds(298.15, 1300)
        n = 0
        pdens = 500
        pred_DG = np.nan
        pred_Ms = row["martensite_start"]
        attempts = []
        remove_component = False
        DG_threshold = (
            0.25  # J/mol, threshold for calling a minimization "good." Chosen semi-arbitrarily
        )
        while (pred_DG > DG_threshold or pd.isnull(pred_DG)) and pred_Ms > 298.15 and n < 10:
            if (
                remove_component
            ):  # Remove the lowest concentration component until the calculation completes.
                components, conditions = remove_min_component(
                    components, conditions, dependent_element="FE"
                )
                row["pred_martensite_conditions"] = conditions
                row["calc_error"] = True
                row["error_message"].append(f"Removed {n} components from martensite calculation.")
            try:
                model_params = pd.read_json(
                    "".join([model_param_dir, f"mf_{ms_type}_parameters.json"])
                )
                if ms_type == "plate":
                    if storm:
                        calc = minimize(
                            get_model_Ms,
                            pred_Ms,
                            args=(
                                "plate-storm",
                                db,
                                components,
                                phases,
                                conditions,
                                pags,
                                pdens,
                                [0],
                                False,
                            ),
                            method="Nelder-Mead",
                            bounds=temp_bounds,
                        )
                    else:
                        mf_fits = model_params["mf_plate_fits"][0]
                        calc = minimize(
                            get_model_Ms,
                            pred_Ms,
                            args=(
                                "plate-mf",
                                db,
                                components,
                                phases,
                                conditions,
                                pags,
                                pdens,
                                mf_fits,
                                False,
                            ),
                            method="Nelder-Mead",
                            bounds=temp_bounds,
                        )
                elif ms_type == "lath":
                    if storm:
                        calc = minimize(
                            get_model_Ms,
                            pred_Ms,
                            args=(
                                "lath-storm",
                                db,
                                components,
                                phases,
                                conditions,
                                pags,
                                pdens,
                                [0],
                                False,
                            ),
                            method="Nelder-Mead",
                            bounds=temp_bounds,
                        )
                    else:
                        mf_fits = model_params["mf_lath_fits"][0]
                        calc = minimize(
                            get_model_Ms,
                            pred_Ms,
                            args=(
                                "lath-mf",
                                db,
                                components,
                                phases,
                                conditions,
                                pags,
                                pdens,
                                mf_fits,
                                False,
                            ),
                            method="Nelder-Mead",
                            bounds=temp_bounds,
                        )
                elif ms_type == "epsilon":  # no Stormvinter epsilon model
                    mf_fits = model_params["mf_epsilon_fits"][0]
                    calc = minimize(
                        get_model_Ms,
                        pred_Ms,
                        args=(
                            "epsilon-mf",
                            db,
                            components,
                            phases,
                            conditions,
                            pags,
                            pdens,
                            mf_fits,
                            False,
                        ),
                        method="Nelder-Mead",
                        bounds=temp_bounds,
                    )
                else:
                    return np.nan
                pred_Ms = calc.x[0]
                pred_DG = calc.fun

            except Exception as e:
                print(e)
                pred_Ms = np.nan
                pred_DG = np.nan

            print(f"Attempt: {n + 1}, {pred_Ms}, {pred_DG}")
            attempts.append(pred_Ms)
            if pred_DG > DG_threshold or np.isnan(pred_DG):  # Unsuccesfull calculation
                if pdens < 3000:
                    pdens += 500
                else:
                    remove_component = True
                if n > 0:
                    if (
                        np.abs(attempts[n] - attempts[n - 1]) < 0.1
                    ):  # basically the same calculation as last time --> refining point density didn't help
                        remove_component = True
            n += 1

        if storm:
            row["storm_martensite_start"] = pred_Ms
            row["contrib_dict"] = np.nan
        else:
            row["mf_martensite_start"] = pred_Ms
            row["contrib_dict"], row["mf_martensite_DG"] = get_model_Ms(
                pred_Ms,
                f"{ms_type}-mf",
                db,
                components,
                phases,
                conditions,
                pags,
                pdens,
                mf_fits,
                True,
            )

        row["martensite_DG"] = pred_DG

        if np.abs(pred_DG) > DG_threshold or pd.isnull(
            pred_DG
        ):  # Find where the solver is performing poorly
            print(f"Solver problem with: {ms_type}, {conditions}")
            print(f"Calculated Ms: {pred_Ms}")
            print(f"Calculated DG: {pred_DG}")
            row["calc_error"] = True
            row["error_message"].append(f"Solver problem with: {ms_type}, {conditions}.")

        print(f"Ms DG: {pred_DG} for {ms_type}, {row['martensite_start']}, {conditions}")
        return row


def get_model_Ms(
    T, model_type, db, components, phases, conditions, pags, pdens, C=[0], ret_contrib_dict=False
):
    conditions.update({v.T: T})
    model_energy = np.nan
    try:
        eq_DF = -DG(db, components, phases, conditions, calc_opts={"pdens": pdens})
        if "lath" in model_type and "storm" in model_type:
            model_energy = get_lath_model_storm(T, components, conditions)
        if "plate" in model_type and "storm" in model_type:
            model_energy = get_plate_model_storm(components, conditions)
        if "plate" in model_type and "mf" in model_type:
            model_energy, contrib_dict = get_model(components, conditions, C, model_type, pags)
        if "lath" in model_type and "mf" in model_type:
            model_energy, contrib_dict = get_model(components, conditions, C, model_type, pags)
        if "epsilon" in model_type and "mf" in model_type:
            model_energy, contrib_dict = get_model(components, conditions, C, model_type, pags)
        assert not np.isnan(model_energy), (
            f"Model energy incorrectly calculated for {model_type, conditions}"
        )
    except Exception as e:
        print(e)
        return [np.nan]  # brackets needed for compatibility with other models

    if ret_contrib_dict:
        return contrib_dict, model_energy
    else:
        return np.abs(eq_DF - model_energy)


def get_plate_model_storm(components, conditions):
    conditions = conditions.copy()
    comp_dict = defaultdict(lambda: 0)
    for component in components:
        if v.X(component) in conditions.keys():
            comp_dict.update({component: conditions[v.X(component)]})

    plate_barrier = (
        2100
        + 75000 * comp_dict["C"] ** 2 / (1 - comp_dict["C"])
        - 11500 * comp_dict["C"]
        - 2970 * comp_dict["CR"]
        + 3574 * comp_dict["MN"]
        - 5104 * comp_dict["NI"]
        + 441700 * comp_dict["C"] * comp_dict["CR"] / (1 - comp_dict["C"])
    )
    return plate_barrier


def get_lath_model_storm(T, components, conditions):
    comp_dict = defaultdict(lambda: 0)
    for component in components:
        if v.X(component) in conditions.keys():
            comp_dict.update({component: conditions[v.X(component)]})

    lath_barrier = (
        3640
        - 2.92 * T
        + 346400 * comp_dict["C"] ** 2 / (1 - comp_dict["C"])
        - 16430 * comp_dict["C"]
        - 785.5 * comp_dict["CR"]
        + 7119 * comp_dict["MN"]
        - 4306 * comp_dict["NI"]
        + 350600 * comp_dict["C"] * comp_dict["CR"] / (1 - comp_dict["C"])
    )

    return lath_barrier


def fit_model(orders, data, model_type):
    """
    This function fits experimental data using SciPy's curve_fit function. The intent is to create a summation of interaction terms
    that are consistent with Redlich-Kister polynomial expansions used in CALPHAD. When sufficient ternary system data
    are available, the summation includes a Muggiano extrapolation term for the non-Fe binary subsystem.

    Epsilon models are fit without regard for the Mn-X interactions due to insufficient data.
    """
    terms = list(orders.keys())
    assert "martensite_start" in terms, (
        "Need to pass the Ms temperatures for fitting the non-chemistry-dependent terms...try again."
    )
    fits = {}
    data = data.query("PAGS_energy < 25").copy()  # Just remove the tiny grained stuff for fitting

    data["excess_DG"] = data["DG"]  # initialize excess

    # First subtract out the driving force required for pure iron so that the alloying elements can be fit to account
    # for the excess driving force
    if model_type == "epsilon":
        sub_data = data
        if len(sub_data) == 0:
            return [0]  # no Mn system, no epsilon data.
        fits["martensite_start"] = [mf_epsilon_baseline_fits]
        data["excess_DG"] = data["DG"] - mf_epsilon_baseline_fits[0]
        data["excess_DG"] = data.apply(
            lambda row: row["excess_DG"]
            if np.isnan(row["PAGS"])
            else row["excess_DG"]
            - (
                mf_epsilon_PAGS_fits[0]
                * np.log(row["PAGS"] ** mf_epsilon_PAGS_fits[1]) ** mf_epsilon_PAGS_fits[2]
                + mf_epsilon_PAGS_fits[3]
            ),
            axis=1,
        )
    elif model_type == "lath":
        sub_data = data.query("alloy_system == 'Fe-'")
        fits["martensite_start"] = [mf_lath_baseline_fits]
        data["excess_DG"] = data["DG"] - mf_lath_baseline_fits[0]
        data["excess_DG"] = data.apply(
            lambda row: row["excess_DG"]
            if np.isnan(row["PAGS"])
            else row["excess_DG"]
            - (
                mf_lath_PAGS_fits[0] * np.log(row["PAGS"] ** mf_lath_PAGS_fits[1])
                + mf_lath_PAGS_fits[2]
            ),
            axis=1,
        )
    elif model_type == "plate":
        fits["martensite_start"] = [mf_plate_baseline_fits]
        data["excess_DG"] = data["DG"] - mf_plate_baseline_fits[0]
        data["excess_DG"] = data.apply(
            lambda row: row["excess_DG"]
            if np.isnan(row["PAGS"])
            else row["excess_DG"]
            - (mf_plate_PAGS_fits[0] * np.log(row["PAGS"]) + mf_plate_PAGS_fits[1]),
            axis=1,
        )

    quat_systems = []
    processed_Mn = False
    delay = False
    processed_systems = []
    # Arrange systems to fit those with Fe in them first
    fe_systems = [system for system in terms if "Fe" in system.split("-")]
    non_fe_systems = [system for system in terms if system not in fe_systems]
    ordered_terms = [*fe_systems, *non_fe_systems]

    for fit_term in ordered_terms:
        print(fit_term)
        if "Fe" not in fit_term:
            for term in fit_term.split("-"):
                if "Fe-" + term in terms and "Fe-" + term not in processed_systems:
                    delay = True
                    print(f"Trying to process {fit_term} before {term}, delaying.")
            if delay:
                terms.append(fit_term)  # move to the back of the line!
                continue
        if model_type == "epsilon" and fit_term == "Fe-Mn":
            processed_Mn = True
        if len(fit_term.split("-")) == 3:  # quaternary interaction
            quat_systems.append(fit_term)
            continue
        if fit_term == "martensite_start":
            continue
        order = orders[fit_term]
        if (
            model_type == "epsilon" and fit_term != "Fe-Mn"
        ):  # need to add manganese because it's required for epsilon to form
            assert processed_Mn == True, "Trying to fit another epsilon term before Mn."

        system_term = fit_term
        system_term = system_term.split("-")
        system_term.sort()
        if "Fe" in system_term:
            system_term.remove("Fe")
        system_term = "Fe-" + "-".join(system_term)
        sub_data = data.query("alloy_system == @system_term & type == @model_type")
        assert len(sub_data["excess_DG"]) > 0, (
            f"No experimental data passed for {system_term}, {model_type} model."
        )
        split_term = fit_term.split("-")
        x_b = x_c = x_n = x_o = np.zeros(len(sub_data))  # initialize interstitial values as zeros.
        fullx_b = fullx_c = fullx_n = fullx_o = np.zeros(len(data))
        if len(split_term) == 2:
            x_i = split_term[0]
            x_j = split_term[1]
            if x_i == "Fe":  # Fe-X binary
                x_0 = sub_data[x_j]
                x_1 = np.ones(len(sub_data))
                fullx_0 = data[x_j]
                fullx_1 = np.ones(len(data))
            elif x_j == "Fe":  # Fe-X binary
                x_0 = sub_data[x_i]
                x_1 = np.ones(len(sub_data))
                fullx_0 = data[x_i]
                fullx_1 = np.ones(len(data))
            # elif model_type == 'epsilon' and x_i == 'Mn':
            #     x_0 = sub_data[x_j]
            #     x_1 = np.ones(len(sub_data))
            #     fullx_0 = data[x_j]
            #     fullx_1 = np.ones(len(data))
            # elif model_type == 'epsilon' and x_j == 'Mn':
            #     x_0 = sub_data[x_i]
            #     x_1 = np.ones(len(sub_data))
            #     fullx_0 = data[x_i]
            #     fullx_1 = np.ones(len(data))
            else:  # solute interaction
                x_0 = sub_data[x_i]
                x_1 = sub_data[x_j]
                fullx_0 = data[x_i]
                fullx_1 = data[x_j]
            if (
                any([x_i in interstitials, x_j in interstitials]) and order > 0
            ):  # need to change the denominator to account for higher order and interstitial interactions
                x_b = sub_data["B"]
                x_c = sub_data["C"]
                x_n = sub_data["N"]
                x_o = sub_data["O"]
                fullx_b = data["B"]
                fullx_c = data["C"]
                fullx_n = data["N"]
                fullx_o = data["O"]
            print(f"Modeling interactions for {fit_term}.")
            if order == 0:
                fits[fit_term] = curve_fit(
                    lambda x, C_0: x[0] * x[1] * C_0 / (1 - x[2] - x[3] - x[4] - x[5]),
                    np.vstack([x_0, x_1, x_b, x_c, x_n, x_o]),
                    sub_data["excess_DG"],
                )
                data["excess_DG"] = data["excess_DG"] - (
                    fullx_0
                    * fullx_1
                    * fits[fit_term][0][0]
                    / (1 - fullx_b - fullx_c - fullx_n - fullx_o)
                )
            elif order == 1:
                fits[fit_term] = curve_fit(
                    lambda x, C_0, C_1: x[0]
                    * x[1]
                    / (1 - x[2] - x[3] - x[4] - x[5])
                    * (C_0 + C_1 * (x[0] * x[1])),
                    np.vstack([x_0, x_1, x_b, x_c, x_n, x_o]),
                    sub_data["excess_DG"],
                )
                data["excess_DG"] = data["excess_DG"] - (
                    fullx_0
                    * fullx_1
                    / (1 - fullx_b - fullx_c - fullx_n - fullx_o)
                    * (fits[fit_term][0][0] + fits[fit_term][0][1] * (fullx_0 * fullx_1))
                )
            elif order == 2:
                fits[fit_term] = curve_fit(
                    lambda x, C_0, C_1, C_2: x[0]
                    * x[1]
                    / (1 - x[2] - x[3] - x[4] - x[5])
                    * (C_0 + (C_1 * (x[0] * x[1]) + C_2 * (x[0] * x[1]) ** 2)),
                    np.vstack([x_0, x_1, x_b, x_c, x_n, x_o]),
                    sub_data["excess_DG"],
                )
                data["excess_DG"] = data["excess_DG"] - (
                    fullx_0
                    * fullx_1
                    / (1 - fullx_b - fullx_c - fullx_n - fullx_o)
                    * (
                        fits[fit_term][0][0]
                        + (
                            fits[fit_term][0][1] * (fullx_0 * fullx_1)
                            + fits[fit_term][0][2] * (fullx_0 * fullx_1) ** 2
                        )
                    )
                )
            else:
                raise NotImplementedError(f"Fit order for {model_type} {fit_term} not implemented.")
            processed_systems.append(fit_term)

    if len(quat_systems) > 0:
        raise NotImplementedError(
            f"No quaternary systems are implemented, but the following were passed: {quat_systems}."
        )

    return fits


def get_model(components, conditions, fits, model_type, pags):
    terms = list(fits.keys())
    assert "martensite_start" in terms, (
        "Don't have terms to fit the Ms temperatures for non-chemistry-dependent terms...try again."
    )

    conditions = conditions.copy()
    comp_dict = defaultdict(lambda: 0)
    for component in components:
        if v.X(component) in conditions.keys():
            comp_dict.update({component: conditions[v.X(component)]})

    comp_dict["FE"] = 1 - np.sum(
        [comp_dict[i] for i in list(comp_dict.keys())]
    )  # Fe will be dependent component in these conditions
    energy_barrier = 0
    contrib_dict = {}

    # PAGS energy contribution
    pags_energy = 0
    if pd.isnull(pags):
        pass
    elif model_type == "epsilon-mf":
        pags_energy = (
            mf_epsilon_PAGS_fits[0]
            * np.log(pags ** mf_epsilon_PAGS_fits[1]) ** mf_epsilon_PAGS_fits[2]
            + mf_epsilon_PAGS_fits[3]
        )
    elif model_type == "lath-mf":
        pags_energy = (
            mf_lath_PAGS_fits[0] * np.log(pags ** mf_lath_PAGS_fits[1]) + mf_lath_PAGS_fits[2]
        )
    elif model_type == "plate-mf":
        pags_energy = mf_plate_PAGS_fits[0] * np.log(pags) + mf_plate_PAGS_fits[1]
    contrib_dict["PAGS"] = pags_energy
    energy_barrier += pags_energy

    for term in terms:
        fit = fits[term][0]
        splits = term.split("-")
        term_contrib = 0
        x_b = x_c = x_n = x_o = (
            0  # initialize interstitials for a purely substitutional fitting term
        )
        if term == "martensite_start":
            # Energy barrier for pure iron
            if model_type == "epsilon-mf":
                term_contrib = mf_epsilon_baseline_fits[0]
            if model_type == "lath-mf":
                term_contrib = mf_lath_baseline_fits[0]
            if model_type == "plate-mf":
                term_contrib = mf_plate_baseline_fits[0]
        elif len(splits) == 2:  # binary system, either Fe-X or X-Y
            if (
                any(map(lambda comp: comp in splits, interstitials)) and len(fit) > 1
            ):  # should evaluate to true if an interstitial element is being fit and order > 0
                x_b = comp_dict["B"]
                x_c = comp_dict["C"]
                x_n = comp_dict["N"]
                x_o = comp_dict["O"]
            if splits[0] == "Fe":  # Fe-X
                x_j = comp_dict[splits[1].upper()]
                term_contrib += np.sum(
                    np.fromiter(
                        [
                            fit[i] * (x_j) ** (i + 1) / (1 - x_b - x_c - x_n - x_o)
                            for i in range(len(fit))
                        ],
                        float,
                    )
                )
            elif splits[1] == "Fe":  # X-Fe
                x_i = comp_dict[splits[0].upper()]
                term_contrib += np.sum(
                    np.fromiter(
                        [
                            fit[i] * (x_i) ** (i + 1) / (1 - x_b - x_c - x_n - x_o)
                            for i in range(len(fit))
                        ],
                        float,
                    )
                )
            else:
                x_i = comp_dict[splits[0].upper()]
                x_j = comp_dict[splits[1].upper()]
                term_contrib += np.sum(
                    np.fromiter(
                        [
                            fit[i] * (x_i * x_j) ** (i + 1) / (1 - x_b - x_c - x_n - x_o)
                            for i in range(len(fit))
                        ],
                        float,
                    )
                )

        else:
            raise KeyError(f"Incorrect term: {term} provided to {model_type} model.")
        energy_barrier += term_contrib
        contrib_dict[term] = term_contrib

    return energy_barrier, contrib_dict


def predict_DG(row):
    if row["alloy_system"] == "all":  # already modeled
        return row["DG"]
    model_type = row["type"]
    components = ["FE", "VA"]
    conditions = row["conditions"]
    pags = np.nan  # PAGS not a factor in this function and should not affect the energy prediction

    for cond in list(conditions.keys()):
        # if conditions[cond] == 0:
        #     del conditions[cond]
        #     continue
        cond = cond.__str__()
        if "X" in cond:
            components = np.concatenate([components, [cond.split("_")[1]]])

    if model_type == "plate-mf":
        model_params = pd.read_json("".join([model_param_dir, "mf_plate_parameters.json"]))
        model_fits = model_params["mf_plate_fits"][0]
        model_DG, contrib_dict = get_model(components, conditions, model_fits, model_type, pags)
    elif model_type == "plate-storm":
        model_DG = get_plate_model_storm(components, conditions)
        contrib_dict = {}
    if model_type == "lath-mf":
        model_params = pd.read_json("".join([model_param_dir, "mf_lath_parameters.json"]))
        model_fits = model_params["mf_lath_fits"][0]
        model_DG, contrib_dict = get_model(components, conditions, model_fits, model_type, pags)
    if model_type == "lath-storm":
        model_DG = get_lath_model_storm(row["martensite_start"], components, conditions)
        contrib_dict = {}
    if model_type == "epsilon-mf":
        model_params = pd.read_json("".join([model_param_dir, "mf_epsilon_parameters.json"]))
        model_fits = model_params["mf_epsilon_fits"][0]
        model_DG, contrib_dict = get_model(components, conditions, model_fits, model_type, pags)

    print(f"{model_type} DG: {model_DG}, conditions: {conditions}")
    row["DG"] = model_DG
    row["contrib_dict"] = contrib_dict

    return row


def calc_model_Ms(row, db):
    if row["alloy_system"] == "all":
        return row["martensite_start"]  # already modeled
    ms_type = row["type"]
    conditions = row["conditions"]
    components = get_components_from_conditions(conditions, dependent_component="FE")
    components = ["FE", "VA"]
    pags = (
        np.nan
    )  # This function is for fitting the chemistry-dependent models, PAGS is not a factor here.

    for cond in list(conditions.keys()):
        if conditions[cond] == 0:
            del conditions[cond]
            print(f"Removing {cond.__str__()}")
            continue
        cond = cond.__str__()
        if "X" in cond:
            components = np.concatenate([components, [cond.split("_")[1]]])

    new_fits = None
    temp_bounds = Bounds(1e-6, 2500)
    pdens = 500
    if "epsilon" in ms_type:
        phases = ["HCP_A3", "FCC_A1"]
    else:
        phases = ["BCC_A2", "FCC_A1"]
    solved = False
    n = 0
    while not solved and n < 5:
        if "plate" in ms_type:
            if "storm" in ms_type:
                DG_calc = minimize(
                    get_model_Ms,
                    500,
                    args=("plate-storm", db, components, phases, conditions, pags, pdens, new_fits),
                    method="Nelder-Mead",
                    bounds=temp_bounds,
                )
            else:
                model_params = pd.read_json("".join([model_param_dir, "mf_plate_parameters.json"]))
                new_fits = model_params["mf_plate_fits"][0]
                DG_calc = minimize(
                    get_model_Ms,
                    500,
                    args=("plate-mf", db, components, phases, conditions, pags, pdens, new_fits),
                    method="Nelder-Mead",
                    bounds=temp_bounds,
                )
        elif "lath" in ms_type:
            if "storm" in ms_type:
                DG_calc = minimize(
                    get_model_Ms,
                    500,
                    args=("lath-storm", db, components, phases, conditions, pags, pdens, new_fits),
                    method="Nelder-Mead",
                    bounds=temp_bounds,
                )
            else:
                model_params = pd.read_json("".join([model_param_dir, "mf_lath_parameters.json"]))
                new_fits = model_params["mf_lath_fits"][0]
                DG_calc = minimize(
                    get_model_Ms,
                    500,
                    args=("lath-mf", db, components, phases, conditions, pags, pdens, new_fits),
                    method="Nelder-Mead",
                    bounds=temp_bounds,
                )
        elif "epsilon" in ms_type:  # no storm epsilon model
            model_params = pd.read_json("".join([model_param_dir, "mf_epsilon_parameters.json"]))
            new_fits = model_params["mf_epsilon_fits"][0]
            DG_calc = minimize(
                get_model_Ms,
                500,
                args=("epsilon-mf", db, components, phases, conditions, pags, pdens, new_fits),
                method="Nelder-Mead",
                bounds=temp_bounds,
            )
        else:
            return np.nan

        model_Ms = DG_calc.x[0]
        model_DG = DG_calc.fun
        if model_DG < 0.5 or model_Ms < 298.15:
            solved = True
        else:
            n += 1
            pdens += n * 500
    if model_DG > 0.5 or model_Ms == 500:  # Find where the solver is performing poorly
        print(f"Solver problem with: {ms_type}, {conditions}")
        print(f"Calculated Ms: {model_Ms}")
        print(f"Calculated DG: {model_DG}")
    print(f"{ms_type} Ms: {model_Ms}, conditions: {conditions}")
    return model_Ms


def fit_models(db, do_curve_fit=False, DG_refit=False):
    exp_data = pd.read_json("".join([exp_data_dir, "DG_calcs_with_gpc_type.json"]))
    exp_data = exp_data.query(
        "type != 'alpha' & not type.isnull() & type != 'mixed' & not DG.isnull() & predicted_type == False & ignore == False & martensite_start >= 298.15"
    ).copy()
    systems = ["Fe-"]
    for model_type in list(conc_arrays.keys()):
        for term in list(conc_arrays[model_type].keys()):
            term = term.split("-")
            if "Fe" in term:
                term.remove("Fe")
            term.sort()
            term = "Fe-" + "-".join(term)
            if term not in systems:
                systems.append(term)
    print(systems)
    exp_data = exp_data.query("alloy_system in @systems")
    exp_data.to_json("".join([model_param_dir, "martensite_experimental_model_training_data.json"]))
    save_training_data_json(exp_data)  # save as individual json files for ESPEI
    print("Training with ESPEI")
    train_espei()
    model_params = pd.DataFrame([])

    # Plate model
    plate_parameters = pd.DataFrame()
    old_plate_parameters = pd.read_json("".join([model_param_dir, "mf_plate_parameters.json"]))
    ignored_refs = list(
        itertools.chain(
            *[selected_ignore["plate"][key] for key in list(selected_ignore["plate"].keys())]
        )
    )
    mf_plate_data = exp_data.query("type == 'plate' & reference not in @ignored_refs").reset_index(
        drop=True
    )
    plate_terms = [term for term in list(conc_arrays["plate-mf"].keys())]
    plate_terms.append("martensite_start")
    plate_parameters["terms"] = [plate_terms]
    plate_parameters["orders"] = [plate_orders]

    assert all([term in list(plate_orders.keys()) for term in plate_terms]), (
        "MF-plate model terms do not system parameters passed."
    )
    mf_plate_fits = fit_model(plate_orders, mf_plate_data, "plate")
    plate_parameters["mf_plate_fits"] = [mf_plate_fits]
    plate_time = time.time()
    plate_parameters["model_version"] = plate_time
    print("Plate Model")
    print(mf_plate_fits)
    plate_parameters.to_json("".join([model_param_dir, "mf_plate_parameters.json"]))

    # Lath Model
    lath_parameters = pd.DataFrame()
    ignored_refs = list(
        itertools.chain(
            *[selected_ignore["lath"][key] for key in list(selected_ignore["lath"].keys())]
        )
    )
    mf_lath_data = exp_data.query("type == 'lath' & reference not in @ignored_refs").reset_index(
        drop=True
    )
    lath_terms = [term for term in list(conc_arrays["lath-mf"].keys())]
    lath_terms.append("martensite_start")
    lath_parameters["terms"] = [lath_terms]
    lath_parameters["orders"] = [lath_orders]

    assert all([term in list(lath_orders.keys()) for term in lath_terms])

    mf_lath_fits = fit_model(lath_orders, mf_lath_data, "lath")
    lath_parameters["mf_lath_fits"] = [mf_lath_fits]
    lath_time = time.time()
    print("Lath Model")
    print(mf_lath_fits)

    lath_parameters["model_version"] = lath_time
    lath_parameters.to_json("".join([model_param_dir, "mf_lath_parameters.json"]))

    # Epsilon model
    epsilon_parameters = pd.DataFrame()
    ignored_refs = list(
        itertools.chain(
            *[selected_ignore["epsilon"][key] for key in list(selected_ignore["epsilon"].keys())]
        )
    )
    mf_epsilon_data = exp_data.query(
        "type == 'epsilon' & reference not in @ignored_refs"
    ).reset_index(drop=True)
    epsilon_terms = [term for term in list(conc_arrays["epsilon-mf"].keys())]
    epsilon_terms.append("martensite_start")
    epsilon_parameters["terms"] = [epsilon_terms]

    epsilon_parameters["orders"] = [epsilon_orders]
    assert all([term in list(epsilon_orders.keys()) for term in epsilon_terms]), (
        "MF-epsilon model terms do not match parameters passed."
    )

    mf_epsilon_fits = fit_model(epsilon_orders, mf_epsilon_data, "epsilon")
    epsilon_parameters["mf_epsilon_fits"] = [mf_epsilon_fits]
    print("Epsilon Model")
    print(mf_epsilon_fits)

    epsilon_parameters["model_version"] = lath_time
    epsilon_parameters.to_json("".join([model_param_dir, "mf_epsilon_parameters.json"]))
    model_types = ["lath-mf", "plate-mf", "plate-storm", "lath-storm", "epsilon-mf"]

    if do_curve_fit:
        # model_data = pd.DataFrame([])
        if DG_refit:
            t0_data = pd.DataFrame([], columns=["alloy_system"])
        else:
            t0_data = pd.read_json("".join([model_param_dir, "t0_model.json"]))

        model_params["model_types"] = [model_types]
        model_params["baseline_temperatures"] = [
            np.linspace(
                np.min(exp_data["martensite_start"]), np.max(exp_data["martensite_start"]), 350
            )
        ]

        # Model baseline values that will be adjusted by RK polynomials of the alloying elements
        print("Modeling baseline")
        for temp in model_params["baseline_temperatures"][0]:
            lath_baseline = mf_lath_baseline_fits[0]
            # lath_baseline = mf_lath_baseline_fits[0] + mf_lath_baseline_fits[1]*temp
            local_dict = {
                "alloy_system": "all",
                "type": "lath-mf",
                "martensite_start": [temp],
                "DG": [lath_baseline],
                "conditions": [{v.N: 1, v.P: 101325}],
                "components": [["FE", "VA"]],
            }
            # model_data = pd.concat([model_data, pd.DataFrame(local_dict)])
            model_data = pd.DataFrame(local_dict)

            # plate_baseline = mf_plate_fits['martensite_start'][0][0]
            plate_baseline = mf_plate_baseline_fits[0]
            local_dict.update({"type": "plate-mf", "DG": [plate_baseline]})
            model_data = pd.concat([model_data, pd.DataFrame(local_dict)])

            # epsilon_baseline = mf_epsilon_fits['martensite_start'][0][0] + mf_epsilon_fits['martensite_start'][0][1]*temp
            # epsilon_baseline = mf_epsilon_fits['martensite_start'][0][0]
            epsilon_baseline = mf_epsilon_baseline_fits[0]
            # epsilon_baseline = mf_epsilon_baseline_fits[0] + mf_epsilon_baseline_fits[1]*temp + mf_epsilon_baseline_fits[2]*temp**2
            local_dict.update({"type": "epsilon-mf", "DG": [epsilon_baseline]})
            model_data = pd.concat([model_data, pd.DataFrame(local_dict)])

        print("Modeling systems")
        for model_type in list(conc_arrays.keys()):
            if "storm" in model_type and DG_refit == False:
                continue
            systems = list(conc_arrays[model_type])
            for system in systems:
                split_system = system.split("-")
                ordered_system = system.split("-")
                # if model_type == 'epsilon-mf' and 'Mn' not in split_system:
                #     ordered_system.append('Mn')
                if "Fe" in ordered_system:
                    ordered_system.remove("Fe")
                ordered_system.sort()
                ordered_system = "-".join(ordered_system)
                ordered_system = "Fe-" + ordered_system
                components = ["FE", "VA"]
                if split_system[0] == "Fe":
                    element0 = split_system[1]
                    components.append(element0.upper())
                    conc_array0 = conc_arrays[model_type][system]
                    conc_array1 = [0]
                    element1 = None
                else:
                    element0 = split_system[0]
                    components.append(element0.upper())
                    conc_array0 = conc_arrays[model_type][system][0]
                    element1 = split_system[1]
                    components.append(element1.upper())
                    conc_array1 = conc_arrays[model_type][system][1]

                model_dict = defaultdict(lambda: 0)
                conditions = {v.N: 1, v.P: 101325}
                for x in conc_array0:
                    conditions.update({v.X(element0.upper()): x})
                    for y in conc_array1:
                        if element1 == None:
                            model_dict["Fe"] = 1 - x
                        else:
                            conditions.update({v.X(element1.upper()): y})
                        model_dict["alloy_system"] = ordered_system
                        model_dict[element0] = [x]
                        model_dict[element1] = [y]
                        model_dict["conditions"] = [conditions.copy()]
                        model_dict["database"] = [dbf]
                        model_dict["type"] = model_type
                        model_dict["components"] = [components]
                        model_data = pd.concat([model_data, pd.DataFrame(model_dict)])
                        if (
                            model_type == "epsilon-mf" or model_type == "lath-mf"
                        ):  # populate T0 temperatures for the two types that are needed
                            model_dict["type"] = "T0-" + model_type.split("-")[0]
                            t0_data = pd.concat([t0_data, pd.DataFrame(model_dict)])

        model_data = model_data.reset_index(drop=True)
        print(model_data["alloy_system"].unique())
        data_to_fit = model_data.query("alloy_system != 'all'")
        model_data = model_data.drop(data_to_fit.index)
        data_to_fit.loc[:, "martensite_start"] = data_to_fit.parallel_apply(
            lambda row: calc_model_Ms(row.copy(), db), axis=1
        )
        data_to_fit = data_to_fit.parallel_apply(lambda row: predict_DG(row.copy()), axis=1)
        model_data = pd.concat([model_data, data_to_fit]).fillna(0)
        model_data["projected_point"] = False
        model_params.to_json("".join([model_param_dir, "martensite_model_parameters.json"]))
        mf_model_data = model_data.query(
            "type == 'lath-mf' or type == 'plate-mf' or type == 'epsilon-mf'"
        )
        mf_model_data.to_json("".join([model_param_dir, "martensite_model.json"]))
        if DG_refit:
            print("Fitting Stormvinter's model")
            storm_model_data = model_data.query("type == 'lath-storm' or type == 'plate-storm'")
            storm_model_data.to_json(
                "".join([model_param_dir, "stormvinter_martensite_model.json"])
            )
            print("Recalculating T0 temperatures")
            t0_data = t0_data.parallel_apply(lambda row: calc_t0(row.copy(), db), axis=1)
            t0_data = t0_data.reset_index(drop=True)
            t0_data.to_json("".join([model_param_dir, "t0_model.json"]))
        print(model_data["alloy_system"].unique())


def model_pags(db):
    # Function to model the effect of PAGS.
    model_types = ["plate", "lath", "epsilon"]
    exp_data = (
        pd.read_json("".join([exp_data_dir, "DG_calcs_with_gpc_type.json"]))
        .query("not PAGS.isnull() & type in @model_types")
        .copy()
    )
    exp_data["exp_PAGS"] = exp_data[
        "PAGS"
    ]  # temporarily store the experimental PAGS to prevent them from being included in the calculation
    exp_data["PAGS"] = np.nan
    exp_data = exp_data.parallel_apply(lambda row: predict_Ms(row, db), axis=1)
    exp_data["PAGS_ms_increment"] = exp_data["martensite_start"] - exp_data["mf_martensite_start"]
    exp_data["offset_DF"] = exp_data.parallel_apply(lambda row: calc_pags_df(row, db), axis=1)
    exp_data.to_json("".join([exp_data_dir, "exp_data_with_pags.json"]))


def calc_pags_df(row, db):
    conditions = convert_conditions(row["pred_martensite_conditions"])
    components = get_components_from_conditions(conditions, dependent_component="FE")
    conditions.update({v.T: row["mf_martensite_start"]})
    if "epsilon" in row["type"]:
        phases = ["HCP_A3", "FCC_A1"]
    else:
        phases = ["BCC_A2", "FCC_A1"]
    DF_no_PAGS = -DG(db, components, phases, conditions)

    conditions.update({v.T: row["martensite_start"]})
    DF_with_PAGS = -DG(db, components, phases, conditions)

    return DF_with_PAGS - DF_no_PAGS


def project_models():
    # interpolate and project models below 298 K
    plate_params = pd.read_json("".join([model_param_dir, "mf_plate_parameters.json"]))
    plate_terms = plate_params["terms"][0]
    lath_params = pd.read_json("".join([model_param_dir, "mf_lath_parameters.json"]))
    lath_terms = lath_params["terms"][0]
    epsilon_params = pd.read_json("".join([model_param_dir, "mf_epsilon_parameters.json"]))
    epsilon_terms = epsilon_params["terms"][0]

    # generic 0th-order RK expansion for Fe-X systems
    def bin_fit_func1(x, C0, C1):
        return C0 + x * C1

    # generic 1st-order RK expansion for Fe-X systems
    def bin_fit_func2(x, C0, C1, C2):
        return C0 + x * (C1 + x * C2)

    # generic RK expansions for Fe-X-Y systems
    def bin_fit_func3(x, C0, C1):
        return C0 + x[0] * x[1] * C1

    def bin_fit_func4(x, C0, C1, C2):
        return C0 + x[0] * x[1] * (C1 + C2 * (x[0] - x[1]))

    mf_model_data = pd.read_json("".join([model_param_dir, "martensite_model.json"])).fillna(0)
    storm_model_data = pd.read_json(
        "".join([model_param_dir, "stormvinter_martensite_model.json"])
    ).fillna(0)
    t0_data = pd.read_json("".join([model_param_dir, "t0_model.json"]))
    model_data = pd.concat([mf_model_data, storm_model_data, t0_data])
    model_data = model_data.query(
        "projected_point == False"
    )  # drop old projected points before refitting
    sub_data = model_data.query("alloy_system != 'all'")

    for model_type in sub_data["type"].unique():
        model_type_data = sub_data.query("type == @model_type")

        for system in model_type_data["alloy_system"].unique():
            if "epsilon" in model_type:
                orders = epsilon_orders
            elif "lath" in model_type:
                orders = lath_orders
            else:
                orders = plate_orders

            bad_points = []
            model_system_data = model_type_data.query("alloy_system == @system")
            model_dict = defaultdict(lambda: 0)
            conditions = {v.N: 1, v.P: 101325}
            split_system = system.split("-")
            bad_points = model_system_data.query("martensite_start < 300")
            binary = False
            if len(bad_points) > 0:  # find and remove bad points
                model_system_data = model_system_data.drop(bad_points.index)
                model_data = model_data.drop(bad_points.index)
            if len(split_system) == 2:
                x_i = "Fe"
                x_j = split_system[1]
                order_system = system
                binary = True
            if len(split_system) == 3:  # Fe-X-Y ternary systems
                split_system.remove("Fe")
                x_i = split_system[0]
                x_j = split_system[1]
                order_system = "-".join(split_system)
            order = orders[order_system]
            if "Fe" in split_system:
                solute_range = [np.min(model_system_data[x_j]), np.max(model_system_data[x_j])]
                solute_arr = np.linspace(0, solute_range[1] * 1.2)
                if len(model_system_data["DG"]) <= 1:
                    print(f"Cannot model {system}, {model_type}. Not enough points.")
                    continue
                elif (
                    len(model_system_data["DG"]) > 3 and order > 0
                ):  # can't fit a 3-parameter curve to a 2-point system, which only affects some systems
                    DG_fit = curve_fit(
                        bin_fit_func2, model_system_data[x_j], model_system_data["DG"]
                    )
                    ms_fit = curve_fit(
                        bin_fit_func2, model_system_data[x_j], model_system_data["martensite_start"]
                    )
                else:
                    DG_fit = curve_fit(
                        bin_fit_func1, model_system_data[x_j], model_system_data["DG"]
                    )
                    ms_fit = curve_fit(
                        bin_fit_func1, model_system_data[x_j], model_system_data["martensite_start"]
                    )
                for x_sol in solute_arr:
                    if (
                        len(model_system_data["DG"]) > 3 and order > 0
                    ):  # can't fit a 3-parameter curve to a 2-point system, which only affects some systems
                        projected_DG = bin_fit_func2(x_sol, *DG_fit[0])
                        projected_ms = bin_fit_func2(x_sol, *ms_fit[0])
                    else:
                        projected_DG = bin_fit_func1(x_sol, *DG_fit[0])
                        projected_ms = bin_fit_func1(x_sol, *ms_fit[0])
                    if projected_ms < 0:
                        continue
                    conditions.update({v.X(x_j.upper()): x_sol})
                    model_dict["alloy_system"] = system
                    model_dict["Fe"] = [1 - x_sol]
                    model_dict[x_j] = [x_sol]
                    model_dict["conditions"] = [conditions.copy()]
                    model_dict["database"] = [dbf]
                    model_dict["type"] = model_type
                    model_dict["projected_point"] = True
                    model_dict["DG"] = projected_DG
                    model_dict["martensite_start"] = projected_ms
                    model_data = pd.concat([model_data, pd.DataFrame(model_dict)])
            else:
                print(system)
                x_i_range = [np.min(model_system_data[x_i]), np.max(model_system_data[x_i])]
                x_i_arr = np.linspace(0, x_i_range[1])
                x_j_range = [np.min(model_system_data[x_j]), np.max(model_system_data[x_j])]
                x_j_arr = np.linspace(0, x_j_range[1])

                try:
                    if (
                        len(model_system_data["DG"]) > 3 and order > 0
                    ):  # can't fit a 3-parameter curve to a 2-point system, which only affects some systems
                        DG_fit = curve_fit(
                            bin_fit_func3,
                            np.vstack([model_system_data[x_i], model_system_data[x_j]]),
                            model_system_data["DG"],
                        )
                        ms_fit = curve_fit(
                            bin_fit_func3,
                            np.vstack([model_system_data[x_i], model_system_data[x_j]]),
                            model_system_data["martensite_start"],
                        )
                    else:
                        DG_fit = curve_fit(
                            bin_fit_func4,
                            np.vstack([model_system_data[x_i], model_system_data[x_j]]),
                            model_system_data["DG"],
                        )
                        ms_fit = curve_fit(
                            bin_fit_func4,
                            np.vstack([model_system_data[x_i], model_system_data[x_j]]),
                            model_system_data["martensite_start"],
                        )
                    for i in x_i_arr:
                        for j in x_j_arr:
                            if (
                                len(model_system_data["DG"]) > 3 and order > 0
                            ):  # can't fit a 3-parameter curve to a 2-point system, which only affects some systems
                                projected_DG = bin_fit_func3([i, j], *DG_fit[0])
                                projected_ms = bin_fit_func3([i, j], *ms_fit[0])
                            else:
                                projected_DG = bin_fit_func4([i, j], *DG_fit[0])
                                projected_ms = bin_fit_func4([i, j], *ms_fit[0])
                            if projected_ms < 0:
                                continue
                            conditions.update({v.X(x_i.upper()): i, v.X(x_j.upper()): j})
                            model_dict["alloy_system"] = system
                            model_dict["Fe"] = [1 - i - j]
                            model_dict[x_i] = [i]
                            model_dict[x_j] = [j]
                            model_dict["conditions"] = [conditions.copy()]
                            model_dict["database"] = [dbf]
                            model_dict["type"] = model_type
                            model_dict["projected_point"] = True
                            model_dict["DG"] = projected_DG
                            model_dict["martensite_start"] = projected_ms
                except Exception as e:
                    print(e)
                    print(f"Skipping projection of {system} due to fitting failure")
                    continue

    storm_model_data = model_data.query("type == 'lath-storm' or type == 'plate-storm'")
    model_data = model_data.query(
        "type == 'lath-mf' or type == 'plate-mf' or type == 'epsilon-mf'"
    )  # drop the storm model info
    model_data = model_data.reset_index(drop=True)
    storm_model_data = storm_model_data.reset_index(drop=True)
    model_data.to_json("".join([model_param_dir, "martensite_model.json"]))
    storm_model_data.to_json("".join([model_param_dir, "stormvinter_martensite_model.json"]))


def make_plots():
    sns.color_palette("viridis", as_cmap=True)
    exp_data = (
        pd.read_json("".join([model_param_dir, "martensite_experimental_model_training_data.json"]))
        .query("ignore == False & predicted_type == False")
        .sort_values(by="type")
    )
    mf_model_data = pd.read_json("".join([model_param_dir, "martensite_model.json"]))
    storm_model_data = pd.read_json("".join([model_param_dir, "stormvinter_martensite_model.json"]))
    t0_data = pd.read_json("".join([model_param_dir, "t0_model.json"]))
    model_data = pd.concat([mf_model_data, storm_model_data, t0_data]).sort_values(by="type")

    system_model_data = model_data.query("alloy_system == 'all'")
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(data=exp_data, x="martensite_start", y="DG", hue="type", ax=ax)
    sns.lineplot(data=system_model_data, x="martensite_start", y="DG", hue="type", ax=ax)
    # ax.set_title("Driving Force versus Ms Temperature")
    ax.set_xlabel("Martensite Start (K)")
    ax.set_ylabel(
        r"$\Delta G_\mathrm{m}^{x \rightarrow \gamma} \quad \left(\mathrm{J} \thickspace \mathrm{mol}^{-1} \right)$"
    )
    fig.tight_layout()
    fig.savefig("".join([figure_dir, "all_Ms.png"]))
    plt.close()

    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(data=exp_data, x="martensite_start", y="DG", hue="alloy_system", ax=ax)
    fig.tight_layout()
    fig.savefig("".join([figure_dir, "all_DG.png"]))
    plt.close()

    plate_params = pd.read_json("".join([model_param_dir, "mf_plate_parameters.json"]))
    plate_terms = plate_params["terms"][0]
    lath_params = pd.read_json("".join([model_param_dir, "mf_lath_parameters.json"]))
    lath_terms = lath_params["terms"][0]
    epsilon_params = pd.read_json("".join([model_param_dir, "mf_epsilon_parameters.json"]))
    epsilon_terms = epsilon_params["terms"][0]
    all_terms = np.concatenate([plate_terms, lath_terms, epsilon_terms])

    all_terms = [term for term in list(dict.fromkeys(all_terms)) if term != "martensite_start"]
    color_dict = {
        "lath": "blue",
        "lath-mf": "blue",
        "lath-storm": "blue",
        "plate": "orange",
        "plate-mf": "orange",
        "plate-storm": "orange",
        "epsilon": "green",
        "epsilon-mf": "green",
        "T0-alpha": "black",
        "T0-epsilon": "black",
    }

    for term in all_terms:
        models = []
        split_term = term.split("-")
        if "Fe" in split_term:  # Fe-X systems
            x = split_term[1]
            hue = "type"
            systems = [term]
            if term in epsilon_terms and term != "Fe-Mn":
                mn_term = term + "-Mn"
                mn_term = mn_term.split("-")
                mn_term.sort()
                mn_term = "-".join(mn_term)
                systems.append(mn_term)
        else:
            system_term = term.split("-")
            x = system_term[1]
            hue = system_term[0]
            system_term.sort()
            system_term = "-".join(system_term)
            system_term = "Fe-" + system_term
            systems = [system_term]
            if term in lath_terms:
                models.append("lath-mf")
            if term in plate_terms:
                models.append("plate-mf")
            if term in epsilon_terms:
                models.append("epsilon-mf")

        print(systems)

        system_exp_data = exp_data.query("alloy_system in @systems")

        if "Fe" in split_term:  # only Fe-X binary systems, can include all models on one plot
            for martensite_type in ["lath-plate", "epsilon"]:
                if martensite_type == "epsilon":
                    exp_type_query = ["epsilon"]
                    model_DG_query = ["epsilon-mf"]
                    model_Ms_query = ["epsilon-mf", "T0-epsilon"]
                    label_var = "\\epsilon"
                else:
                    exp_type_query = ["lath", "plate"]
                    model_DG_query = ["lath-mf", "plate-mf", "lath-storm", "plate-storm"]
                    model_Ms_query = [
                        "lath-mf",
                        "plate-mf",
                        "T0-alpha",
                        "lath-storm",
                        "plate-storm",
                    ]
                    label_var = "\\alpha"
                model_exp_data = system_exp_data.query("type in @exp_type_query")
                if (
                    len(model_exp_data) == 0
                ):  # system not modeled for this model type if this evaluates to true
                    continue

                model_projected_data = model_data.query(
                    "alloy_system in @systems & projected_point == True & type in @model_DG_query"
                )
                fig, ax = plt.subplots(figsize=figsize)
                sns.scatterplot(
                    data=model_exp_data, x=x, y="DG", hue="type", ax=ax, style="reference", s=50
                )
                sns.lineplot(data=model_projected_data, x=x, y="DG", hue="type", ax=ax)
                ax.set_xlabel(r"$x_{sol}$".replace("sol", x))
                ax.set_ylabel(
                    r"$\Delta G_\mathrm{m}^{var \rightarrow \gamma} \quad \left(\mathrm{J} \thickspace \mathrm{mol}^{-1} \right)$".replace(
                        "var", label_var
                    )
                )

                fig.tight_layout()
                fig.savefig("".join([figure_dir, term, "_", martensite_type, "_DG.png"]))
                plt.close()

                model_projected_data = model_data.query(
                    "alloy_system in @systems & projected_point == True & type in @model_Ms_query"
                )
                fig, ax = plt.subplots(figsize=figsize)
                sns.scatterplot(
                    data=model_exp_data,
                    x=x,
                    y="martensite_start",
                    hue="type",
                    ax=ax,
                    style="reference",
                    s=50,
                )
                sns.lineplot(
                    data=model_projected_data, x=x, y="martensite_start", hue="type", ax=ax
                )
                ax.set_xlabel(r"$x_{sol}$".replace("sol", x))
                ax.set_ylabel("Martensite Start Temperature (K)")
                # ax.set_title(f"Martensite Start Temperature for {systems[-1]} System")
                fig.tight_layout()
                fig.savefig("".join([figure_dir, term, "_", martensite_type, "_Ms.png"]))
                plt.close()
        else:  # other interacting systems, need to graph differently
            for model in models:
                print(model)
                if "plate" in model:
                    model_type = "plate"
                    model_Ms_query = ["plate-mf", "T0-alpha"]
                    label_var = "\\alpha"
                if "lath" in model:
                    model_type = "lath"
                    model_Ms_query = ["lath-mf", "T0-alpha"]
                    label_var = "\\alpha"
                if "epsilon" in model:
                    model_type = "epsilon"
                    model_Ms_query = ["epsilon-mf", "T0-epsilon"]
                    label_var = "\\epsilon"
                if x == "Mn" or x == "Ni":  # want to switch axes
                    old_hue = hue
                    hue = x
                    x = old_hue

                model_exp_data = system_exp_data.query("type == @model_type")
                if (
                    len(model_exp_data) == 0
                ):  # system not modeled for this model type if this evaluates to true
                    continue
                # system_model_data = model_data.query(f"type in @model_Ms_query & alloy_system in @systems")
                system_model_data = model_data.query(f"type == @model & alloy_system in @systems")
                fig, ax = plt.subplots(figsize=figsize)
                sns.scatterplot(data=model_exp_data, x=x, y="martensite_start", hue=hue, ax=ax)
                sns.lineplot(data=system_model_data, x=x, y="martensite_start", hue=hue, ax=ax)
                ax.set_xlabel(r"$x_{sol}$".replace("sol", x))
                ax.set_ylabel("Martensite Start Temperature (K)")
                # ax.set_title(f"{term} Martensite Start Temperature for MF {model_type.capitalize()} Martensite Model")
                fig.tight_layout()
                fig.savefig("".join([figure_dir, term, "_", model, "_Ms.png"]))
                plt.close()

                fig, ax = plt.subplots(figsize=figsize)
                sns.scatterplot(data=model_exp_data, x=x, y="DG", hue=hue, ax=ax)
                sns.lineplot(data=system_model_data, x=x, y="DG", hue=hue, ax=ax)
                ax.set_xlabel(r"$x_{sol}$".replace("sol", x))
                ax.set_ylabel(
                    r"$\Delta G_\mathrm{m}^{var \rightarrow \gamma} \quad \left(\mathrm{J} \thickspace \mathrm{mol}^{-1} \right)$".replace(
                        "var", label_var
                    )
                )
                # ax.set_title(f"{term} Driving Force for MF {model_type.capitalize()} Martensite Model")
                fig.tight_layout()
                fig.savefig("".join([figure_dir, term, "_", model, "_DG.png"]))
                plt.close()

                # make contour plots
                crange = [
                    np.min(np.concatenate([system_model_data["DG"], model_exp_data["DG"]])),
                    np.max(np.concatenate([system_model_data["DG"], model_exp_data["DG"]])),
                ]
                xrange = [
                    np.min(np.concatenate([system_model_data[x], model_exp_data[x]])),
                    np.max(np.concatenate([system_model_data[x], model_exp_data[x]])),
                ]
                cmid = np.mean(crange)
                cbar_dict = dict(
                    # title=r"$ \Delta g \frac{\mathrm{J}}{\mathrm{mol}}$ \n",
                    titleside="top",
                    tickmode="array",
                    tickvals=[crange[0], cmid, crange[1]],
                    ticks="outside",
                )
                dpi = plt.rcParams["figure.dpi"]
                contour = go.Contour(
                    x=system_model_data[x],
                    y=system_model_data[hue],
                    z=system_model_data["DG"],
                    line_smoothing=1,
                    zmin=crange[0],
                    zmax=crange[1],
                    colorbar=cbar_dict,
                    colorscale="bluered",
                )
                scatter = go.Scatter(
                    x=model_exp_data[x],
                    y=model_exp_data[hue],
                    mode="markers",
                    marker=dict(
                        size=16,
                        color=model_exp_data["DG"],
                        cmin=crange[0],
                        cmax=crange[1],
                        colorbar=cbar_dict,
                        colorscale="bluered",
                        showscale=False,
                        line=dict(color="White", width=2),
                    ),
                )
                data = [contour, scatter]
                fig = go.Figure(data=data)
                fig.update_layout(  # title_text=f"{term} Driving Force Contour Plot for MF {model_type.capitalize()} Martensite Model",
                    autosize=False,
                    width=12 * dpi,
                    height=8 * dpi,
                    font=dict(size=19.2),
                )
                fig.update_xaxes(
                    range=xrange,
                    title_text=r"$x_\mathrm{sol}$".replace("sol", x),
                    title_font_size=24,
                )
                fig.update_yaxes(
                    range=[
                        np.min(np.concatenate([system_model_data[hue], model_exp_data[hue]])),
                        np.max(np.concatenate([system_model_data[hue], model_exp_data[hue]])),
                    ],
                    title_text=r"$x_\mathrm{hue}$".replace("hue", hue),
                    title_font_size=24,
                )
                plt.xlabel(r"$x_\mathrm{hue}$".replace("hue", hue), size=19.2)
                fig.write_image("".join([figure_dir, term, "_", model, "_contour_DG.png"]))

    # Make PAGS plots
    pags_exp_data = pd.read_json("".join([exp_data_dir, "predicted_martensite_start.json"]))

    ## Lath
    def lath_fit(x, a, b, c):
        return a * np.log(x**b) + c

    ignored_references = ["Harmelin1982"]  # Harmelin1982 only has a single point
    fig_query = pags_exp_data.query(
        "~PAGS.isnull() & type == 'lath' & predicted_type == False & reference not in @ignored_references"
    ).copy()
    fig_query["offset_DG"] = 0.0

    fig, ax = plt.subplots(figsize=figsize)
    for reference in fig_query["reference"].unique():
        if reference in ignored_references:
            continue
        fig_sub_query = fig_query.query("reference == @reference").copy()
        fig_sub_query["offset_DG"] = fig_sub_query["DG_no_PAGS"] - np.min(
            fig_sub_query["DG_no_PAGS"]
        )
        fig_query.loc[fig_sub_query.index, "offset_DG"] = fig_sub_query["offset_DG"]
    # fig_query = fig_query.query("offset_DG < 400").copy()
    fig_query["offset_DG"] = (
        fig_query["offset_DG"] + 30
    )  # add 30 to make intercept at x=100 um ~ 0 J/mol
    sns.scatterplot(data=fig_query, x="PAGS", y="offset_DG", hue="reference", ax=ax, s=50)
    x_vals = np.linspace(1e-1, 150)
    fits = curve_fit(
        lath_fit, xdata=fig_query["PAGS"], ydata=fig_query["offset_DG"], bounds=(-50, np.inf)
    )  # Bounds needed for fit
    print(f"Lath PAGS fits: {fits[0]}")
    y_vals = [lath_fit(x, *fits[0]) for x in x_vals]
    y_vals = [lath_fit(x, *mf_lath_PAGS_fits) for x in x_vals]
    ax.plot(x_vals, y_vals)

    ax.set_xlabel(r"Prior-Austenite Grain Size ($\mu$m)")
    ax.set_ylabel(
        r"$\Delta G_\mathrm{m,lath}^{\alpha \rightarrow \gamma} \quad \left(\mathrm{J} \thickspace \mathrm{mol}^{-1} \right)$"
    )
    fig.tight_layout()
    fig.savefig("".join([figure_dir, "mf-lath-pags-model.png"]))
    plt.close()

    ## Plate

    def plate_fit(x, a, b):
        return a * np.log(x) + b

    han_data = pd.read_excel(
        "".join([exp_data_dir, "./Hanumantharaju2018-PAGS.xlsx"])
    )  # Include the data from [1] for now
    han_data["offset_DG"] = han_data["DG"]

    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(data=han_data, x="PAGS", y="offset_DG", hue="reference", ax=ax, s=50)
    fits = curve_fit(plate_fit, han_data["PAGS"], han_data["offset_DG"], bounds=(-100, np.inf))
    print(f"Plate PAGS fits: {fits[0]}")
    x_vals = np.linspace(1e-2, np.max(han_data["PAGS"]))
    y_vals = [plate_fit(x, *mf_plate_PAGS_fits) for x in x_vals]
    ax.plot(x_vals, y_vals)

    ax.set_xlabel(r"Prior-Austenite Grain Size ($\mu$m)")
    ax.set_ylabel(
        r"$\Delta G_\mathrm{m,plate}^{\alpha \rightarrow \gamma} \quad \left(\mathrm{J} \thickspace \mathrm{mol}^{-1} \right)$"
    )
    fig.tight_layout()
    fig.savefig("".join([figure_dir, "./mf-plate-pags-model.png"]))
    plt.close()

    ## Epsilon

    def eps_fit(x, a, b, c, d):
        return a * np.log(x**b) ** c + d

    fig_query = (
        pags_exp_data.query("~PAGS.isnull() & type == 'epsilon' & predicted_type == False")
        .sort_values("error", ascending=False)
        .copy()
    )
    fig_query["offset_DG"] = 0.0
    fig, ax = plt.subplots(figsize=figsize)
    for reference in fig_query["reference"].unique():
        fig_sub_query = fig_query.query("reference == @reference").copy()
        fig_sub_query["offset_DG"] = fig_sub_query["DG_no_PAGS"] - np.min(
            fig_sub_query["DG_no_PAGS"]
        )
        fig_query.loc[fig_sub_query.index, "offset_DG"] = fig_sub_query["offset_DG"]
    sns.scatterplot(data=fig_query, x="PAGS", y="offset_DG", hue="reference", ax=ax, s=50)
    fits = curve_fit(eps_fit, fig_query["PAGS"], fig_query["offset_DG"])
    print(f"Epsilon PAGS fits: {fits[0]}")
    x_vals = np.linspace(np.min(fig_query["PAGS"]), np.max(fig_query["PAGS"]))
    y_vals = [eps_fit(x, *mf_epsilon_PAGS_fits) for x in x_vals]
    ax.plot(x_vals, y_vals)

    ax.set_xlabel(r"Prior-Austenite Grain Size ($\mu$m)")
    ax.set_ylabel(
        r"$\Delta G_\mathrm{m}^{\epsilon \rightarrow \gamma} \quad \left(\mathrm{J} \thickspace \mathrm{mol}^{-1} \right)$"
    )
    fig.tight_layout()
    fig.savefig("".join([figure_dir, "./mf-epsilon-pags-model.png"]))
    plt.close()


def load_exp_data():
    in_files = [
        "".join([exp_data_dir, "martensite_start.ods"]),
        "".join([exp_data_dir, "bhadeshia_data.ods"]),
    ]
    elements = [
        "Al",
        "B",
        "C",
        "Ce",
        "Co",
        "Cr",
        "Cu",
        "Fe",
        "Ge",
        "H",
        "Hf",
        "Mn",
        "Mo",
        "N",
        "Nb",
        "Ni",
        "O",
        "P",
        "Pb",
        "S",
        "Si",
        "Ta",
        "Ti",
        "V",
        "W",
        "Zr",
    ]
    data = pd.concat([pd.read_excel(file) for file in in_files])
    data = data.query("ignore == False & martensite_start > 299 & type != 'mixed'").copy()
    data = data.apply(lambda row: parse_composition(row, "FE"), axis=1)
    data["original_components"] = data.apply(lambda row: row["components"], axis=1)
    data["original_conditions"] = data.apply(lambda row: row["conditions"], axis=1)
    for element in elements:  # replace NaNs with 0.0 for elemental columns
        data[element] = data[element].fillna(0)

    # drop columns that aren't needed for future calculations
    drop_cols = [
        "LCL",
        "UCL",
        "comment",
        "comments",
        "composition",
        "figure_no",
        "heating_rate",
        "material_at%",
        "material_wt%",
        "method",
        "system",
        "verified",
    ]
    data = data.drop(drop_cols, axis=1)
    data = data.reset_index()
    data.to_json("".join([exp_data_dir, "cleaned_exp_data.json"]))


def calc_DG(db, DG_refit):
    """
    Function to calculate the energy barrier for the martensitic transformation. This is given by the calculated
    Gibbs free energy difference between the FCC and BCC or HCP phases.

    To account for relatively high-temperature precipitates that may be present, the equilibrium composition of the FCC phase
    is calculated at the given (or assumed) solutionizing temperature and used for the subsequent calculations.
    """

    print("Predicting austenite composition at austenitizing temperature")
    data = pd.read_json("".join([exp_data_dir, "gpc_predicted_types.json"]))
    data["calc_error"] = False
    data["error_message"] = ""
    print(f"Removing components not included in the fitting.")
    data = data.apply(lambda row: get_conditions(row), axis=1)
    temp_df = []
    for reference in data["reference"].unique():
        sub_df = data.query("reference == @reference").copy()
        if len(sub_df) == 0:
            continue
        sub_df = sub_df.parallel_apply(lambda row: calc_aus_comp(row, db), axis=1)
        sub_df["non_FCC_solution_phases"] = sub_df.apply(
            lambda row: np.sum(
                [phase != "FCC_A1" and phase != "" for phase in row["solutionizing_phases"]]
            ),
            axis=1,
        )
        sub_df = sub_df.parallel_apply(lambda row: get_dG(row, db), axis=1)
        if len(temp_df) == 0:
            temp_df = sub_df
        else:
            temp_df = pd.concat([temp_df, sub_df])
    # data = data.parallel_apply(lambda row: calc_aus_comp(row, db), axis=1)
    data = temp_df.copy()
    if DG_refit:
        data["new_DG"] = True
    else:
        old_data = pd.read_json("".join([exp_data_dir, "DG_calcs.json"]))
        data["DG_version"] = old_data["DG_version"]  # initialize the version
        old_data = old_data.drop(
            [col for col in old_data.columns if col not in data.columns], axis=1
        )  # drop other columns that are added in subsequent calculations

        comp_DG = data.compare(old_data)["DG"]  # find indices that have different DG calcs
        data["new_DG"] = False  # initialize a value
        data.loc[comp_DG.index, "new_DG"] = True  # set different DG values to True
    data.loc[:, "DG_version"] = data.apply(
        lambda row: time.time() if row["new_DG"] == True else row["DG_version"], axis=1
    )  # update DG version if different calculated DG
    # Remove NaNs from DG and save to separate file for adjudication
    calc_data = data.query("not DG.isnull()")
    calc_data.to_json("".join([exp_data_dir, "DG_calcs_with_gpc_type.json"]))
    nan_data = data.query("DG.isnull()")
    nan_data.to_json("".join([exp_data_dir, "nan_DG_calcs.json"]))


def predict_type():
    # Predict missing types
    print("Classifying missing martensite start types using GPC model.")
    data = pd.read_json("".join([exp_data_dir, "cleaned_exp_data.json"]))
    data["martensite_start"] = (
        data["martensite_start"] / 1000
    )  # transform to be similar magnitude to composition fractions
    exp_data = data.query("not type.isnull()").copy()
    exp_data["type"] = exp_data.apply(
        lambda row: row["type"] if row["type"] != "alpha" else "non-martensite", axis=1
    )
    classified_data = exp_data.query(
        "type == 'lath' | type == 'plate' | type == 'epsilon' | type == 'non-martensite'"
    ).copy()
    exp_data["predicted_type"] = False

    drop_cols = [
        "austenitization_temperature",
        "PAGS",
        "reference",
        "ignore",
        "alloy_system",
        "components",
        "conditions",
        "parent",
        "index",
        "original_components",
        "original_conditions",
    ]
    X = classified_data.drop(drop_cols, axis=1)
    # For now the StandardScaler is reducing performance.
    # X = pd.DataFrame(scale.fit_transform(X[X.columns]), columns=X.columns)
    X_train = X.sample(frac=0.8, random_state=654321)
    X_test = X.drop(X_train.index)
    Y_train = X_train.loc[:, "type"]
    Y_test = X_test.loc[:, "type"]
    X_train = X_train.drop("type", axis=1)
    X_test = X_test.drop("type", axis=1)
    classifier_kernel = 1**2 * RBF(length_scale=1)

    sns.set_theme("paper", font_scale=2)
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=figsize)
    gpc = GaussianProcessClassifier(kernel=classifier_kernel)
    _ = gpc.fit(X_train, Y_train)
    R_squared = gpc.score(X_test, Y_test)
    print(gpc.kernel_.get_params())
    print(
        f"{len(Y_train)} data points used for GPC training and {len(Y_test)} for testing. R^2: {str(R_squared)}"
    )
    con_matrix = ConfusionMatrixDisplay.from_estimator(gpc, X_test, Y_test, ax=ax)
    ax.set_xlabel("Predicted Type")
    ax.set_ylabel("Actual Type")
    fig.tight_layout()
    fig.savefig("/".join([figure_dir, "gpc_confusion_matrix.png"]))
    pd.DataFrame(
        {
            "gp_classifier": [gpc],
            "training_data": len(Y_train),
            "testing_data": len(Y_test),
            "R^2": R_squared,
            "version": time.time(),
        }
    ).to_pickle("./gp_classifier_model.pkl")

    sns.set_style("whitegrid")
    data_to_classify = data.query("type.isnull()").copy()
    data_to_classify = data_to_classify.drop(drop_cols, axis=1)
    data_to_classify = data_to_classify.drop("type", axis=1)
    full_data_to_classify = data.query("type.isnull()")
    data_to_classify["type"] = gpc.predict(data_to_classify)
    data_to_classify["predicted_type"] = True
    for col in drop_cols:
        data_to_classify[col] = full_data_to_classify[col]

    data = pd.concat([exp_data, data_to_classify]).reset_index(drop=True)
    data["martensite_start"] = data["martensite_start"] * 1000
    data["type"] = data.apply(
        lambda row: "alpha" if row["type"] == "non-martensite" else row["type"], axis=1
    )
    data.to_json("".join([exp_data_dir, "gpc_predicted_types.json"]))


def do_martensite_start(db, DG_refit):
    print("Calculating martensite start model temperatures")
    types = ["lath", "plate", "epsilon"]
    model_params = {}
    systems = []
    for model_type in types:
        model_params[model_type] = pd.read_json(
            "".join([model_param_dir, f"mf_{model_type}_parameters.json"])
        )
        for term in list(model_params[model_type][f"mf_{model_type}_fits"][0].keys()):
            if term != "martensite_start" and term not in systems:
                systems.append(term)

    print(f"Systems included: {systems}")
    exp_data = pd.read_json("".join([exp_data_dir, "DG_calcs_with_gpc_type.json"]))
    exp_data = exp_data.query(
        "type in @types & martensite_start >= 298.15 & not reference.isnull() & ignore == False & not DG.isnull()"
    ).reset_index(drop=True)
    temp_df = pd.DataFrame()
    for reference in exp_data["reference"].unique():
        ref_df = exp_data.query("reference == @reference").copy()
        if len(ref_df) > 0:
            ref_df = ref_df.parallel_apply(lambda row: predict_Ms(row, db, storm=False), axis=1)
            temp_df = pd.concat([temp_df, ref_df])

    exp_data = temp_df.query("calc_error == False").copy()
    exp_data = exp_data.reset_index(drop=True)
    exp_data["error"] = exp_data["martensite_start"] - exp_data["mf_martensite_start"]
    # exp_data['mf_martensite_DG'] = exp_data.apply(lambda row: np.sum(list(row['contrib_dict'].values())) if 'contrib_dict' in list(row.keys()) else np.nan, axis=1)
    exp_data.to_json("".join([exp_data_dir, "predicted_martensite_start.json"]))
    ms_errors = temp_df.query("calc_error == True")
    ms_errors.to_json("".join([exp_data_dir, "predicted_martensite_start_errors.json"]))

    if DG_refit:  # refit the Stormvinter predictions
        storm_exp = pd.read_json("".join([exp_data_dir, "DG_calcs_with_gpc_type.json"]))
        storm_exp = storm_exp.query(
            "type in @types & martensite_start >= 298.15 & not reference.isnull() & ignore == False & not DG.isnull()"
        ).reset_index(drop=True)
        temp_df = pd.DataFrame()
        for reference in storm_exp["reference"].unique():
            ref_df = storm_exp.query("reference == @reference").copy()
            if len(ref_df) > 0:
                ref_df = ref_df.parallel_apply(lambda row: predict_Ms(row, db, storm=True), axis=1)
                temp_df = pd.concat([temp_df, ref_df])
        storm_exp = temp_df.query("calc_error == False").copy()
        storm_exp = storm_exp.reset_index(drop=True)
        storm_exp["error"] = storm_exp["martensite_start"] - storm_exp["storm_martensite_start"]
        storm_exp.to_json("".join([exp_data_dir, "stormvinter_predicted_martensite_start.json"]))
    else:
        storm_exp = pd.read_json(
            "".join([exp_data_dir, "stormvinter_predicted_martensite_start.json"])
        )


def make_parity_plots():
    exp_data = pd.read_json("".join([exp_data_dir, "predicted_martensite_start.json"]))
    storm_exp = pd.read_json("".join([exp_data_dir, "stormvinter_predicted_martensite_start.json"]))
    storm_exp["Type Prediction"] = storm_exp.apply(
        lambda row: "Predicted" if row["predicted_type"] == True else "Known", axis=1
    )
    exp_data["Type Prediction"] = exp_data.apply(
        lambda row: "Predicted" if row["predicted_type"] == True else "Known", axis=1
    )
    exp_data["Type"] = exp_data.apply(lambda row: row["type"].capitalize(), axis=1)
    lath_plate_exp = exp_data.query("type == 'lath' or type == 'plate'")
    epsilon_exp = exp_data.query("type == 'epsilon'")

    # MF model, all martensite types
    combined_RMSE = np.sqrt(np.mean(exp_data["error"] ** 2))
    line_vals = np.linspace(
        np.min(exp_data["martensite_start"]) - 15, np.max(exp_data["martensite_start"]) + 15
    )
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(
        data=exp_data,
        x="martensite_start",
        y="mf_martensite_start",
        hue="Type Prediction",
        style="Type",
        s=50,
        ax=ax,
    )
    sns.lineplot(x=line_vals, y=line_vals, ax=ax)
    sns.lineplot(x=line_vals, y=line_vals + combined_RMSE, color="black", linestyle="--", ax=ax)
    sns.lineplot(x=line_vals, y=line_vals - combined_RMSE, color="black", linestyle="--", ax=ax)
    ax.set_xlabel(r"Experimental $M_\mathrm{s}$ (K)")
    ax.set_ylabel(r"Predicted $M_\mathrm{s}$ (K)")
    ax.text(400, 1050, f"Alloys Predicted: {len(exp_data)}")
    ax.text(400, 1000, f"RMSE: {round(combined_RMSE, 2)} K")
    fig.tight_layout()
    fig.savefig("".join([figure_dir, "mf_combined_parity_plot_all.png"]))
    plt.close()

    mf_combined_known_type_RMSE = np.sqrt(
        np.mean(exp_data.query("`Type Prediction` == 'Known Type'")["error"] ** 2)
    )
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(
        data=exp_data.query("`Type Prediction` == 'Known Type'"),
        x="martensite_start",
        y="mf_martensite_start",
        hue="alloy_system",
        s=50,
        ax=ax,
    )
    sns.lineplot(x=line_vals, y=line_vals, ax=ax)
    sns.lineplot(
        x=line_vals, y=line_vals + mf_combined_known_type_RMSE, color="black", linestyle="--", ax=ax
    )
    sns.lineplot(
        x=line_vals, y=line_vals - mf_combined_known_type_RMSE, color="black", linestyle="--", ax=ax
    )
    ax.set_xlabel(r"Experimental $M_\mathrm{s}$ (K)")
    ax.set_ylabel(r"Predicted $M_\mathrm{s}$ (K)")
    ax.text(
        400,
        1050,
        "Alloys Predicted: " + str(len(exp_data.query("`Type Prediction` == 'Known Type'"))),
    )
    ax.text(400, 1000, f"RMSE: {round(mf_combined_known_type_RMSE, 2)} K")
    fig.tight_layout()
    fig.savefig("".join([figure_dir, "mf_combined_parity_plot_known_type.png"]))
    plt.close()

    # MF model, lath  & plate
    mf_lp_RMSE = np.sqrt(np.mean(lath_plate_exp["error"] ** 2))
    line_vals = np.linspace(
        np.min(lath_plate_exp["martensite_start"]) - 15,
        np.max(lath_plate_exp["martensite_start"]) + 15,
    )
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(
        data=lath_plate_exp,
        x="martensite_start",
        y="mf_martensite_start",
        hue="Type Prediction",
        style="Type",
        s=50,
        ax=ax,
    )
    sns.lineplot(x=line_vals, y=line_vals, ax=ax)
    sns.lineplot(x=line_vals, y=line_vals + mf_lp_RMSE, color="black", linestyle="--", ax=ax)
    sns.lineplot(x=line_vals, y=line_vals - mf_lp_RMSE, color="black", linestyle="--", ax=ax)
    ax.set_xlabel(r"Experimental $M_\mathrm{s}$ (K)")
    ax.set_ylabel(r"Predicted $M_\mathrm{s}$ (K)")
    ax.text(400, 1050, f"Alloys Predicted: {len(lath_plate_exp)}")
    ax.text(400, 1000, f"RMSE: {round(mf_lp_RMSE, 2)} K")
    fig.tight_layout()
    fig.savefig("".join([figure_dir, "mf_lath-plate_parity_plot_all.png"]))
    plt.close()

    mf_lp_known_type_RMSE = np.sqrt(
        np.mean(lath_plate_exp.query("`Type Prediction` == 'Known Type'")["error"] ** 2)
    )
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(
        data=lath_plate_exp.query("`Type Prediction` == 'Known'"),
        x="martensite_start",
        y="mf_martensite_start",
        hue="alloy_system",
        s=50,
        ax=ax,
    )
    sns.lineplot(x=line_vals, y=line_vals, ax=ax)
    sns.lineplot(x=line_vals, y=line_vals + mf_lp_RMSE, color="black", linestyle="--", ax=ax)
    sns.lineplot(x=line_vals, y=line_vals - mf_lp_RMSE, color="black", linestyle="--", ax=ax)
    ax.set_xlabel(r"Experimental $M_\mathrm{s}$ (K)")
    ax.set_ylabel(r"Predicted $M_\mathrm{s}$ (K)")
    ax.text(
        400,
        1050,
        f"Alloys Predicted: " + str(len(lath_plate_exp.query("`Type Prediction` == 'Known'"))),
    )
    ax.text(400, 1000, f"RMSE: {round(mf_lp_known_type_RMSE, 2)} K")
    fig.tight_layout()
    fig.savefig("".join([figure_dir, "mf_lath-plate_parity_plot_known_type.png"]))
    plt.close()

    # MF model, epsilon
    mf_eps_RMSE = np.sqrt(np.mean(epsilon_exp["error"] ** 2))
    line_vals = np.linspace(
        np.min(epsilon_exp["martensite_start"]) - 15, np.max(epsilon_exp["martensite_start"]) + 15
    )
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(
        data=epsilon_exp,
        x="martensite_start",
        y="mf_martensite_start",
        hue="Type Prediction",
        s=50,
        ax=ax,
    )
    sns.lineplot(x=line_vals, y=line_vals, ax=ax)
    sns.lineplot(x=line_vals, y=line_vals + mf_eps_RMSE, color="black", linestyle="--", ax=ax)
    sns.lineplot(x=line_vals, y=line_vals - mf_eps_RMSE, color="black", linestyle="--", ax=ax)
    ax.set_xlabel(r"Experimental $M_\mathrm{s}$ (K)")
    ax.set_ylabel(r"Predicted $M_\mathrm{s}$ (K)")
    ax.text(350, 460, f"Alloys Predicted: {len(epsilon_exp)}")
    ax.text(350, 450, f"RMSE: {round(mf_eps_RMSE, 2)} K")
    fig.tight_layout()
    fig.savefig("".join([figure_dir, "mf_epsilon_parity_plot_all.png"]))
    plt.close()

    mf_eps_known_type_RMSE = np.sqrt(
        np.mean(epsilon_exp.query("`Type Prediction` == 'Known'")["error"] ** 2)
    )
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(
        data=epsilon_exp.query("`Type Prediction` == 'Known'"),
        x="martensite_start",
        y="mf_martensite_start",
        hue="alloy_system",
        ax=ax,
    )
    sns.lineplot(x=line_vals, y=line_vals, ax=ax)
    sns.lineplot(
        x=line_vals, y=line_vals + mf_eps_known_type_RMSE, color="black", linestyle="--", ax=ax
    )
    sns.lineplot(
        x=line_vals, y=line_vals - mf_eps_known_type_RMSE, color="black", linestyle="--", ax=ax
    )
    ax.set_xlabel(r"Experimental $M_\mathrm{s}$ (K)")
    ax.set_ylabel(r"Predicted $M_\mathrm{s}$ (K)")
    ax.text(
        350,
        460,
        f"Alloys Predicted: " + str(len(epsilon_exp.query("`Type Prediction` == 'Known'"))),
    )
    ax.text(350, 450, f"RMSE: {round(mf_eps_known_type_RMSE, 2)} K")
    fig.tight_layout()
    fig.savefig("".join([figure_dir, "mf_epsilon_parity_plot_known_type.png"]))
    plt.close()

    # systems = [sys for sys in exp_data['alloy_system'].unique() if 'Ni' in sys]
    # exp_data = exp_data.query("alloy_system in @systems")
    # MF model
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(data=lath_plate_exp, x="martensite_start", y="error", hue="alloy_system", ax=ax)
    plt.hlines(0, 0, np.max(epsilon_exp["martensite_start"]))
    ax.set_xlabel(r"Experimental $M_\mathrm{s}$ (K)")
    ax.set_ylabel(r"Predicted $M_\mathrm{s}$ Error (K)")
    fig.tight_layout()
    fig.savefig("".join([figure_dir, "Ms_lath-plate_residuals_plot_all.png"]))
    plt.close()

    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(
        data=lath_plate_exp.query("`Type Prediction` == 'Known'"),
        x="martensite_start",
        y="error",
        hue="alloy_system",
        ax=ax,
    )
    plt.hlines(0, 0, np.max(epsilon_exp["martensite_start"]))
    ax.set_xlabel(r"Experimental $M_\mathrm{s}$ (K)")
    ax.set_ylabel(r"Predicted $M_\mathrm{s}$ Error (K)")
    fig.tight_layout()
    fig.savefig("".join([figure_dir, "Ms_lath-plate_residuals_plot_known_type.png"]))
    plt.close()

    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(data=epsilon_exp, x="martensite_start", y="error", hue="alloy_system", ax=ax)
    plt.hlines(0, 0, np.max(epsilon_exp["martensite_start"]))
    ax.set_xlabel(r"Experimental $M_\mathrm{s}$ (K)")
    ax.set_ylabel(r"Predicted $M_\mathrm{s}$ Error (K)")
    fig.tight_layout()
    fig.savefig("".join([figure_dir, "Ms_epsilon_residuals_plot_all.png"]))
    plt.close()

    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(
        data=epsilon_exp.query("`Type Prediction` == 'Known'"),
        x="martensite_start",
        y="error",
        hue="alloy_system",
        ax=ax,
    )
    plt.hlines(0, 0, np.max(epsilon_exp["martensite_start"]))
    ax.set_xlabel(r"Experimental $M_\mathrm{s}$ (K)")
    ax.set_ylabel(r"Predicted $M_\mathrm{s}$ Error (K)")
    fig.tight_layout()
    fig.savefig("".join([figure_dir, "Ms_epsilon_residuals_plot_known_type.png"]))
    plt.close()

    # plot the largest contributors to the error
    exp_data.loc[:, "error"] = np.abs(exp_data.loc[:, "error"])
    exp_data = exp_data.sort_values("error")

    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(
        data=exp_data.tail(25),
        x="martensite_start",
        y="error",
        hue="Type",
        style="alloy_system",
        s=50,
        ax=ax,
    )
    ax.set_xlabel(r"Experimental $M_\mathrm{s}$ (K)")
    ax.set_ylabel(r"Predicted $M_\mathrm{s}$ Error (K)")
    fig.tight_layout()
    fig.savefig("".join([figure_dir, "mf-steel_highest_residuals_all.png"]))
    plt.close()

    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(
        data=exp_data.query("`Type Prediction` == 'Known'").tail(25),
        x="martensite_start",
        y="error",
        hue="Type",
        style="alloy_system",
        s=50,
        ax=ax,
    )
    ax.set_xlabel(r"Experimental $M_\mathrm{s}$ (K)")
    ax.set_ylabel(r"Predicted $M_\mathrm{s}$ Error (K)")
    fig.tight_layout()
    fig.savefig("".join([figure_dir, "mf-steel_highest_residuals_known_type.png"]))
    plt.close()

    # Stormvinter model
    sv_RMSE = np.sqrt(np.mean(storm_exp["error"] ** 2))
    line_vals = np.linspace(0, np.max(storm_exp["martensite_start"]))
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(
        data=storm_exp, x="martensite_start", y="storm_martensite_start", hue="alloy_system", ax=ax
    )
    sns.lineplot(x=line_vals, y=line_vals + sv_RMSE, color="black", linestyle="--", ax=ax)
    sns.lineplot(x=line_vals, y=line_vals - sv_RMSE, color="black", linestyle="--", ax=ax)
    sns.lineplot(x=line_vals, y=line_vals, ax=ax)
    ax.set_xlabel(r"Experimental $M_\mathrm{s}$ (K)")
    ax.set_ylabel(r"Predicted $M_\mathrm{s}$ (K)")
    # ax.set_title("Stormvinter2012 Steel Lath and Plate Martensite Start Model")

    ax.text(400, 1200, f"RMSE: {round(sv_RMSE, 2)} K")
    fig.tight_layout()
    fig.savefig("".join([figure_dir, "stormvinter_parity_plot_all.png"]))
    plt.close()

    sv_RMSE = np.sqrt(np.mean(storm_exp.query("`Type Prediction` == 'Known'")["error"] ** 2))
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(
        data=storm_exp.query("`Type Prediction` == 'Known'"),
        x="martensite_start",
        y="storm_martensite_start",
        hue="alloy_system",
        ax=ax,
    )
    sns.lineplot(x=line_vals, y=line_vals + sv_RMSE, color="black", linestyle="--", ax=ax)
    sns.lineplot(x=line_vals, y=line_vals - sv_RMSE, color="black", linestyle="--", ax=ax)
    sns.lineplot(x=line_vals, y=line_vals, ax=ax)
    ax.set_xlabel(r"Experimental $M_\mathrm{s}$ (K)")
    ax.set_ylabel(r"Predicted $M_\mathrm{s}$ (K)")
    # ax.set_title("Stormvinter2012 Steel Lath and Plate Martensite Start Model")

    ax.text(400, 1200, f"RMSE: {round(sv_RMSE, 2)} K")
    fig.tight_layout()
    fig.savefig("".join([figure_dir, "stormvinter_parity_plot_known_type.png"]))
    plt.close()

    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(data=storm_exp, x="martensite_start", y="error", hue="alloy_system", ax=ax)
    plt.hlines(0, 0, np.max(storm_exp["martensite_start"]))
    ax.set_xlabel(r"Experimental $M_\mathrm{s}$ (K)")
    ax.set_ylabel(r"Predicted $M_\mathrm{s}$ Error (K)")
    # ax.set_title("Stormvinter2012 Steel Martensite Start Model Residuals")
    fig.tight_layout()
    fig.savefig("".join([figure_dir, "stormvinter_residual_plot_all.png"]))
    plt.close()

    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(
        data=storm_exp.query("`Type Prediction` == 'Known'"),
        x="martensite_start",
        y="error",
        hue="alloy_system",
        ax=ax,
    )
    plt.hlines(0, 0, np.max(storm_exp["martensite_start"]))
    ax.set_xlabel(r"Experimental M$M_\mathrm{s}$s (K)")
    ax.set_ylabel(r"Predicted $M_\mathrm{s}$ Error (K)")
    # ax.set_title("Stormvinter2012 Steel Martensite Start Model Residuals")
    fig.tight_layout()
    fig.savefig("".join([figure_dir, "stormvinter_residual_plot_known_type.png"]))
    plt.close()

    # Recreate commercial steel parity plots from Stormvinter2012 to compare performance

    exp_data["predicted_martensite_start"] = exp_data["mf_martensite_start"]
    exp_data["model_type"] = "This Work"
    storm_exp["predicted_martensite_start"] = storm_exp["storm_martensite_start"]
    storm_exp["model_type"] = "Stormvinter"
    all_exp_data = pd.concat([storm_exp, exp_data])

    # Plot error versus number of components
    exp_data["num_components"] = exp_data.apply(lambda row: len(row["components"]) - 1, axis=1)
    exp_data["error"] = np.abs(exp_data["error"])
    error_data = exp_data.sort_values("error", ascending=False)

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=exp_data, x="num_components", y="error", ax=ax)
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Error (K)")
    fig.tight_layout()
    fig.savefig("/".join([figure_dir, "error_vs_components.png"]))
    plt.close()

    print(f"MF total RMS Error: {combined_RMSE} K")
    print(f"MF Lath/Plate RMS Error: {mf_lp_RMSE} K")
    print(f"MF Epsilon RMS Error: {mf_eps_RMSE} K")
    print(f"Stormvinter RMS Error: {sv_RMSE} K")


def make_a_step_plot(row, db):
    filename = f"{row['index']}-step-{row['reference']}-Ms_{row['martensite_start']}.png"
    figure_name = "/".join([figure_dir, "step", filename])
    if filename in os.listdir("/".join([figure_dir, "step"])):
        return figure_name  # already made the step and not re-making them
    else:
        print(f"Generating new step plot: {filename}")
        if pd.isnull(
            row["austenitization_temperature"]
        ):  # solutionizing tempeature not provided by publication
            row["austenitization_temperature"] = 1273.15  # assume 1000 C
        phases = [phase for phase in list(db.phases.keys()) if phase not in disabled_phases]
        conditions = convert_conditions(row["conditions"])
        conditions.update({v.T: (300, 2000, 5), v.P: 101325})
        components = get_components_from_conditions(conditions, dependent_component="FE")

        fig, ax = step_plot(db, components, phases, conditions, fontsize=24)
        # ax.set_title(f"Step plot for {row['reference']}, Ms {row['martensite_start']} \n Composition: {row['conditions']}")
        ax.vlines(row["austenitization_temperature"], 1e-5, 1.1, color="red")
        ax.vlines(row["martensite_start"], 1e-5, 1.1, color="blue")
        fig.tight_layout()
        fig.savefig(figure_name)
        plt.close()
        return figure_name


def step_plot_wrapper(db):
    # Make step plots for each entry
    data = pd.read_json("".join([exp_data_dir, "cleaned_exp_data.json"]))
    data["step_file"] = data.apply(lambda row: make_a_step_plot(row, db), axis=1)
    data.to_json("".join([exp_data_dir, "cleaned_exp_data_with_step.json"]))


def update_param_markdown():
    # Update the current fitting parameters markdown file
    model_param_dir = "./model_params"
    plate_params = pd.read_json("/".join([model_param_dir, "mf_plate_parameters.json"]))
    lath_params = pd.read_json("/".join([model_param_dir, "mf_lath_parameters.json"]))
    epsilon_params = pd.read_json("/".join([model_param_dir, "mf_epsilon_parameters.json"]))
    md_file = "./model_parameters.md"

    with open(md_file, "w") as f:
        f.write("# Instructions\n")
        f.write(
            "This file contains the current fitting parameters of the martensite-start temperature model. Each type of martensite is included. The fitting parameters are listed below by the binary (Fe-X) or ternary (Fe-X-Y) system. Parameters with more than value are for higher order fits.\n"
        )
        f.write("# Lath model terms:\n")
        for term, fits in lath_params["mf_lath_fits"][0].items():
            f.write(f"{term}: {fits[0]}")
            f.write("\n")

        f.write("\n")
        f.write("# Plate model terms\n")
        for term, fits in plate_params["mf_plate_fits"][0].items():
            f.write(f"{term}: {fits[0]}")
            f.write("\n")

        f.write("\n")
        f.write("Epsilon model terms:\n")
        for term, fits in epsilon_params["mf_epsilon_fits"][0].items():
            f.write(f"{term}: {fits[0]}")
            f.write("\n")


def save_training_data_json(training_df):
    def save_json_row(row):
        if row["ignore"]:
            return row
        cols_to_keep = [
            "components",
            "conditions",
            "output",
            "phases",
            "reference",
            "solver",
            "values",
        ]
        interstitials = ["B", "C", "H", "N", "O", "VA"]
        type_dir = None
        site_ratios = None
        if row["type"] == "lath":
            type_dir = "MSL"
            row["phases"] = ["BCT_LATH"]
            site_ratios = [1, 3]
            row["output"] = "MSL"
        elif row["type"] == "plate":
            type_dir = "MSP"
            site_ratios = [1, 3]
            row["phases"] = ["BCT_PLATE"]
            row["output"] = "MSP"
        elif row["type"] == "epsilon":
            type_dir = "MSE"
            site_ratios = [1, 0.5]
            row["phases"] = ["HCP_EPS"]
            row["output"] = "MSE"
        assert type_dir is not None and site_ratios is not None, f"Cannot identify type for {row}."
        subs = [comp for comp in row["components"] if comp not in interstitials]
        ints = [comp for comp in row["components"] if comp in interstitials]
        if len(subs) > 1 and len(ints) == 1:
            sublattice_configurations = [[*subs], *ints]
        if len(subs) == 1 and len(ints) > 1:
            sublattice_configurations = [*subs, [*ints]]
        else:
            sublattice_configurations = [[*subs], [*ints]]
        sublattice_configurations = [[*subs, *ints]]
        row["solver"] = {
            "mode": "manual",
            "sublattice_site_ratios": site_ratios,
            "sublattice_configurations": sublattice_configurations,
        }
        row["conditions"] = row["solutionizing_conditions"]
        row["conditions"]["T"] = row["martensite_start"]
        row["values"] = [[[row["DG"]]]]

        if not os.path.isdir(os.path.join(exp_data_dir, "training")):
            os.makedirs(os.path.join(exp_data_dir, "training"))
        if not os.path.isdir(os.path.join(exp_data_dir, "training", type_dir)):
            os.makedirs(os.path.join(exp_data_dir, "training", type_dir))
        if not os.path.isdir(os.path.join(exp_data_dir, "training", type_dir, row["reference"])):
            os.makedirs(os.path.join(exp_data_dir, "training", type_dir, row["reference"]))
        ref_dir = os.path.join(exp_data_dir, "training", type_dir, row["reference"])
        index = row["index"]
        row = row.drop([col for col in row.index if col not in cols_to_keep])
        row.to_json(os.path.join(ref_dir, row["reference"] + "-" + str(index) + ".json"))

    # training_df.apply(lambda row: save_json_row(row), axis=1)
    print(training_df["alloy_system"].unique())
    types_dict = {"lath": "MSL", "plate": "MSP", "epsilon": "MSE"}
    phases_dict = {"lath": "BCT_LATH", "plate": "BCT_PLATE", "epsilon": "HCP_EPS"}
    site_ratios_dict = {"lath": [1, 3], "plate": [1, 3], "epsilon": [1, 0.5]}
    if not os.path.isdir(os.path.join(exp_data_dir, "training", "1-unary")):
        os.makedirs(os.path.join(exp_data_dir, "training", "1-unary"))
    if not os.path.isdir(os.path.join(exp_data_dir, "training", "2-binary")):
        os.makedirs(os.path.join(exp_data_dir, "training", "2-binary"))
    if not os.path.isdir(os.path.join(exp_data_dir, "training", "3-ternary")):
        os.makedirs(os.path.join(exp_data_dir, "training", "3-ternary"))

    interstitials = ["B", "C", "H", "N", "O", "VA"]
    for system in training_df["alloy_system"].unique():
        system_df = training_df.query("alloy_system == @system & not ignore")
        if system == "Fe-":
            num_components = 1
            system = "Fe"
            sorted_system = ["FE", "VA"]
        else:
            num_components = len(system.split("-"))
            sorted_system = system.split("-")
            sorted_system.sort()
            system = "-".join(sorted_system)
            sorted_system.append("VA")
        system_path = None
        if num_components == 1:  # Should only be Fe
            system_path = os.path.join(exp_data_dir, "training", "1-unary", system)
        if num_components == 2:
            system_path = os.path.join(exp_data_dir, "training", "2-binary", system)
        if num_components == 3:
            system_path = os.path.join(exp_data_dir, "training", "3-ternary", system)
        assert system_path is not None, (
            f"Cannot identify system path for training data in {system}."
        )
        if not os.path.isdir(system_path):
            os.makedirs(system_path)
        system_ints = [comp.upper() for comp in sorted_system if comp in interstitials]
        system_subs = [comp.upper() for comp in sorted_system if comp not in interstitials]
        for martensite_type in system_df["type"].unique():
            out_dict = {}
            type_df = system_df.query("type == @martensite_type").reset_index(drop=True)
            for i in type_df.index:
                entry = type_df.iloc[i]
                out_dict["components"] = [comp.upper() for comp in sorted_system]
                out_dict["reference"] = entry["reference"]
                out_file = os.path.join(
                    system_path, f"{types_dict[martensite_type]}-{entry['martensite_start']}.json"
                )
                out_dict["phases"] = phases_dict[martensite_type]
                site_ratios = site_ratios_dict[martensite_type]
                out_dict["output"] = types_dict[martensite_type]
                out_dict["conditions"] = {
                    "P": 101325,
                    "N": 1,
                    "T": entry["martensite_start"],
                }
                if len(system_subs) > 1 and len(system_ints) > 1:
                    sublattice_configurations = [system_subs, system_ints]
                elif len(system_ints) > 1:
                    sublattice_configurations = [*system_subs, system_ints]
                elif len(system_subs) > 1:
                    sublattice_configurations = [system_subs, *system_ints]
                else:
                    sublattice_configurations = [*system_subs, *system_ints]
                # sublattice_configurations = list(
                # itertools.repeat(sublattice_configurations, len(type_df["martensite_start"]))
                # )
                sublattice_occupancies = []
                this_subs_occupancy = []
                this_ints_occupancy = []
                va_conc = 1
                fe_conc = 1
                for comp in sorted_system:
                    if comp == "FE" or comp == "VA":
                        continue
                    comp_conc = entry[comp]
                    if comp.upper() in system_subs:
                        fe_conc -= comp_conc
                    else:
                        va_conc -= comp_conc
                for comp in sorted_system:
                    if comp.upper() in system_subs:
                        if comp == "FE":
                            this_subs_occupancy.append(1 / site_ratios[0] * fe_conc)
                        else:
                            comp_conc = type_df.iloc[i][comp]
                            this_subs_occupancy.append(1 / site_ratios[0] * comp_conc)
                    else:
                        if comp == "VA":
                            this_ints_occupancy.append(
                                1 - (1 / site_ratios[1] * (1 - va_conc) / va_conc)
                            )
                        else:
                            comp_conc = type_df.iloc[i][comp]
                            this_ints_occupancy.append(1 / site_ratios[1] * comp_conc / va_conc)
                if num_components == 1:  # Fe only
                    this_occupancy = [1, 0]
                if len(system_subs) > 1 and len(system_ints) > 1:
                    this_occupancy = [this_subs_occupancy, this_ints_occupancy]
                elif len(system_ints) > 1:
                    this_occupancy = [*this_subs_occupancy, this_ints_occupancy]
                elif len(system_subs) > 1:
                    this_occupancy = [this_subs_occupancy, *this_ints_occupancy]
                else:
                    this_occupancy = [*this_subs_occupancy, *this_ints_occupancy]
                sublattice_occupancies.append(this_occupancy)
                out_dict["solver"] = {
                    "mode": "manual",
                    "sublattice_site_ratios": site_ratios,
                    "sublattice_configurations": [sublattice_configurations],
                    "sublattice_occupancies": sublattice_occupancies,
                }
                out_dict["values"] = [[[entry["DG"]]]]
                with open(out_file, "w") as f:
                    json.dump(out_dict, f)


if __name__ == "__main__":
    if pandarallel is not None:
        pandarallel.initialize(nb_workers=nb_workers)
    args = " ".join(sys.argv[1:])
    start_time = time.perf_counter()
    db = load_database(dbf)
    DG_refit = False
    do_curve_fit = True
    load_exp_data()
    if "refit" in args:
        DG_refit = True
    if "step" in args:
        step_plot_wrapper(db)
    if "predict type" in args or "all" in args:
        predict_type()
    if "DG" in args or "all" in args:  # update DG calcs
        calc_DG(db, DG_refit)
    if "curve fit" in args or "all" in args:
        print("Refitting models")
        fit_models(db, do_curve_fit, DG_refit)
        print("Projecting and interpolating.")
        project_models()
    if "plot" in args or "curve fit" in args or "all" in args:
        print("Making plots")
        make_plots()
    if any(["markdown" in args, "all" in args]):
        update_param_markdown()
    if "pags" in args or "all" in args:
        print("Modeling PAGS effects.")
        model_pags(db)
    if (
        "martensite start" in args or "all" in args
    ):  # calculate martensite start model temperatures to compare to experimental data
        do_martensite_start(db, DG_refit)
    if any(["parity" in args, "all" in args]):
        make_parity_plots()
    else:
        print("Anything else you want me to do?")
        print(f"Arguments were: {args}")
        print(
            "Options are 'curve fit', 'DG', 'martensite start', 'pags', 'parity', 'plot', 'predict type', 'project', 'refit', or 'all'."
        )
        print("Please try again.")

    end_time = time.perf_counter()
    print(f"Fini! Total elapsed time={(end_time - start_time) / 60} minutes.")
