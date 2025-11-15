from espei.datasets import load_datasets, recursive_glob
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from tinydb import Query

R = 8.314462


def calculate_energy_from_activity(
    local_folder: str, components: list[str], phases: list[str]
) -> pd.DataFrame:
    """
    Function to calculate energy from activity data
    """

    datasets = load_datasets(recursive_glob(local_folder))
    datasets = datasets.search(
        (Query().components == components) & (Query().phases == phases)
    )
    activity_df = pd.DataFrame()
    for dataset in datasets:
        if dataset["output"].startswith("ACR"):
            out_dict = {}
            component = dataset["output"].split("_")[1]
            concentrations = np.array(dataset["conditions"][f"X_{component}"])
            temperature = np.array(dataset["conditions"]["T"])
            pressure = np.array(dataset["conditions"]["P"])
            activities = np.array(dataset["values"][0][0])
            out_dict["concentration"] = concentrations
            out_dict["activity"] = activities
            out_dict["activity_coefficient"] = activities / concentrations
            out_dict["component"] = np.repeat(component, len(activities))
            out_dict["temperature"] = np.repeat(temperature, len(activities))
            out_dict["pressure"] = np.repeat(pressure, len(activities))
            out_dict["reference"] = dataset["reference"]
            if activity_df.empty:
                activity_df = pd.DataFrame.from_dict(out_dict)
            else:
                activity_df = pd.concat([activity_df, pd.DataFrame.from_dict(out_dict)])
    activity_df["GM"] = R * activity_df["temperature"] * np.log(activity_df["activity"])
    activity_df["GM_EX"] = (
        R * activity_df["temperature"] * np.log(activity_df["activity_coefficient"])
    )
    activity_df = activity_df.dropna()
    activity_df = activity_df.reset_index(drop=True)
    return activity_df


def fit_ideal_solution_model(activity_df):
    activity_df["SM_MIX"] = (
        -R * activity_df["concentration"] * np.log(activity_df["concentration"])
    )

    return activity_df


def fit_regular_solution_model(activity_df):
    """
    Function to fit a regular solution model to the enthalpy of mixing.
    TODO: Generalize to fit more than binary systems.
    """

    def _regular_solution_fit(x, dh0, w):
        HM_MIX = dh0 + w * x[0] * x[1]
        return HM_MIX

    activity_df["activity_coefficient"] += 1e-12  # need to handle 0 values
    activity_df = fit_ideal_solution_model(activity_df)

    activity_df["HM_MIX"] = (
        activity_df["GM_EX"] + R * activity_df["temperature"] * activity_df["SM_MIX"]
    )
    x_values = np.vstack(
        [
            activity_df["concentration"],
            1 - activity_df["concentration"],
        ]
    )
    fits = curve_fit(
        _regular_solution_fit,
        xdata=x_values,
        ydata=activity_df["HM_MIX"],
        nan_policy="omit",
    )

    return fits, activity_df


def fit_subregular_solution_model(activity_df):
    """
    Function to fit a regular solution model to the enthalpy of mixing.
    TODO: Generalize to fit more than binary systems.
    """

    def _subregular_solution_fit(x, dh0, w, a):
        HM_MIX = dh0 + x[0] * x[1] * (w + (2 * a) / x[2])
        return HM_MIX

    activity_df["activity_coefficient"] += 1e-12  # need to handle 0 values
    activity_df = fit_ideal_solution_model(activity_df)

    activity_df["HM_MIX"] = (
        activity_df["GM_EX"] + R * activity_df["temperature"] * activity_df["SM_MIX"]
    )
    x_values = np.vstack(
        [
            activity_df["concentration"],
            1 - activity_df["concentration"],
            activity_df["temperature"],
        ]
    )
    fits = curve_fit(
        _subregular_solution_fit,
        xdata=x_values,
        ydata=activity_df["HM_FORM"],
        nan_policy="omit",
    )

    return fits, activity_df
