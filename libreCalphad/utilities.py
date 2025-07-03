import json
from libreCalphad.databases.db_utils import load_database
import numpy as np
import pandas as pd
import pycalphad.variables as v
from scipy.optimize import minimize


def convert_c_ratio(y_c: float) -> float:
    def calc_c_ratio(x_c: float) -> float:
        x_fe = 1 - x_c
        return x_c / x_fe

    return minimize(lambda x: np.abs(y_c - calc_c_ratio(x)), x0=y_c).x[0]


def _format_espei_data_folder(espei_data_folder, components):
    if espei_data_folder.endswith("/"):
        espei_data_folder = espei_data_folder[:-1]
    if all(
        [
            espei_data_folder.split("/")[-1] != "datasets",
            espei_data_folder.split("/")[-2] != "datasets",
        ]
    ):
        espei_data_folder = "/".join([espei_data_folder, "datasets"])
    system = components.copy()
    if "VA" in system:
        system.remove("VA")
        system.sort()
    system = list(map(str.capitalize, system))
    if len(system) == 2:
        espei_data_folder += "/" + "2-binary" + "/"
    if len(system) == 3:
        espei_data_folder += "/" + "3-ternary" + "/"
    system = "-".join(system)
    espei_data_folder += system + "/"
    return espei_data_folder


def write_zpf_json(
    input_file: str,
    input_dict: dict,
    values_dict: dict,
    dbf,
    broadcast_conditions=False,
    conditions=None,
    espei_data_folder=None,
):
    input_df = pd.read_csv(
        input_file, skiprows=[1]
    )  # need to skip the second row that simply contains "X" and "Y"
    components = input_dict["components"]
    if espei_data_folder is not None:
        espei_data_folder = _format_espei_data_folder(espei_data_folder, components)
    for key1, value1 in values_dict.items():
        out_df = pd.DataFrame()
        out_dict = input_dict.copy()
        out_dict["broadcast_conditions"] = broadcast_conditions
        out_dict["output"] = "ZPF"
        phases = []
        out_dict["values"] = {}
        if conditions is None:
            conditions = {"P": 101325}  # Assume atmospheric pressure if not provided.

        for key2, value2 in value1.items():
            if value2["values"] is not None:
                out_df[value2["values"]] = input_df[value2["values"]].copy()
            if key2 == "temperatures":
                out_df[value2["values"]] = out_df[value2["values"]].astype("float")
                if value2["units"] == "degC":
                    out_df[value2["values"]] += 273.15
                conditions["T"] = value2["values"]
            else:
                this_phase = value2["phase"]
                if this_phase not in phases:
                    phases.append(this_phase)
                if value2["values"] is not None:
                    if value2["units"] == "weight_percent":
                        out_df[value2["values"]] = (
                            out_df[value2["values"]] / 100
                        )  # convert to weight fraction
                        if len(components) == 2 or (
                            len(components) == 3 and "VA" in components
                        ):
                            # binary system
                            dep_comp = [
                                comp
                                for comp in components
                                if comp != "VA" and comp != value2["component"]
                            ][0]
                            out_df[value2["values"]] = out_df.apply(
                                lambda row: v.get_mole_fractions(
                                    {v.W(value2["component"]): row[value2["values"]]},
                                    dep_comp,
                                    dbf,
                                )[v.X(value2["component"])],
                                axis=1,
                            )
                        else:
                            raise NotImplementedError(
                                "Only binary systems are implemented"
                            )
                    elif value2["units"] == "C_ratio":
                        out_df[value2["values"]] = out_df.apply(
                            lambda row: convert_c_ratio(row[value2["values"]]), axis=1
                        )
                    else:
                        pass  # already in molar fractions
                out_dict["values"][this_phase] = {
                    "component": value2["component"],
                    "values": value2["values"],
                }
        out_df = out_df.dropna(how="any").reset_index(drop=True)

        # Now build the output from the parsed dataframe and metadata already collected
        conditions["T"] = list(out_df[conditions["T"]].values)
        out_dict["phases"] = phases
        out_dict["conditions"] = conditions
        out_values = []
        for i in range(len(out_df)):
            this_row = []
            for phase, values in out_dict["values"].items():
                this_val = []
                this_val.append(phase)
                this_val.append([out_dict["values"][phase]["component"]])
                if out_dict["values"][phase]["values"] is not None:
                    this_val.append(
                        [out_df.loc[i][out_dict["values"][phase]["values"]]]
                    )
                else:
                    this_val.append([None])
                this_row.append(this_val)
            out_values.append(this_row)
        out_dict["values"] = out_values
        out_file = f"./{'-'.join([comp for comp in components if comp != 'VA'])}-ZPF-"
        out_file += f"{'-'.join([phase for phase in phases])}-"
        out_file += input_dict["bibtex"] + ".json"
        with open(out_file, "w") as f:
            json.dump(out_dict, f, indent=4)
        if espei_data_folder is not None:
            espei_data_folder += "zpf" + out_file[1:]
            print("Saving to ESPEI-datasets at: " + espei_data_folder)
            with open(espei_data_folder, "w") as f:
                json.dump(out_dict, f, indent=4)

    return True


def write_activity_json(
    input_file,
    input_dict,
    values_dict,
    dbf,
    out_file=None,
    conditions=None,
    espei_data_folder=None,
):
    """
    Function to write an ESPEI-style JSON file for ZPF data from a WebPlotDigitizer csv passed in as a Pandas DataFrame.
    Currently assumes a standard isobaric phase diagram.
    TODO: Allow for isochoric phase data with varying pressure.

    Parameters: input_file, string
                    Pointer to the input csv file.
                input_dict, dictionary
                    The initial input dictionary with reference metadata and component list.
                values_dict, dictionary
                    A dictionary containing metadata for the dataset.
                dbf, pycalphad Database
                    The database to use.
                out_file, string or None, default value None
                    A string for the name of the output file. If None a default name based on the name of the input file is created.
                broadcast_conditions, boolean, default value False
                    The broadcast_conditions boolean.
                conditions, dictionary or None, default value None
                    The pycalphad conditions dictionary. If none is provided, defaults to STP conditions.

    Returns: True if success.
    """

    input_df = pd.read_csv(input_file, skiprows=[1])
    components = input_dict["components"]
    phases = input_dict["phases"]
    ref_phase = input_dict["reference_state"]["phases"][0]
    conditions = {"P": 101325}
    out_df = pd.DataFrame()
    if espei_data_folder is not None:
        espei_data_folder = _format_espei_data_folder(espei_data_folder, components)

    for key1, value1 in values_dict.items():
        out_dict = input_dict.copy()
        this_out_df = pd.DataFrame()
        # out_df = pd.DataFrame()
        component = None
        for key2, value2 in value1.items():
            if key2 == "temperatures":
                if value2["values"] in list(input_df.columns):
                    this_temperature = input_df[value2["values"]].astype("float")
                else:  # assume an actual temperature is provided
                    this_temperature = value2["values"]
                if value2["units"] == "degC":
                    this_temperature += 273.15
            if key2.startswith("ACR_"):
                out_dict["output"] = key2
                component = key2.split("_")[-1]
                conc_col = value2["concentration"]
                act_col = value2["activity"]
                this_out_df["concentration"] = input_df[conc_col].astype("float")
                this_out_df["activity"] = input_df[act_col].astype("float")
                if value2["units"] == "weight_percent":
                    this_out_df["concentration"] = this_out_df["concentration"] / 100
                    this_out_df["activity"] = this_out_df["activity"] / 100
                    if len(components) == 2 or (
                        len(components) == 3 and "VA" in components
                    ):
                        dep_component = [
                            comp
                            for comp in components
                            if comp != "VA" and comp != component
                        ][0]
                        this_out_df["concentration"] = this_out_df.apply(
                            lambda row: v.get_mole_fractions(
                                {v.W(component): row["concentration"]},
                                dep_component,
                                dbf,
                            )[v.X(component)],
                            axis=1,
                        )
                        this_out_df["activity"] = this_out_df.apply(
                            lambda row: v.get_mole_fractions(
                                {v.W(component): row["activity"]}, dep_component, dbf
                            )[v.X(component)],
                            axis=1,
                        )
            this_out_df["temperatures"] = this_temperature
            this_out_df = this_out_df.dropna(axis=0, how="any")
        if out_df.empty:
            out_df = this_out_df.copy()
        else:
            out_df = pd.concat([out_df, this_out_df])

    for temperature in out_df["temperatures"].unique():
        out_sub = out_df.query("temperatures == @temperature")
        out_dict["reference_state"]["T"] = temperature
        conditions["T"] = temperature
        conditions[f"X_{component}"] = list(out_sub["concentration"].values)
        out_dict["conditions"] = conditions
        out_dict["values"] = [[list(out_sub["activity"])]]
        out_file = f"./{'-'.join([comp for comp in components if comp != 'VA'])}-ACR_{component}-"
        out_file += f"{'-'.join([phase for phase in phases if phase != ref_phase])}-"
        out_file += f"{temperature}K-"
        out_file += input_dict["bibtex"] + ".json"
        with open(out_file, "w") as f:
            json.dump(out_dict, f, indent=4)
        if espei_data_folder is not None:
            this_espei_data_folder = espei_data_folder + "activity" + out_file[1:]
            print("Saving to ESPEI-datasets at: " + this_espei_data_folder)
            with open(this_espei_data_folder, "w") as f:
                json.dump(out_dict, f, indent=4)


def write_energy_json(
    input_file,
    input_dict,
    values_dict,
    dbf,
    conditions=None,
    espei_data_folder=None,
):
    """
    Function to write an ESPEI-style JSON file for non-equilibrium thermochemical data from a WebPlotDigitizer csv passed in as a Pandas DataFrame.
    Currently assumes a standard isobaric conditions.
    TODO: Allow for isochoric phase data with varying pressure.

    Parameters: input_file, string
                    Pointer to the input csv file.
                input_dict, dictionary
                    The initial input dictionary with reference metadata and component list.
                values_dict, dictionary
                    A dictionary containing metadata for the dataset.
                dbf, pycalphad Database
                    The database to use.
                out_file, string or None, default value None
                    A string for the name of the output file. If None a default name based on the name of the input file is created.
                broadcast_conditions, boolean, default value False
                    The broadcast_conditions boolean.
                conditions, dictionary or None, default value None
                    The pycalphad conditions dictionary. If none is provided, defaults to STP conditions.

    Returns: True if success.
    """

    input_df = pd.read_csv(input_file, skiprows=[1])
    components = input_dict["components"]
    phases = input_dict["phases"]
    if espei_data_folder is not None:
        espei_data_folder = _format_espei_data_folder(espei_data_folder, components)

    for keys1, values1 in values_dict.items():
        conditions = {"P": 101325}
        out_dict = input_dict.copy()
        out_df = pd.DataFrame()
        for keys2, values2 in values1.items():
            if keys2 == "temperatures":
                out_df["temperatures"] = input_df[values2["values"]].astype("float")
                if values2["units"] == "degC":
                    out_df["temperatures"] += 273.15
                conditions["T"] = list(out_df["temperatures"].values)
            else:
                out_dict["output"] = keys2
                out_df["values"] = input_df[values2["values"]].astype("float")
                if values2["units"] == "cal/mol":
                    out_df["values"] = out_df["values"] * 4.184
                elif values2["units"] == "kcal/mol":
                    out_df["values"] = out_df["values"] * 4184
                elif values2["units"] == "cal":
                    out_df["units"] = out_df["values"] * 4.184 / values2["moles"]
                elif values2["units"] == "kcal":
                    out_df["units"] = out_df["values"] * 4184 / values2["moles"]
                elif values2["units"] == "cal/gram-atom":
                    pass  # future reservation for this conversion
                out_dict["values"] = [[[val] for val in list(out_df["values"].values)]]

        out_dict["conditions"] = conditions
        out_file = "./"
        out_file += "-".join([comp for comp in components if comp != "VA"])
        out_file += "-" + out_dict["output"] + "-"
        out_file += phases[0] + "-"
        out_file += out_dict["bibtex"] + ".json"
        with open(out_file, "w") as f:
            json.dump(out_dict, f, indent=4)
        if espei_data_folder is not None:
            this_espei_data_folder = (
                espei_data_folder
                + "non-equilibrium-thermochemical/"
                + phases[0]
                + out_file[1:]
            )
            print("Saving to ESPEI-datasets at: " + espei_data_folder)
            with open(this_espei_data_folder, "w") as f:
                json.dump(out_dict, f, indent=4)
