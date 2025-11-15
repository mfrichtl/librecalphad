from copy import deepcopy
import json
from libreCalphad.models.thermodynamics import fit_regular_solution_model
import numpy as np
import pandas as pd
from pycalphad import Database, equilibrium, variables as v
from scipy.optimize import curve_fit, minimize


R = 8.314463  # J/mol K


def _convert_c_ratio(y_c: float) -> float:
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
        out_dict = deepcopy(input_dict)
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
                            lambda row: _convert_c_ratio(row[value2["values"]]), axis=1
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
                elif value2["units"] == "mixed":
                    # assume mixed units with unit definitions following |
                    # This may be reported for gas equilibrium experiments
                    concentration_unit = value2["concentration"].split("|")[1]
                    if any(
                        [
                            concentration_unit == "weight_percent",
                            concentration_unit == "wt%",
                        ]
                    ):
                        this_out_df["concentration"] = (
                            this_out_df["concentration"] / 100
                        )
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
                    activity_unit = value2["activity"].split("|")[1]
                    if any([activity_unit == "weight_percent", activity_unit == "wt%"]):
                        if len(components) == 2 or (
                            len(components) == 3 and "VA" in components
                        ):
                            dep_component = [
                                comp
                                for comp in components
                                if comp != "VA" and comp != component
                            ][0]
                            this_out_df["activity"] = this_out_df.apply(
                                lambda row: v.get_mole_fractions(
                                    {v.W(component): row["activity"]},
                                    dep_component,
                                    dbf,
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
        out_dict["reference_state"]["conditions"]["T"] = temperature
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
    input_data,
    input_dict,
    values_dict,
    dbf=None,
    conditions=None,
    espei_data_folder=None,
):
    """
    Function to write an ESPEI-style JSON file for non-equilibrium thermochemical data from a WebPlotDigitizer csv or a Pandas DataFrame.
    Currently assumes a standard isobaric conditions.
    TODO: Allow for isochoric phase data with varying pressure.

    Parameters: input_data, string or pandas.DataFrame
                    Pointer to the input csv file or a DataFrame containing columns identified in values_dict.
                input_dict, dictionary
                    The initial input dictionary with reference metadata and component list.
                values_dict, dictionary
                    A dictionary containing metadata for the dataset.
                dbf, pycalphad Database
                    The database to use.
                out_file, string or None, default value None
                    A string for the name of the output file. If None a default name based on the name of the input file is created.
                conditions, dictionary or None, default value None
                    The pycalphad conditions dictionary. If none is provided, defaults to STP conditions.

    Returns: True if success.
    """

    if isinstance(input_data, str):
        input_df = pd.read_csv(input_data, skiprows=[1])
    else:
        input_df = input_data
    if isinstance(dbf, str):
        dbf = Database(dbf)
    components = input_dict["components"]
    phases = input_dict["phases"]
    # normalize to atoms per mole formula unit
    if "atoms" in list(input_dict.keys()):
        atoms = input_dict["atoms"]
    else:
        atoms = 1
    if espei_data_folder is not None:
        espei_data_folder = _format_espei_data_folder(espei_data_folder, components)

    # Define non-mixing components in conditions in input dictionary
    component_dict = {}
    if "conditions" in list(input_dict.keys()):
        for condition in list(input_dict.keys()):
            if condition.startswith("X_"):
                component_dict[condition] = input_dict[condition.split("_")[1]]
    mixing_components = []

    calc = False
    site_fracs = {}
    out_conditions = input_dict["conditions"].copy()
    mixing = False

    for keys1, values1 in values_dict.items():
        for key, value in values1.items():
            if key.startswith("X_") and isinstance(value, (int, float)):
                component_dict[key] = value
        out_dict = deepcopy(
            input_dict
        )  # need deep copy here to avoid mutating the input dict later
        out_df = pd.DataFrame()
        if "model" in list(values1.keys()):
            calc = True
        for keys2, values2 in values1.items():
            if keys2 == "temperatures":
                out_df["temperatures"] = input_df[values2["values"]].astype("float")
                if values2["units"] == "degC":
                    out_df["temperatures"] += 273.15
                out_conditions["T"] = list(out_df["temperatures"].values)
            elif keys2.startswith("X_"):  # mixing concentrations passed
                assert dbf is not None, (
                    "Need to pass a pycalphad database file to calculate site fractions."
                )
                assert len(input_dict["phases"]) == 1, (
                    "More than one phase passed for mixing calculation."
                )
                if "mixing_sublattice" in list(values2.keys()):
                    mixing = True
                mixing_component = keys2.split("_")[1]
                mixing_components.append(mixing_component)
                if values2["units"] == "molar_fraction":
                    component_dict[mixing_component] = input_df[
                        values2["values"]
                    ].values
                else:
                    raise NotImplementedError(
                        f"{values2['units']} not implemented yet."
                    )
                out_df[keys2] = component_dict[mixing_component]
            elif keys2 == "model":
                continue
            else:
                out_dict["output"] = keys2
                if calc:
                    if "regular" in list(values1.values()):
                        fits, input_df = fit_regular_solution_model(input_df)

                out_df["values"] = input_df[values2["values"]].astype("float")
                if values2["units"] == "cal/mol":
                    out_df["values"] = out_df["values"] * 4.184
                elif values2["units"] == "kcal/mol":
                    out_df["values"] = out_df["values"] * 4184
                elif values2["units"] == "cal":
                    out_df["values"] = out_df["values"] * 4.184
                elif values2["units"] == "kcal":
                    out_df["values"] = out_df["values"] * 4184
                elif values2["units"] == "cal/gram-atom":
                    pass  # future reservation for this conversion
                elif values2["units"] == "cal/mol/K":
                    out_df["values"] = out_df["values"] * 4.184
                # normalize to atoms per mole-formula unit
                out_df["values"] = out_df["values"] / atoms

                if "mixing_sublattice" in list(values2.keys()):
                    mixing = values2["mixing"]

            if mixing:
                if "conditions" in list(input_dict.keys()):
                    eq_conditions = input_dict["conditions"].copy()
                else:
                    eq_conditions = {}
                for cond, value in input_dict["conditions"].items():
                    if cond == "P":
                        eq_conditions[v.P] = value
                    elif cond == "N":
                        eq_conditions[v.N] = value
                max_length = 0
                for component, concentration in component_dict.items():
                    if isinstance(concentration, np.ndarray):
                        if len(concentration) > max_length:
                            max_length = len(concentration)

                sublattice_configurations = []
                sublattice_occupancies = []
                for i in out_df.index:
                    for component, concentration in component_dict.items():
                        if isinstance(concentration, np.ndarray):
                            eq_conditions[v.X(component)] = concentration[i]
                            out_df.loc[i, f"X_{component}"] = concentration[i]
                        else:
                            eq_conditions[v.X(component)] = concentration
                            out_df.loc[i, f"X_{component}"] = concentration
                        eq_conditions[v.T] = out_df.iloc[i]["temperatures"]
                    eq = equilibrium(dbf, components, phases, eq_conditions)

                    # Build sublattice occupancies array, assuming components in sublattice_configurations array are
                    # sorted in same order as pycalphad equilibrium. This may be wrong.
                    j = 0
                    for sl in input_dict["solver"]["sublattice_configurations"]:
                        if isinstance(sl, str):  # single-component sublattice
                            this_site_frac = float(
                                eq.sel(vertex=0).Y.squeeze().values[j]
                            )
                            out_df.loc[i, f"Y_{sl}"] = this_site_frac
                            j += 1
                        else:
                            for component in sl:
                                if j == 3:
                                    breakpoint()
                                this_site_frac = float(
                                    eq.sel(vertex=0).Y.squeeze().values[j]
                                )
                                out_df.loc[i, f"Y_{component}"] = this_site_frac
                                j += 1
                mixing = False
        out_conditions["T"] = []
        sublattice_configurations = []
        sublattice_occupancies = []
        out_values = []
        for temp in out_df["temperatures"].unique():
            temp_df = out_df.query("temperatures == @temp")
            out_conditions["T"] = temp
            this_temp_occ = []
            this_temp_conf = []
            this_temp_values = []
            for i in temp_df.index:
                this_occ = []
                for sl in input_dict["solver"]["sublattice_configurations"]:
                    if isinstance(sl, str):
                        this_occ.append(out_df.iloc[i][f"Y_{sl}"])
                    else:
                        sl_occ = []
                        for component in sl:
                            sl_occ.append(out_df.iloc[i][f"Y_{component}"])
                        this_occ.append(sl_occ)
                this_temp_occ.append(this_occ)
                this_temp_conf.append(input_dict["solver"]["sublattice_configurations"])
                this_temp_values.append(out_df.iloc[i]["values"])
            out_values.append([this_temp_values])
            sublattice_configurations.append(this_temp_conf)
            sublattice_occupancies.append(this_temp_occ)

            out_dict["conditions"] = out_conditions
            out_dict["solver"]["sublattice_occupancies"] = this_temp_occ
            out_dict["solver"]["sublattice_configurations"] = this_temp_conf
            out_dict["values"] = [[this_temp_values]]
            out_file = "./"
            out_file += "-".join([comp for comp in components if comp != "VA"])
            out_file += "-" + out_dict["output"] + "-"
            out_file += phases[0] + "-"
            out_file += f"{temp}K-"
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
        # TODO: Separate the output files by temperature, like I had to do with the energy files above.
        # out_dict["conditions"] = out_conditions
        # out_dict["solver"]["sublattice_occupancies"] = sublattice_occupancies
        # out_dict["solver"]["sublattice_configurations"] = sublattice_configurations
        # out_dict["values"] = [out_values]
        # out_file = "./"
        # out_file += "-".join([comp for comp in components if comp != "VA"])
        # out_file += "-" + out_dict["output"] + "-"
        # out_file += phases[0] + "-"
        # out_file += out_dict["bibtex"] + ".json"
        # with open(out_file, "w") as f:
        #     json.dump(out_dict, f, indent=4)
        # if espei_data_folder is not None:
        #     this_espei_data_folder = (
        #         espei_data_folder
        #         + "non-equilibrium-thermochemical/"
        #         + phases[0]
        #         + out_file[1:]
        #     )
        #     print("Saving to ESPEI-datasets at: " + espei_data_folder)
        #     with open(this_espei_data_folder, "w") as f:
        #         json.dump(out_dict, f, indent=4)
