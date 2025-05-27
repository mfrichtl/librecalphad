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


def write_zpf_json(
    input_file,
    input_dict,
    values_dict,
    dbf,
    out_file=None,
    broadcast_conditions=False,
    conditions=None,
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

    out_dict = input_dict.copy()
    out_dict["broadcast_conditions"] = broadcast_conditions
    out_dict["output"] = "ZPF"
    components = input_dict["components"]
    phases = []
    if conditions is None:
        conditions = {"P": 101325}  # Assume atmospheric pressure if not provided.
    values_string = []
    input_df = pd.read_csv(
        input_file, skiprows=[1]
    )  # need to skip the second row that simply contains "X" and "Y"
    for segment, value in values_dict.items():
        temperatures = input_df[input_df[value["temperatures"]["column"]].notnull()][
            value["temperatures"]["column"]
        ].values
        temp_unit = value["temperatures"]["units"]
        if temp_unit == "degC":
            temperatures += 273.15
        if "T" not in list(conditions.keys()):
            conditions["T"] = list(temperatures)
        else:
            conditions["T"].extend(list(temperatures))
        for i in range(len(temperatures)):
            temp_output = []
            segment_no = 0
            seg_nan = False
            for segment_key, segment_values in value.items():
                segment_output = []
                if segment_key == "temperatures":
                    segment_no += 1
                    continue
                else:
                    this_phase = segment_values["phase"]
                    if this_phase not in phases:
                        phases.append(this_phase)
                    segment_output.append(this_phase)
                    ind_component = segment_values["component"]
                    segment_output.append([ind_component])
                    if segment_values["values"] != "null":
                        concentration = input_df.loc[i][segment_values["values"]]
                        if pd.isnull(
                            concentration
                        ):  # blank values from WPD will be NaN
                            seg_nan = True
                            continue
                        elif segment_values["units"] == "weight_percent":
                            concentration = (
                                concentration / 100
                            )  # convert to weight fraction
                            if len(components) == 2:
                                dependent_component = [
                                    comp for comp in components if comp != ind_component
                                ][0]
                                conc_conditions = {v.W(ind_component): concentration}
                                concentration = v.get_mole_fractions(
                                    conc_conditions, dependent_component, dbf
                                )[v.X(ind_component)]
                        elif segment_values["units"] == "C_ratio":
                            concentration = convert_c_ratio(
                                input_df.loc[i][segment_values["values"]]
                            )
                            print(concentration)
                    else:
                        concentration = None
                    segment_output.append([concentration])
                temp_output.append(segment_output)
                segment_no += 1
            if not seg_nan:
                values_string.append(temp_output)
    out_dict["phases"] = phases
    out_dict["conditions"] = conditions
    out_dict["values"] = values_string

    if out_file is None:
        out_file = input_file[:-3] + "JSON"
    with open(out_file, "w") as f:
        json.dump(out_dict, f)
    return True
