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
    input_file: str,
    input_dict: dict,
    values_dict: dict,
    dbf,
    broadcast_conditions=False,
    conditions=None,
):
    input_df = pd.read_csv(
        input_file, skiprows=[1]
    )  # need to skip the second row that simply contains "X" and "Y"

    for key1, value1 in values_dict.items():
        out_df = pd.DataFrame()
        out_dict = input_dict.copy()
        out_dict["broadcast_conditions"] = broadcast_conditions
        out_dict["output"] = "ZPF"
        components = input_dict["components"]
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
    return True
