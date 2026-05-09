# Plotting functions
import libreCalphad.models.heat_capacity as hc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tinydb import where


def plot_heat_capacity_from_models(
    model_dict, datasets, phase=None, components=None, fig=None, ax=None
):
    function_dict = {
        "bcm": hc._bent_cable_Cp,
        "einstein": hc._einstein_Cp,
        "holzapfel": hc._holzapfel_debye_Cp,
        "linear": hc._linear_Cp,
        "melt": hc._melt_Cp,
        "symbolic": hc._symbolic_Cp,
        "two-state": hc._twostate_Cp,
        "xiong": hc._xiong_Cp,
    }

    if not isinstance(datasets, list):
        # Need to construct a query
        query = (
            (where("phases") == phase)
            & (where("components") == components)
            & (where("output") == "CPM")
        )
        search_results = datasets.search(query)
    else:
        search_results = datasets.copy()

    exp_df = pd.DataFrame()
    for dataset in search_results:
        data_dict = {
            "T": dataset["conditions"]["T"],
            "CPM": np.array(dataset["values"]).squeeze(),
            "reference": dataset["reference"],
        }
        exp_df = pd.concat([exp_df, pd.DataFrame(data_dict)])
    df_model = pd.DataFrame()
    temp_array = np.linspace(0, np.max(exp_df["T"]) + 500, num=1000)
    combined_Cp = np.zeros(len(temp_array))

    for model, params in model_dict.items():
        keyword_args = {"T_arr": temp_array}
        cpm_dict = {"T": temp_array, "model": model}
        if "symbolic" in model:
            for kwarg, value in params.items():
                if kwarg in ["expression", "temp_bounds", "variable_values"]:
                    keyword_args[kwarg] = value
                else:
                    continue
            model_Cp = hc._symbolic_Cp(**keyword_args)
        elif model == "offset":
            continue
        elif model == "two-state":
            keyword_args["dE"] = params["dE"][0]
            keyword_args["coef_list"] = params["dE"][1]
            model_Cp = hc._twostate_Cp(**keyword_args)
        else:
            for kwarg, value in params.items():
                keyword_args[kwarg] = value[0]
            model_Cp = function_dict[model](**keyword_args)
        cpm_dict["CPM"] = model_Cp
        df_model = pd.concat([df_model, pd.DataFrame(cpm_dict)])
        combined_Cp = combined_Cp + model_Cp
    cpm_dict = {"T": temp_array, "model": "combined", "CPM": combined_Cp}
    df_model = pd.concat([df_model, pd.DataFrame(cpm_dict)])

    if fig is None:
        fig, ax = plt.subplots()
    sns.scatterplot(data=exp_df, x="T", y="CPM", hue="reference", ax=ax)
    sns.lineplot(data=df_model, x="T", y="CPM", hue="model", ax=ax)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Heat Capacity (J/mol/K)")
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), ncol=2)
    fig.tight_layout()
    return fig, ax
