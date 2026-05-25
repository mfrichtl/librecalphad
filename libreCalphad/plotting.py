"""
Functions to help make useful plots.
"""

import libreCalphad.models.heat_capacity as hc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pycalphad import calculate, Database, equilibrium, variables as v
from pycalphad.plot.utils import phase_legend
import seaborn as sns
import sympy as sp
from tinydb import where


def plot_heat_capacity_from_models(
    model_dict: dict[str, list[float | int] | str],
    datasets,
    phase: list[str] | None = None,
    components: list[str] | None = None,
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
):
    """
    Function to plot modeled heat capacity versus provided datasets. The provided datasets can either be a query containing the datasets you want to plot or, if the phase and components are provided as keyword arguments, the query will be run.
    """
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
            keyword_args["expression"] = sp.parse_expr(params["expression"])
            keyword_args["symbols"] = params["symbols"]
            for symbol in params["symbols"]:
                keyword_args[symbol] = params[symbol]
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
        fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=exp_df, x="T", y="CPM", hue="reference", ax=ax)
    sns.lineplot(data=df_model, x="T", y="CPM", hue="model", ax=ax)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Heat Capacity (J/mol/K)")
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), ncol=2)
    fig.tight_layout()
    return fig, ax


def plot_calculated_heat_capacity(
    dbf: Database,
    components: list[str],
    phases: list[str],
    conditions: dict[v.StateVariable, float | set[int]],
    datasets: list[dict[str, str]] = None,
    fig=None,
    ax=None,
):
    if all([fig is None, ax is None]):
        fig, ax = plt.subplots(figsize=(8, 6))
    eq_res = equilibrium(dbf, components, phases, conditions, output="heat_capacity")
    search_results = None
    for phase in phases:
        cpm_res = calculate(
            dbf,
            components,
            phase,
            T=conditions[v.T],
            P=conditions[v.P],
            N=1,
            output="heat_capacity",
        )
        ax.plot(cpm_res.T, cpm_res.heat_capacity.squeeze(), label=phase)
        if datasets is not None:
            if not isinstance(datasets, list):
                query = (
                    (where("phases") == phase)
                    & (where("components") == components)
                    & (where("output") == "CPM")
                )
                search_results = datasets.search(query)
            else:
                search_results = datasets
            if search_results is not None:
                for result in search_results:
                    ax.scatter(
                        result["conditions"]["T"],
                        np.array(result["values"]).squeeze(),
                        label=result["reference"],
                    )

    if len(phases) > 1:
        ax.plot(eq_res.T, eq_res.heat_capacity.squeeze(), label="equilibrium")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_calculated_gibbs_energies(
    dbf,
    components,
    phases,
    conditions,
    datasets=None,
    print_transition_temps=False,
    fig=None,
    ax=None,
):
    if all([fig is None, ax is None]):
        fig, ax = plt.subplots(figsize=(8, 6))
    eq_res = equilibrium(dbf, components, phases, conditions)
    if print_transition_temps:
        active_phase = (
            eq_res.where(eq_res.T == eq_res.T[0]).Phase.squeeze().values.flatten()[0]
        )
        print(f"Initial stability {active_phase}")
        for temp in eq_res.T:
            for val in eq_res.where(eq_res.T == temp).Phase.squeeze().values.flatten():
                if val == active_phase:
                    continue
                elif isinstance(val, str):
                    if val == "":
                        continue
                    else:
                        active_phase = val
                        print(f"Transition to {val} at {temp.T.values} K")
                else:
                    try:
                        np.isnan(val)
                    except:
                        breakpoint()
    search_results = None
    for phase in phases:
        if datasets is not None:
            if not isinstance(datasets, list):
                query = (
                    (where("phases") == [phase])
                    & (where("components") == components)
                    & (where("output") == "GM")
                )
                search_results = datasets.search(query)
        else:
            search_results = datasets
        res = calculate(
            dbf,
            components,
            phase,
            T=conditions[v.T],
            P=conditions[v.P],
            N=conditions[v.N],
        )
        ax.plot(res.T, res.GM.squeeze(), label=phase)
        if search_results is not None:
            for result in search_results:
                ax.scatter(
                    result["conditions"]["T"],
                    np.array(result["values"]).squeeze(),
                    label=f"{phase}-{result['reference']}",
                )

    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Gibbs Energy (J/mol-formula)")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    return fig, ax


def step_plot(db, components, conditions, disabled_phases=[], **fig_kw):
    """
    Function to make a step plot.
    """

    fig, ax = plt.subplots(**fig_kw)

    phases = [phase for phase in list(db.phases.keys()) if phase not in disabled_phases]
    eq = equilibrium(db, components, phases, conditions)
    eq_phases = []
    for entry in eq.Phase.squeeze():
        for phase in entry.values:
            if phase != "" and phase not in eq_phases:
                eq_phases.append(phase)
    phase_handles, phasemap = phase_legend(eq_phases)

    for name in eq_phases:
        phase_indices = np.nonzero(eq.Phase.values == name)
        plt.scatter(
            np.take(eq["T"].values, phase_indices[2]),
            eq.NP.values[phase_indices],
            color=phasemap[name],
        )
        ax.legend(phase_handles, eq_phases, loc="lower right")

    return fig, ax
