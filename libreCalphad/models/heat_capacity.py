"""
Segmented regression approach to modeling the heat capacity of a solid described by Roslyakova et
al. [1]. Uses ESPEI to load experimental datasets of CPM measurements for fitting. At low
temperatures, either the Debye or Einstein models can be used. The Holzapfel model [2] is used to
create an analytical expression for the Debye model. If a magnetic transition takes place,
the magnetic model developed by Xiong et al. [2] is used. Parameters for this model can be provided
by a pycalphad Database object or directly provided.

Notes: Assuming alpha == tau in [1]

    References:
    [1] - I. Roslyakova, B. Sundman, H. Dette, L. Zhang, and I. Steinbach, “Modeling of Gibbs
    energies of pure elements down to 0 K using segmented regression,” Calphad, vol. 55,
    pp. 165–180, Dec. 2016, doi: 10.1016/j.calphad.2016.09.001.

    [2] - E. Brosh, “Application of Holzapfel’s Analytic Expression for Approximating the Debye
    Heat Capacity Function in CALPHAD Databases,” Journal of Phase Equilibria and Diffusion, vol. 46,
    no. 4, pp. 357–365, Jul. 2025, doi: 10.1007/s11669-025-01201-7.

    [3] - W. Xiong, Q. Chen, P. A. Korzhavyi, and M. Selleby, “An improved magnetic model for
    thermodynamic modeling,” Calphad, vol. 39, pp. 11–20, 2012.

TODO: Investigate adding a cutoff temperature to the bent-cable model so that it doesn't start from zero
and introduce error to the Debye model at low temperatures.
"""

from collections import OrderedDict
from libreCalphad.databases.db_utils import load_database
from libreCalphad.models.utilities import identify_variables
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pycalphad import calculate, variables as v
from scipy.integrate import quad
from scipy.optimize import curve_fit, minimize
import symengine as se
import sympy as sp
from tinydb import where
import warnings
import yaml

R = 8.314472  # J/mol/K
N = 6.02214076e23
P, T = se.symbols("P T")
# Holzapfel parameters
w0 = 5.657e-2
w1 = 3.911e-2
f1 = 4.155e-1
f2 = 8.116e-1
w2 = 9.0432e-1
aD = 5.133e-2


def _symbolic_Cp(T_arr=0, variable_values=[], expression="", temp_bounds=()):
    if isinstance(expression, str):
        expression = sp.parse_expr(expression)
        expression = se.sympify(expression)
    # Assuming the other symbols besides T and P are variables to be fit
    expr_symbols = [symbol for symbol in expression.free_symbols]

    if isinstance(T_arr, (int, float)):
        if all([T_arr > temp_bounds[0], T_arr <= temp_bounds[1]]):
            symbol_values = identify_variables(expr_symbols, variable_values, T_arr)
            ret_arr = np.float64(expression.subs(expr_symbols, symbol_values))
        else:
            ret_arr = 0
    else:
        ret_arr = np.array([])
        for temp in T_arr:
            if all([temp > temp_bounds[0], temp <= temp_bounds[1]]):
                symbol_values = identify_variables(expr_symbols, variable_values, temp)
                Cp = np.float64(expression.subs(expr_symbols, symbol_values))
            else:
                Cp = 0
            ret_arr = np.append(ret_arr, Cp)
    return ret_arr


def _twostate_Cp(T_arr=0, dE=0, coef_list=[]):
    # Following the approach from Becker2013
    g0 = 1
    g1 = 1

    if isinstance(dE, list):
        dE = np.sum([dE[i] * coef_list[i] for i in range(len(dE))])

    def _calc_twostate_Cp(temp, dE):
        dH = -(T**2) * (dE / T).diff(T)
        chi = se.exp(-dE / (R * T)) / (1 + se.exp(-dE / R * T))
        twostate_Cp = (
            chi * dH.diff(T)
            + R
            * (dH / (R * T)) ** 2
            * se.exp(-dE / (R * T))
            / (1 + se.exp(-dE / (R * T))) ** 2
        )
        return np.float64(twostate_Cp.subs(T, temp))

    if isinstance(T_arr, (int, float)):
        ret_arr = _calc_twostate_Cp(T_arr, dE)
    else:
        ret_arr = np.array([])
        for temp in T_arr:
            ret_arr = np.append(ret_arr, _calc_twostate_Cp(temp, dE))
    ret_arr = np.nan_to_num(ret_arr)
    return ret_arr


def _holzapfel_debye_Cp(T_arr=0, thetaD=0):
    def _calc_holzapfel_Cp(x):
        if x == 0:
            return 0
        term1 = 4 / aD * x**3 / (1 + x**3 / (aD * w0))
        term2 = -3 / (aD**2 * w0) * x**6 / (1 + x**3 / (aD * w0)) ** 2
        term3 = w1 * (f1 / x) ** 2 * np.exp(-f1 / x) / (1 - np.exp(-f1 / x)) ** 2
        term4 = w2 * (f2 / x) ** 2 * np.exp(-f2 / x) / (1 - np.exp(-f2 / x)) ** 2
        return 3 * R * np.sum([term1, term2, term3, term4])

    if isinstance(T_arr, (int, float)):
        x = T_arr / thetaD
        ret_arr = _calc_holzapfel_Cp(x)

    else:
        ret_arr = np.array([])
        for temp in T_arr:
            x = temp / thetaD
            Cp = _calc_holzapfel_Cp(x)
            ret_arr = np.append(ret_arr, Cp)
    ret_arr = np.nan_to_num(ret_arr)
    return ret_arr


def _melt_Cp(T_arr=0, T_melt=0, a=0, b=0, c=0, **kwargs):
    # following the approach of Chen and Sundman (2001)
    if isinstance(T_arr, (int, float)):
        if T_arr <= T_melt:
            ret_arr = 0
        else:
            ret_arr = a + b * T_arr**-6 + c * T_arr**-12
    else:
        ret_arr = np.array([])
        for temp in T_arr:
            if temp <= T_melt:
                Cp = 0
            else:
                Cp = a + b * temp**-6 + c * temp**-12
            ret_arr = np.append(ret_arr, Cp)
    return ret_arr


def _linear_Cp(T_arr=0, alpha=0, T_melt=0):
    if isinstance(T_arr, (int, float)):
        if T_arr <= T_melt:
            ret_arr = alpha * T_arr
        else:
            ret_arr = 0
    else:
        ret_arr = np.array([])
        for temp in T_arr:
            if temp <= T_melt:
                Cp = alpha * temp
            else:
                Cp = 0
            ret_arr = np.append(ret_arr, Cp)
    return ret_arr


def _bent_cable_Cp(T_arr=0, beta_1=0, beta_2=0, tau=0, gamma=0, T_melt=0):
    def _q(temp, tau, gamma):
        def _indfunc1(temp, tau, gamma):
            if np.abs(temp - tau) <= gamma:
                return 1
            else:
                return 0

        def _indfunc2(temp, tau, gamma):
            if temp > tau + gamma:
                return 1
            else:
                return 0

        return ((temp - tau + gamma) ** 2 / (4 * gamma)) * _indfunc1(
            temp, tau, gamma
        ) + (temp - tau) * _indfunc2(temp, tau, gamma)

    if isinstance(T_arr, (int, float)):
        if T_arr == 0 or T_arr > T_melt:
            return 0
        return beta_1 * T_arr + beta_2 * _q(T_arr, tau, gamma)
    else:
        ret_arr = np.array([])
        for temp in T_arr:
            if temp == 0 or temp > T_melt:
                Cp = 0
            else:
                Cp = beta_1 * temp + beta_2 * _q(temp, tau, gamma)
            ret_arr = np.append(ret_arr, Cp)
        return ret_arr


def _einstein_Cp(T_arr=0, theta=0):
    if isinstance(T_arr, (int, float)):
        if T_arr == 0:
            ret_arr = 0
    else:
        ret_arr = np.array([])
        for temp in T_arr:
            if temp == 0:
                Cp = 0
            else:
                Cp = (
                    3
                    * R
                    * (theta / temp) ** 2
                    * np.exp(theta / temp)
                    / (np.exp(theta / temp) - 1) ** 2
                )
            ret_arr = np.append(ret_arr, Cp)
    ret_arr = np.nan_to_num(ret_arr)  # replace nan with 0
    return ret_arr


def _debye_Cp(T_arr=0, theta=0):
    def _integrand(x):
        return x**4 * np.exp(x) / (np.exp(x) - 1) ** 2

    if isinstance(T_arr, (int, float)):
        ret_arr = 9 * R * (T_arr / theta) ** 3 * quad(_integrand, 0, theta / T_arr)
    else:
        ret_arr = np.array([])
        for temp in T_arr:
            Cp = 9 * R * (temp / theta) ** 3 * quad(_integrand, 0, theta / temp)[0]
            ret_arr = np.append(ret_arr, Cp)
    return ret_arr


def _xiong_Cp(T_arr=0, beta=0, p=0, Tc=0):
    def _D(p):
        return 0.33471979 + 0.49649686 * (1 / p - 1)

    def _g(tau, p):
        if tau <= 0:
            return 0
        elif tau <= 1:
            return (
                0.63570895
                / _D(p)
                * (1 / p - 1)
                * (2 * tau**3 + 2 / 3 * tau**9 + 2 / 5 * tau**15 + 2 / 7 * tau**21)
            )
        else:
            return (
                1
                / _D(p)
                * (2 * tau**-7 + 2 / 3 * tau**-21 + 2 / 5 * tau**-35 + 2 / 7 * tau**-49)
            )

    if isinstance(T_arr, (int, float)):
        if T_arr == 0:
            return 0
        tau = T_arr / Tc
        return R * tau * _g(tau, p) * np.log(beta + 1)
    else:
        ret_arr = np.array([])
        for temp in T_arr:
            if temp == 0:
                Cp = 0
            else:
                tau = temp / Tc
                Cp = R * tau * _g(tau, p) * np.log(beta + 1)
            ret_arr = np.append(ret_arr, Cp)
        return ret_arr


def _calc_RSE(model_array, Cp_array, x):
    # error = np.sqrt(np.mean(np.square(np.nan_to_num((model_array - Cp_array)))))
    # error = np.sqrt(
    #     np.sum(np.square(np.nan_to_num(model_array - Cp_array)))
    #     / (len(Cp_array) - len(x) - 1)
    # )
    error = np.sqrt(
        np.sum(np.square((np.nan_to_num((model_array - Cp_array) / Cp_array))))
    )
    return error


def _fit_heat_capacity(x, arg_dict):
    """
    Function to perform the segmented regression to optimize all modeling parameters.
    """

    temperature_array = arg_dict["temperature_array"]
    Cp_array = arg_dict["Cp_array"]
    models = arg_dict["models"]
    melt_base_Cp = None

    model_Cp = np.zeros(len(Cp_array))
    if "einstein" in list(models.keys()):
        model_params = []
        einstein_dict = models["einstein"]
        theta_list = einstein_dict["theta"]
        if "fit" in theta_list:
            idx = theta_list[-1]
            model_Cp += _einstein_Cp(T_arr=temperature_array, theta=x[idx])
        else:
            model_Cp += _einstein_Cp(T_arr=temperature_array, theta=theta_list[0])
    elif "holzapfel" in list(models.keys()):
        model_params = []
        holzapfel_dict = models["holzapfel"]
        theta_list = holzapfel_dict["theta"]
        if "fit" in theta_list:
            idx = theta_list[-1]
            model_Cp += _holzapfel_debye_Cp(T_arr=temperature_array, thetaD=x[idx])
        else:
            model_Cp += _holzapfel_debye_Cp(
                T_arr=temperature_array, thetaD=theta_list[0]
            )
    for symbolic_key in [key for key in list(models.keys()) if "symbolic" in key]:
        symbolic_dict = models[symbolic_key]
        model_params = []
        for param, param_list in symbolic_dict.items():
            if param in ["expression", "param_bounds", "temp_bounds"]:
                continue
            idx = param_list[-1]
            model_params.append(x[idx])
        model_Cp = model_Cp + _symbolic_Cp(
            T_arr=temperature_array,
            variable_values=model_params,
            expression=symbolic_dict["expression"],
            temp_bounds=symbolic_dict["temp_bounds"],
        )
    if "xiong" in list(models.keys()):
        model_params = []
        # Need beta, p, and Tc (or Tn)
        xiong_dict = models["xiong"]
        beta_list = xiong_dict["beta"]
        if "fit" in beta_list:
            idx = beta_list[-1]
            beta = x[idx]
        else:
            beta = beta_list[0]

        structure_factor_list = xiong_dict["p"]
        if "fit" in structure_factor_list:
            idx = structure_factor_list[-1]
            structure_factor = x[idx]
        else:
            structure_factor = structure_factor_list[0]

        if "Tc" in list(xiong_dict.keys()):
            temp = "Tc"
        else:
            temp = "Tn"
        critical_temp_list = xiong_dict[temp]
        if "fit" in critical_temp_list:
            idx = critical_temp_list[-1]
            critical_temp = x[idx]
        else:
            critical_temp = critical_temp_list[0]
        model_Cp += _xiong_Cp(
            T_arr=temperature_array, beta=beta, p=structure_factor, Tc=critical_temp
        )
    if "bcm" in list(models.keys()):
        bcm_dict = models["bcm"]
        model_params = {}
        for param in ["beta_1", "beta_2", "tau", "gamma", "T_melt"]:
            param_list = bcm_dict[param]
            if "fit" in param_list:
                idx = param_list[-1]
                model_params[param] = x[idx]
            else:
                model_params[param] = param_list[0]
        model_Cp += _bent_cable_Cp(T_arr=temperature_array, **model_params)
    if "two-state" in list(models.keys()):
        param_list = models["two-state"]["dE"]
        model_params = {"dE": [], "coef_list": param_list[1]}
        if "fit" in param_list:
            if isinstance(param_list[0], (float, int)):
                idx = param_list[-1]
                model_params["dE"].append(x[idx])
            else:
                idx_list = param_list[-1]
                for j in idx_list:
                    model_params["dE"].append(x[j])
        else:
            if isinstance(param_list[0], (float, int)):
                model_params["dE"].append(param_list[0])
            else:
                for param in param_list[0]:
                    model_params["dE"].append(param)
        model_Cp += _twostate_Cp(T_arr=temperature_array, **model_params)
    if "linear" in list(models.keys()):
        linear_dict = models["linear"]
        model_params = {}
        for param in ["alpha", "T_melt"]:
            param_list = linear_dict[param]
            if "fit" in param_list:
                idx = param_list[-1]
                model_params[param] = x[idx]
            else:
                model_params[param] = param_list[0]
        model_Cp += _linear_Cp(T_arr=temperature_array, **model_params)

    if "melt" in list(models.keys()):
        # Now add the high-temp heat capacity model
        melt_dict = models["melt"]
        model_params = {}
        for param in ["T_melt", "a", "b", "c"]:
            param_list = melt_dict[param]
            if "fit" in param_list:
                idx = param_list[-1]
                model_params[param] = x[idx]
            else:
                model_params[param] = param_list[0]
        if "liquid_Cp" not in list(melt_dict.keys()):
            melt_base_Cp = np.max(model_Cp)
        else:
            melt_base_Cp = melt_dict["liquid_Cp"][0]
        assert melt_base_Cp is not None, (
            "Need to specify a base melt heat capacity in another method."
        )
        model_Cp += _melt_Cp(T_arr=temperature_array, **model_params)
    return _calc_RSE(model_Cp, Cp_array, x)


def fit_heat_capacity(datasets, models, verbose=False):
    implemented_models = [
        "bcm",
        "einstein",
        "holzapfel",
        "linear",
        "melt",
        "offset",
        "symbolic",
        "two-state",
        "xiong",
    ]
    for model in list(models.keys()):
        if all([model not in implemented_models, "symbolic" not in model]):
            raise NotImplementedError(
                f"{model} not implemented. Options are {implemented_models}"
            )

    if not isinstance(datasets, pd.DataFrame):
        # Load CPM data for modeling
        df = pd.DataFrame()
        # CPM plotting
        assert len(datasets) > 0, "No datasets passed."
        for result in datasets:
            assert result["output"] == "CPM", (
                f"Can only fit CPM data, but {result['output']} was passed."
            )
            temps = result["conditions"]["T"]
            Cp = result["values"]
            this_df = pd.DataFrame(
                {
                    "temperature": temps,
                    "Cp": np.array(Cp).squeeze(),
                    "reference": result["reference"],
                }
            )
            if df.empty:
                df = this_df
            else:
                df = pd.concat([df, this_df], ignore_index=True)
    else:
        df = datasets
    df = df.sort_values("temperature")

    if "melt" in list(models.keys()):
        # Need to pad the Cp array to include some liquid temperatures and target Cp
        melt_dict = models["melt"]
        T_melt = melt_dict["T_melt"][0]
        if "liquid_Cp" in list(melt_dict.keys()):
            liq_Cp = melt_dict["liquid_Cp"][0]
        else:
            liq_Cp = df["Cp"][-1]
        while len(df.query("temperature > @T_melt")) < 5:
            new_entry = {
                "temperature": [np.max(df["temperature"].values) + 300],
                "Cp": [liq_Cp],
                "reference": "melt",
            }
            df = pd.concat([df, pd.DataFrame(new_entry)])

    arg_index = []
    params = []
    arg_dict = {
        "temperature_array": df["temperature"].values,
        "Cp_array": df["Cp"].values,
    }
    default_bounds = {
        "theta": (0, 1500),
        "beta": (0, 5),
        "p": (0, 1),
        "Tc": (0, 3000),
        "Tn": (0, 3000),
        "beta_1": (0, 1),
        "beta_2": (0, 1),
        "tau": (0, 3000),
        "gamma": (1e-6, 200),
        "dE0": (0, 15000),
        "dE1": (-100, 100),
        "dE2": (-100, 100),
        "alpha": (0, 1),
        "T_melt": (250, 4000),
        "Cp_max": (5, 100),
        "a": (-50, 50),
        "b": (-np.inf, np.inf),
        "c": (-np.inf, np.inf),
    }
    bounds = []
    i = 0
    if all(["einstein" in list(models.keys()), "holzapfel" in list(models.keys())]):
        raise ValueError(
            "Cannot specify both Einstein and Holzapfel (Debye) models. Pick one."
        )
    if "einstein" in list(models.keys()):
        einstein_model_dict = models["einstein"]
        assert "theta" in list(einstein_model_dict.keys()), (
            "Einstein model specified without instructions for fitting theta."
        )
        # param list format: [value, fit/fix
        param_list = einstein_model_dict["theta"]
        if "fit" in param_list:
            params.append(param_list[0])
            if len(param_list) == 3:
                bounds.append(param_list[2])
            else:
                bounds.append(default_bounds["theta"])
            param_list.append(i)
            i += 1
        elif "fix" in param_list:
            pass
        else:
            raise ValueError(f"Einstein model specified, but not set correctly.")
    if "holzapfel" in list(models.keys()):
        holzapfel_model_dict = models["holzapfel"]
        assert "theta" in list(holzapfel_model_dict.keys()), (
            "Holzapfel-Debye model specified without instructions for fitting theta."
        )
        if "theta" in list(holzapfel_model_dict.keys()):
            # param list format: [value, fit/fix
            param_list = holzapfel_model_dict["theta"]
            if "fit" in param_list:
                params.append(param_list[0])
                if len(param_list) == 3:
                    bounds.append(param_list[2])
                else:
                    bounds.append(default_bounds["theta"])
                param_list.append(i)
                i += 1
            elif "fix" in param_list:
                pass  # nothing to do
            else:
                raise ValueError(
                    f"Holzapfel-Debye model specified, but not set correctly."
                )
    if "xiong" in list(models.keys()):
        xiong_model_dict = models["xiong"]
        assert all(
            [param in list(xiong_model_dict.keys()) for param in ("beta", "p")]
        ), (
            f"Xiong model requires 'beta' and 'p' parameters, you passed {list(xiong_model_dict.keys())}."
        )
        assert any(
            [param in list(xiong_model_dict.keys()) for param in ("Tc", "Tn")]
        ), (
            f"Xiong model requires 'Tc' or 'Tn', you passed {list(xiong_model_dict.keys())}."
        )
        assert not all(
            [param in list(xiong_model_dict.keys()) for param in ("Tc", "Tn")]
        ), "Cannot specify both Tc and Tn for Xiong model."
        for param in ["beta", "p", "Tc", "Tn"]:
            if param == "Tc" and not "Tc" in list(xiong_model_dict.keys()):
                continue
            elif param == "Tn" and not "Tn" in list(xiong_model_dict.keys()):
                continue
            param_list = xiong_model_dict[param]
            if "fit" in param_list:
                params.append(param_list[0])
                if len(param_list) == 3:
                    bounds.append(param_list[2])
                else:
                    bounds.append(default_bounds[param])
                param_list.append(i)
                i += 1
            elif "fix" in param_list:
                pass
            else:
                raise ValueError(
                    f"Xiong model specified, but not setup correctly for parameter [{param}]."
                )
    if "bcm" in list(models.keys()):
        bcm_dict = models["bcm"]
        for param in ["beta_1", "beta_2", "tau", "gamma", "T_melt"]:
            if param not in list(bcm_dict.keys()):
                raise ValueError(
                    f"Bent-cable model specified without parameter [{param}]."
                )
            param_list = bcm_dict[param]
            if "fit" in param_list:
                params.append(param_list[0])
                if len(param_list) == 3:
                    bounds.append(param_list[2])
                else:
                    bounds.append(default_bounds[param])
                if param == "tau":
                    bounds[-1] = (0, bcm_dict["T_melt"][0] - 300)

                param_list.append(i)
                i += 1
            elif "fix" in param_list:
                pass
            else:
                raise ValueError(
                    f"Bent-cable model specified, but no setup correctly for parameter [{param}]."
                )

    if "two-state" in list(models.keys()):
        # shape of two-state param list is [[coef_list], [term_list], [bound_list], fit/fix]
        param_list = models["two-state"]["dE"]
        if "fit" in param_list:
            if isinstance(param_list[0], (float, int)):
                # Only a constant value supplied. Not sure if this will ever really happen
                params.append(param_list[0])
                if len(param_list) == 4:
                    bounds.append(param_list[3])
                else:
                    bounds.append(default_bounds["dE0"])
                param_list.append(i)
                i += 1
            else:  # Need to fit multiple parameters in a list because dE=f(T)
                param_sublist = []
                for j in range(len(param_list[0])):
                    params.append(param_list[0][j])
                    if len(param_list) == 4:
                        bounds.append(param_list[3][j])
                    else:
                        bounds.append(default_bounds[f"dE{j}"])
                    param_sublist.append(i)
                    i += 1
                param_list.append(param_sublist)
        elif "fix" in param_list:
            pass
        else:
            raise ValueError(f"Two-state model specified but not setup correctly.")

    if "linear" in list(models.keys()):
        linear_dict = models["linear"]
        for param in ["alpha", "T_melt"]:
            if param not in list(linear_dict.keys()):
                raise ValueError(f"Linear model specified without parameter [{param}]")
            param_list = linear_dict[param]
            if "fit" in param_list:
                params.append(param_list[0])
                if len(param_list) == 3:
                    bounds.append(param_list[2])
                else:
                    bounds.append(default_bounds[param])
                param_list.append(i)
                i += 1
            elif "fix" in param_list:
                pass
            else:
                raise ValueError(f"Linear model specified but not setup correctly.")
    if "melt" in list(models.keys()):
        melt_dict = models["melt"]
        for param in ["T_melt", "a", "b", "c"]:
            if param not in list(melt_dict.keys()):
                raise ValueError(f"Melt model specified without parameter [{param}].")
            param_list = melt_dict[param]
            if "fit" in param_list:
                params.append(param_list[0])
                if len(param_list) == 3:
                    bounds.append(param_list[2])
                else:
                    bounds.append(default_bounds[param])
                param_list.append(i)
                i += 1
            elif "fix" in param_list:
                pass
            else:
                raise ValueError(f"Melt model specified but not setup correctly.")
    for symbolic_key in [key for key in list(models.keys()) if "symbolic" in key]:
        symbolic_dict = models[symbolic_key]
        Cp_expr = symbolic_dict["expression"]
        if "temp_bounds" not in list(symbolic_dict.keys()):
            symbolic_dict["temp_bounds"] = (0, np.inf)
        symbols = [symbol for symbol in Cp_expr.free_symbols if symbol not in (P, T)]
        for symbol in symbols:
            # Assume fitting all symbols not P and T
            symbolic_dict[str(symbol)] = [10.0, "fit", i]
            params.append(10.0)
            if "param_bounds" in list(symbolic_dict.keys()):
                if str(symbol) in list(symbolic_dict["param_bounds"].keys()):
                    bounds.append(symbolic_dict["param_bounds"][str(symbol)])
            else:
                bounds.append((-np.inf, np.inf))
            i += 1

    arg_dict["models"] = models
    # optimize the model parameters
    # initialize with values for iron, not sure how sensitive it will be
    bounds = tuple(bounds)

    if len(params) > 0:
        min_fits = minimize(
            _fit_heat_capacity,
            x0=params,
            args=arg_dict,
            bounds=bounds,
            method="nelder-mead",
            options={"adaptive": False, "maxiter": len(params) * 200},
        )
    else:
        min_fits = _fit_heat_capacity(params, arg_dict)
    # update the model dict to access the parameters
    for model, mdict in models.items():
        for param, param_list in mdict.items():
            if param == "expression":
                mdict[param] = str(mdict[param])
                continue
            if "fit" in param_list:
                # update the provided parameter with the fitted parameter
                if isinstance(param_list[-1], int):
                    param_list[0] = min_fits.x[param_list[-1]]
                else:
                    for j in range(len(param_list[0])):
                        param_list[0][j] = min_fits.x[param_list[-1][j]]
        if "symbolic" in model:
            symbolic_values = [
                value[0]
                for key, value in mdict.items()
                if key not in ["expression", "param_bounds", "temp_bounds"]
            ]
            models[model]["variable_values"] = symbolic_values
    if verbose:
        print_heat_capacity_fits(min_fits, models)
    return min_fits, models


def print_heat_capacity_fits(min_fits, model_dict):
    # Function to print the fitted parameters.
    print("Optimized model parameters:")
    for model, values in model_dict.items():
        print(f"Model: {model}")
        if model == "two-state":
            out_str = "dE: "
            for i in range(len(values["dE"][0])):
                out_str += f"{values['dE'][0][i] * values['dE'][1][i]} "
            print(out_str)
        else:
            for param_name, param_value in values.items():
                if param_name == "expression":
                    print(f"{param_name}: {param_value}")
                elif param_name == "param_bounds":
                    continue
                else:
                    print(f"{param_name}: {param_value[0]:4e}")
    print(f"RSE: {min_fits.fun:.4f} J/mol/K")
