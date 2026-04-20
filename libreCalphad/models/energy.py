"""
Functions to calculate enthalpy, entropy, and Gibbs energies derived from models for the heat capacity.
"""

from collections import OrderedDict
import json
from libreCalphad.databases.db_utils import load_database
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pycalphad import calculate, variables as v
from scipy.integrate import quad
from scipy.optimize import curve_fit, minimize
import symengine as se
from tinydb import where
import warnings
import yaml

R = 8.314472  # J/mol/K
N = 6.02214076e23
T = se.symbols("T")


def calculate_offset(xdata, ydata):
    def _find_offset(xdata, offset):
        return offset + xdata

    fits = curve_fit(_find_offset, xdata=xdata, ydata=ydata)
    return fits


def calculate_transition_energies(
    dbf, components, phase, transition_temperature, output
):
    """
    Function to calculate the energies at a phase transition where the Cp fitting method
    needs to change. Gibbs energy expressions may need to include enthalpy and entropy
    terms carried over from the lower-temperature phase to the higher-temperature phase
    for continuity.
    """

    def _min_energy_difference(offset, args):
        dbf = args["dbf"]
        components = args["components"]
        phase = args["phase"]
        temp = args["transition_temperature"]
        output = args["output"]
        phase_1_calc = calculate(
            dbf, components, phase, T=temp - 0.001, P=101325, N=1, output=output
        )
        phase_2_calc = calculate(
            dbf, components, phase, T=temp + 0.001, P=101325, N=1, output=output
        )
        energy_1 = getattr(phase_1_calc, output).squeeze().values
        energy_2 = getattr(phase_2_calc, output).squeeze().values
        return np.abs(energy_1 - (energy_2 + offset))

    min_args = {
        "dbf": dbf,
        "components": components,
        "phase": phase,
        "transition_temperature": transition_temperature,
        "output": output,
    }
    min_fits = minimize(_min_energy_difference, x0=100, args=min_args)
    return min_fits


def _offset_gibbs(T_arr, offset_enthalpy, offset_entropy, ret_expr=False):
    # This handles the constants lost during integration
    ret_arr = 0
    if isinstance(T_arr, (int, float)) and not ret_expr:
        ret_arr = offset_enthalpy - offset_entropy * T_arr
    elif not ret_expr:
        ret_arr = np.array([])
        for temp in T_arr:
            gibbs = offset_enthalpy - offset_entropy * temp
            ret_arr = np.append(ret_arr, gibbs)
    else:
        ret_arr = offset_enthalpy - offset_entropy * v.T

    return ret_arr


def _twostate_gibbs(T_arr, dE, coef_list, ret_expr=False):
    if isinstance(dE, list):
        dE = np.sum([dE[i] * coef_list[i] for i in range(len(dE))])

    def _calc_twostate_gibbs(temp, dE):
        return np.float64((-R * T * se.log(1 + se.exp(-dE / (R * T)))).subs(T, temp))

    def _sympy_twostate_gibbs(temp, dE):
        if isinstance(dE, list):
            dE = np.sum([dE[i] * v.T**i for i in range(len(dE))])
        return -v.R * v.T * se.log(1 + se.exp(-dE / (v.R * v.T)))

    if isinstance(T_arr, (int, float)):
        ret_arr = _calc_twostate_gibbs(T_arr, dE)
    elif ret_expr:
        ret_arr = _sympy_twostate_gibbs(None, dE)
    else:
        ret_arr = np.array([])
        for temp in T_arr:
            ret_arr = np.append(ret_arr, _calc_twostate_gibbs(temp, dE))
    return ret_arr


def _twostate_entropy(T_arr, dE, coef_list, ret_expr=False):
    # From Miodownik1970
    g0 = 1
    g1 = 1

    def _calc_twostate_entropy(T, dE):
        if isinstance(dE, list):
            dE = np.sum([dE[i] * coef_list[i] for i in range(len(dE))])
        alpha = g1 / g0 * np.exp(-dE / (R * T))
        return R * (np.log(1 + alpha) - alpha / (1 + alpha) * dE / (R * T))

    def _sympy_twostate_entropy(T, dE):
        if isinstance(dE, list):
            dE = np.sum([dE[i] * v.T**i for i in range(len(dE))])
        alpha = g1 / g0 * se.exp(-dE / (v.R * v.T))
        return v.R * (se.log(1 + alpha) - alpha / (1 + alpha) * dE / (v.R * v.T))

    if isinstance(T_arr, (int, float)):
        ret_arr = _calc_twostate_entropy(T_arr, dE)
    elif ret_expr:
        ret_arr = _sympy_twostate_entropy(None, dE)
    else:
        ret_arr = np.array([])
        for temp in T_arr:
            ret_arr = np.append(ret_arr, _calc_twostate_entropy(temp, dE))
    return ret_arr


def _twostate_enthalpy(T_arr, dE, coef_list, ret_expr=False):
    if ret_expr:
        ret_arr = v.T * _twostate_entropy(T_arr, dE, ret_expr)
    else:
        if isinstance(T_arr, (int, float)):
            ret_arr = T_arr * _twostate_entropy(T_arr, dE, ret_expr)
        else:
            ret_arr = np.array([])
            for temp in T_arr:
                ret_arr = np.append(
                    ret_arr, temp * _calc_twostate_entropy(temp, dE, ret_expr)
                )
    return ret_arr


def _holzapfel_enthalpy(T_arr, thetaD, ret_expr=False):
    def _calc_holzapfel_enthalpy(x, thetaD):
        term0 = 3 / 8
        term1 = x / aD * x**3 / (1 + x**3 / (aD * w0))
        term2 = w1 * f1 * np.exp(-f2 / x) / (1 - np.exp(-f1 / x))
        term3 = w2 * f2 * np.exp(-f2 / x) / (1 - np.exp(-f1 / x))
        return thetaD * np.sum([term0, term1, term2, term3])

    def _sympy_holzapfel_enthalpy(x, thetaD):
        term0 = 3 / 8
        term1 = x / aD * x**3 / (1 + x**3 / (aD * w0))
        term2 = w1 * f1 * se.exp(-f2 / x) / (1 - se.exp(-f1 / x))
        term3 = w2 * f2 * se.exp(-f2 / x) / (1 - se.exp(-f1 / x))
        return thetaD * np.sum([term0, term1, term2, term3])

    if T_arr is not None:
        if isinstance(T_arr, (int, float)):
            x = T_arr / thetaD
            ret_arr = _calc_holzapfel_enthalpy(x, thetaD)
        else:
            ret_arr = np.array([])
            for temp in T_arr:
                x = temp / thetaD
                H_holzapfel = _calc_holzapfel_enthalpy(x, thetaD)
                ret_arr = np.append(ret_arr, H_holzapfel)
    elif ret_expr:
        x = v.T / thetaD
        ret_arr = _sympy_holzapfel_enthalpy(x, thetaD)
    return ret_arr


def _holzapfel_entropy(T_arr, thetaD, ret_expr=False):
    def _calc_holzapfel_entropy(x, thetaD):
        term1 = w0 / 3 * np.log(1 + x**3 / (aD * w0))
        term2 = 1 / aD * x**3 / (1 + x**3 / (aD * w0))
        term3 = -w1 * np.log(1 - np.exp(-f1 / x))
        term4 = w1 * f1 / x * np.exp(-f1 / x) / (1 - np.exp(-f1 / x))
        term5 = -w2 * np.log(1 - np.exp(-f2 / x))
        term6 = w2 * f2 / x * np.exp(-f2 / x) / (1 - np.exp(-f2 / x))
        return np.sum([term1, term2, term3, term4, term5, term6])

    def _sympy_holzapfel_entropy(x, thetaD):
        term1 = w0 / 3 * se.log(1 + x**3 / (aD * w0))
        term2 = 1 / aD * x**3 / (1 + x**3 / (aD * w0))
        term3 = -w1 * se.log(1 - se.exp(-f1 / x))
        term4 = w1 * f1 / x * se.exp(-f1 / x) / (1 - se.exp(-f1 / x))
        term5 = -w2 * se.log(1 - se.exp(-f2 / x))
        term6 = w2 * f2 / x * se.exp(-f2 / x) / (1 - se.exp(-f2 / x))
        return np.sum([term1, term2, term3, term4, term5, term6])

    if T_arr is not None:
        if isinstance(T_arr, (int, float)):
            x = T_arr / thetaD
            ret_arr = _calc_holzapfel_entropy(x, thetaD)
        else:
            ret_arr = np.array([])
            for temp in T_arr:
                x = temp / thetaD
                S_holzapfel = _calc_holzapfel_entropy(x, thetaD)
                ret_arr = np.append(ret_arr, S_holzapfel)
    elif ret_expr:
        x = v.T / thetaD
        ret_arr = _sympy_holzapfel_entropy(x, thetaD)
    return ret_arr


def _holzapfel_gibbs(T_arr, thetaD, ret_expr=False):
    if not ret_expr:
        return (
            3
            * R
            * (
                _holzapfel_enthalpy(T_arr, thetaD, ret_expr)
                - T_arr * _holzapfel_entropy(T_arr, thetaD, ret_expr)
            )
        )
    else:
        return (
            3
            * v.R
            * (
                _holzapfel_enthalpy(T_arr, thetaD, ret_expr)
                - v.T * _holzapfel_entropy(T_arr, thetaD, ret_expr)
            )
        )


def _melt_gibbs(
    T_arr, T_melt, a, b, c, solid_enthalpy=0, solid_entropy=0, ret_expr=False
):
    if isinstance(T_arr, (int, float)) and not ret_expr:
        if T_arr <= T_melt:
            ret_arr = 0
        else:
            ret_arr = (
                solid_enthalpy
                - solid_entropy * T_arr
                + a * T_arr * (1 - np.log(T_arr))
                - b / 30 * T_arr**-5
                - c / 132 * T_arr**-11
            )
    elif not ret_expr:
        ret_arr = np.array([])
        for temp in T_arr:
            if temp <= T_melt:
                gibbs = 0
            else:
                gibbs = (
                    solid_enthalpy
                    - solid_entropy * temp
                    + a * temp * (1 - np.log(temp))
                    - b / 30 * temp**-5
                    - c / 132 * temp**-11
                )
            ret_arr = np.append(ret_arr, gibbs)
    else:
        if T_arr <= T_melt:
            ret_arr = 0
        else:
            ret_arr = (
                solid_enthalpy
                - solid_entropy * v.T
                + a * v.T * (1 - se.log(v.T))
                - b / 30 * v.T**-5
                - c / 132 * v.T**-11
            )
    return ret_arr


def _linear_enthalpy(T_arr, alpha, T_melt, ret_expr):
    ah = alpha / 2
    if isinstance(T_arr, (int, float)) and not ret_expr:
        if T_arr <= T_melt:
            ret_arr = ah * T_arr**2
        else:
            ret_arr = 0
    elif not ret_expr:
        ret_arr = np.array([])
        for temp in T_arr:
            if temp <= T_melt:
                enthalpy = aH * temp**2
            else:
                enthalpy = 0
            ret_arr = np.append(ret_arr, enthalpy)
    else:  # want symengine expression
        if T_arr <= T_melt:
            return ah * v.T**2
        else:
            return 0

    return ret_arr


def _linear_entropy(T_arr, alpha, T_melt, ret_expr):
    if isinstance(T_arr, (int, float)) and not ret_expr:
        if T_arr <= T_melt:
            ret_arr = alpha * T_arr
        else:
            ret_arr = 0
    elif not ret_expr:
        ret_arr = np.array([])
        for temp in T_arr:
            if temp <= T_melt:
                entropy = alpha * temp
            else:
                entropy = 0
            ret_arr = np.append(ret_arr, entropy)
    else:  # symengine expression
        if T_arr <= T_melt:
            return alpha * v.T
        else:
            return 0
    return ret_arr


def _linear_gibbs(T_arr, alpha, T_melt, ret_expr=False):
    if isinstance(T_arr, (int, float)) and not ret_expr:
        ret_arr = _linear_enthalpy(
            T_arr, alpha, T_melt, ret_expr
        ) - T_arr * _linear_entropy(T_arr, alpha, T_melt, ret_expr)
    elif not ret_expr:
        ret_arr = np.array([])
        for temp in T_arr:
            gibbs = _linear_enthalpy(
                temp, alpha, T_melt, ret_expr
            ) - temp * _linear_entropy(temp, alpha, T_melt, ret_expr)
            ret_arr = np.append(ret_arr, gibbs)
    else:  # want symengine expression
        return _linear_enthalpy(T_arr, alpha, T_melt, ret_expr) - v.T * _linear_entropy(
            T_arr, alpha, T_melt, ret_expr
        )
    return ret_arr


def _bent_cable_enthalpy(T_arr, beta_1, beta_2, tau, gamma, T_melt, ret_expr=False):
    a2h = -beta_2 / (12 * gamma) * (tau - gamma) ** 3
    a3h = beta_2 / 2 * (gamma**2 / 3 + tau**2)
    c1h = beta_1 / 2
    c2h = beta_1 / 2 - beta_2 / (4 * gamma) * (tau - gamma)
    c3h = (beta_1 + beta_2) / 2
    b2h = beta_2 / (4 * gamma) * (tau - gamma) ** 2
    b3h = -beta_2 * tau
    d2h = beta_2 / (12 * gamma)

    def _calc_bcm_enthalpy(temp, c1h, a2h, b2h, c2h, d2h, a3h, b3h, c3h):
        if temp < tau - gamma:
            return c1h * temp**2
        elif all([temp >= tau - gamma, temp <= tau + gamma]):
            return a2h + b2h * temp + c2h * temp**2 + d2h * temp**3
        else:
            return a3h + b3h * temp + c3h * temp**2

    def _sympy_bcm_enthalpy(temp, c1h, a2h, b2h, c2h, d2h, a3h, b3h, c3h):
        if temp < tau - gamma:
            return c1h * v.T**2
        elif all([temp >= tau - gamma, temp <= tau + gamma]):
            return a2h + b2h * v.T + c2h * v.T**2 + d2h * v.T**3
        else:
            return a3h + b3h * v.T + c3h * v.T**2

    if isinstance(T_arr, (int, float)) and not ret_expr:
        ret_arr = _calc_bcm_enthalpy(T_arr, c1h, a2h, b2h, c2h, d2h, a3h, b3h, c3h)
    elif not ret_expr:
        ret_arr = np.array([])
        for temp in T_arr:
            H_bcm = _calc_bcm_enthalpy(temp, c1h, a2h, b2h, c2h, d2h, a3h, b3h, c3h)
            ret_arr = np.append(ret_arr, H_bcm)
    elif isinstance(T_arr, (int, float)) and ret_expr:
        ret_arr = _sympy_bcm_enthalpy(T_arr, c1h, a2h, b2h, c2h, d2h, a3h, b3h, c3h)
    return ret_arr


def _bent_cable_entropy(T_arr, beta_1, beta_2, tau, gamma, T_melt, ret_expr=False):
    c1s = beta_1
    b2s = (tau - gamma) ** 2 * (
        3 / 8 * beta_2 / gamma - beta_2 / (4 * gamma) * np.log(tau - gamma)
    )
    b3s = -3 * beta_2 * tau / 2 - beta_2 / (4 * gamma) * (
        (tau - gamma) ** 2 * np.log(tau - gamma)
        - (tau + gamma) ** 2 * np.log(tau + gamma)
    )
    c2s = beta_1 - (beta_2 * (tau - gamma)) / (2 * gamma)
    c3s = beta_1 + beta_2
    d2s = beta_2 / (8 * gamma)
    e2s = beta_2 / (4 * gamma) * (tau - gamma) ** 2
    e3s = -beta_2 * tau

    def _calc_bcm_entropy(temp, c1s, b2s, b3s, c2s, c3s, d2s, e2s, e3s, T_melt):
        if temp < tau - gamma:
            res = c1s * temp**2
        elif all([temp >= tau - gamma, temp <= tau + gamma]):
            res = b2s * temp + c2s * temp**2 + d2s * temp**3 + e2s * temp * np.log(temp)
        elif all([temp > tau + gamma, temp < T_melt]):
            res = b3s * temp + c3s * temp**2 + e3s * temp * np.log(temp)
        else:
            res = 0
        return res / temp

    def _sympy_bcm_entropy(temp, c1s, b2s, b3s, c2s, c3s, d2s, e2s, e3s, T_melt):
        b2s = (tau - gamma) ** 2 * (
            3 / 8 * beta_2 / gamma - beta_2 / (4 * gamma) * se.log(tau - gamma)
        )
        b3s = -3 * beta_2 * tau / 2 - beta_2 / (4 * gamma) * (
            (tau - gamma) ** 2 * se.log(tau - gamma)
            - (tau + gamma) ** 2 * se.log(tau + gamma)
        )
        if temp < tau - gamma:
            res = c1s * v.T**2
        elif all([temp >= tau - gamma, temp <= tau + gamma]):
            res = b2s * v.T + c2s * v.T**2 + d2s * v.T**3 + e2s * v.T * se.log(v.T)
        elif all([temp > tau + gamma, temp < T_melt]):
            res = b3s * v.T + c3s * v.T**2 + e3s * v.T * se.log(v.T)
        else:
            res = 0
        return res / v.T

    if isinstance(T_arr, (int, float)) and not ret_expr:
        ret_arr = _calc_bcm_entropy(
            T_arr, c1s, b2s, b3s, c2s, c3s, d2s, e2s, e3s, T_melt
        )
    elif not ret_expr:
        ret_arr = np.array([])
        for temp in T_arr:
            S_bcm = _calc_bcm_entropy(
                temp, c1s, b2s, b3s, c2s, c3s, d2s, e2s, e3s, T_melt
            )
            ret_arr = np.append(ret_arr, S_bcm)
    elif isinstance(T_arr, (int, float)) and ret_expr:
        ret_arr = _sympy_bcm_entropy(
            T_arr, c1s, b2s, b3s, c2s, c3s, d2s, e2s, e3s, T_melt
        )
    return ret_arr


def _bent_cable_gibbs(T_arr, beta_1, beta_2, tau, gamma, T_melt, ret_expr=False):
    if not ret_expr:
        res = _bent_cable_enthalpy(
            T_arr, beta_1, beta_2, tau, gamma, T_melt, ret_expr
        ) - T_arr * _bent_cable_entropy(
            T_arr, beta_1, beta_2, tau, gamma, T_melt, ret_expr
        )
    else:
        res = _bent_cable_enthalpy(
            T_arr, beta_1, beta_2, tau, gamma, T_melt, ret_expr
        ) - v.T * _bent_cable_entropy(
            T_arr, beta_1, beta_2, tau, gamma, T_melt, ret_expr
        )
    return res


def _einstein_entropy(temp, theta):
    return (
        3
        * R
        * (
            theta
            / (2 * temp)
            * np.cosh(theta / (2 * temp))
            / np.sinh(theta / (2 * temp))
            - np.log(2 * np.sinh(theta / 2 * temp))
        )
    )


def _einstein_gibbs(temp, theta):
    # Chen and Sundman
    return 3 / 2 * R * theta + 3 * R * temp * np.log(1 - np.exp(-theta / temp))


def _xiong_enthalpy(T_arr, beta, p, Tc):

    return _xiong_gibbs(T_arr, beta, p, Tc) + T_arr * _xiong_entropy(T_arr, beta, p, Tc)


def _xiong_entropy(T_arr, beta, p, Tc):
    if isinstance(T_arr, (int, float)):
        return R * np.log(beta + 1)
    else:
        ret_arr = np.array([])
        for temp in T_arr:
            S_xiong = R * np.log(beta + 1)
            ret_arr = np.append(ret_arr, S_xiong)
        return ret_arr


def _xiong_gibbs(T_arr, beta, p, Tc, ret_expr=False):
    # From Hao2024
    def _D(p):
        return 0.33471979 + 0.49649686 * (1 / p - 1)

    def _g_int(temp, p, Tc):
        tau = temp / Tc
        if temp / Tc <= 1:
            coef = 0.63570895 * (1 / p - 1)
            term1 = tau**3 / 6
            term2 = tau**9 / 135
            term3 = tau**15 / 600
            term4 = tau**21 / 1617
        else:
            coef = 1
            term1 = tau**-7 / 21
            term2 = tau**-21 / 630
            term3 = tau**-35 / 2975
            term4 = tau**-49 / 8232
        return coef * np.sum([term1, term2, term3, term4])

    def _sympy_g_int(temp, p, Tc):
        tau = v.T / Tc
        if temp / Tc <= 1:
            coef = 0.63570895 * (1 / p - 1)
            term1 = tau**3 / 6
            term2 = tau**9 / 135
            term3 = tau**15 / 600
            term4 = tau**21 / 1617
        else:
            coef = 1
            term1 = tau**-7 / 21
            term2 = tau**-21 / 630
            term3 = tau**-35 / 2975
            term4 = tau**-49 / 8232
        return coef * np.sum([term1, term2, term3, term4])

    def _gmdo(temp, beta, p, Tc):
        return -R * np.log(beta + 1) * (temp - 0.38438376 * Tc / (p * _D(p)))

    def _sympy_gmdo(temp, beta, p, Tc):
        return -R * se.log(beta + 1) * (v.T - 0.38438376 * Tc / (p * _D(p)))

    if isinstance(T_arr, (int, float)) and not ret_expr:
        ret_arr = -R * T_arr / _D(p) * np.log(beta + 1) * _g_int(T_arr, p, Tc)
        +_gmdo(T_arr, beta, p, Tc)
    elif not ret_expr:
        ret_arr = np.array([])
        for temp in T_arr:
            G_mag = -R * temp / _D(p) * np.log(beta + 1) * _g_int(temp, p, Tc)
            +_gmdo(temp, beta, p, Tc)
            ret_arr = np.append(ret_arr, G_mag)
    elif isinstance(T_arr, (int, float)) and ret_expr:
        ret_arr = -v.R * v.T / _D(p) * se.log(beta + 1) * _sympy_g_int(
            T_arr, p, Tc
        ) + _sympy_gmdo(T_arr, beta, p, Tc)
    return ret_arr


def create_espei_custom_refstate_stable(model_dict):
    """
    This function generates the endmember lattice stabilities based on the heat capacity fitting data.
    Pycalphad currently implements the Einstein model, so using the Holzapfel approximation requires
    incorporating it into the pycalphad base model. The Xiong model is also implemented in pycalphad
    and does not need to be explicitly included in the expression.

    This custom reference state therefore is the Gibbs energy expression from the bent-cable model.
    Currently, for the best accuracy you should fit it using the Einstein model instead of Holzapfel.

    TODO: Incorporate Holzapfel approximation into pycalphad.
    """
    critical_temperatures = [1e-5]
    if "bcm" in list(model_dict.keys()):
        bcm_dict = model_dict["bcm"]
        bcm_args = []
        bcm_kwargs = {"ret_expr": True}
        bcm_args.append(bcm_dict["beta_1"][0])
        bcm_args.append(bcm_dict["beta_2"][0])
        tau = bcm_dict["tau"][0]
        bcm_args.append(tau)
        gamma = bcm_dict["gamma"][0]
        bcm_args.append(gamma)
        bcm_args.append(bcm_dict["T_melt"][0])
        critical_temperatures.append(tau - gamma)
        critical_temperatures.append(tau + gamma)
    if "linear" in list(model_dict.keys()):
        linear_args = []
        linear_kwargs = {"ret_expr": True}
        linear_args.append(model_dict["linear"]["alpha"][0])
        T_melt = model_dict["linear"]["T_melt"][0]
        linear_args.append(T_melt)
        if T_melt not in critical_temperatures:
            critical_temperatures.append(T_melt)
    if "melt" in list(model_dict.keys()):
        melt_args = []
        melt_kwargs = {"ret_expr": True}
        T_melt = model_dict["melt"]["T_melt"][0]
        melt_args.append(T_melt)
        melt_args.append(model_dict["melt"]["a"][0])
        melt_args.append(model_dict["melt"]["b"][0])
        melt_args.append(model_dict["melt"]["c"][0])

        for param in ["solid_enthalpy", "solid_entropy"]:
            if param in list(model_dict["melt"].keys()):
                melt_kwargs[param] = model_dict["melt"][param][0]
        if T_melt not in critical_temperatures:
            critical_temperatures.append(T_melt)
    if "offset" in list(model_dict.keys()):
        offset_args = []
        offset_kwargs = {"ret_expr": True}
        offset_args.append(model_dict["offset"]["enthalpy"][0])
        offset_args.append(model_dict["offset"]["entropy"][0])
    critical_temperatures.append(10000.00)

    critical_temperatures.sort()

    res = []
    for i in range(1, len(critical_temperatures)):
        # Don't need to include  Xiong parameters because pycalphad includes the Xiong model in its calculations
        this_res = 0
        if "bcm" in list(model_dict.keys()):
            bcm_expr = _bent_cable_gibbs(
                critical_temperatures[i] - 1.0, *bcm_args, **bcm_kwargs
            )
            this_res = this_res + bcm_expr
        if "linear" in list(model_dict.keys()):
            linear_expr = _linear_gibbs(
                critical_temperatures[i] - 1.0, *linear_args, **linear_kwargs
            )
            this_res = this_res + linear_expr
        if "melt" in list(model_dict.keys()):
            melt_expr = _melt_gibbs(
                critical_temperatures[i] - 1.0, *melt_args, **melt_kwargs
            )
            this_res = this_res + melt_expr

        if "offset" in list(model_dict.keys()):
            offset_expr = _offset_gibbs(
                critical_temperatures[i] - 1.0, *offset_args, **offset_kwargs
            )
            this_res = this_res + offset_expr

        if this_res != 0:
            res.append(
                tuple(
                    (
                        this_res,
                        se.And(
                            v.T >= critical_temperatures[i - 1],
                            v.T < critical_temperatures[i],
                        ),
                    ),
                )
            )
    res.append((0, True))

    sympy_expr = se.Piecewise(*res)
    return sympy_expr


def upsert_custom_refstate_json(
    refstate_file, element, Cp_fits, phase=None, model_dict=None
):
    if os.path.exists(refstate_file):
        with open(refstate_file, "r") as f:
            custom_refstate = json.load(f)
    else:
        custom_refstate = {}

    if phase is not None:
        element_key = f"{element}-{phase}"
    else:
        element_key = element

    # Need to update the functions used to create the reference state
    custom_refstate[element_key] = {
        "Cp_fits.x": Cp_fits.x.tolist(),
        "Cp_fits.RSE": float(Cp_fits.fun),
    }
    if model_dict is not None:
        for key, value in model_dict.items():
            if key == "two-state":
                value["dE"][1] = str(value["dE"][1])
            custom_refstate[element_key][key] = value
    with open(refstate_file, "w") as f:
        json.dump(custom_refstate, f, indent=True)
    return True
