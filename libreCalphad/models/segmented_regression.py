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
import json
from libreCalphad.databases.db_utils import load_database
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pycalphad import Workspace, as_property, calculate, variables as v
from scipy.integrate import quad
from scipy.optimize import curve_fit, minimize
import symengine as se
from tinydb import where
import warnings
import yaml

R = 8.314472  # J/mol/K
N = 6.02214076e23
T = se.symbols("T")
theta_Fe = 309  # Einstein temperature Chen & Sundman
beta_Fe = 2.22
struct_fact_bcc = 0.37
struct_fact_fcc = 0.28
Tc_Fe = 1043
# Holzapfel parameters
w0 = 5.657e-2
w1 = 3.911e-2
f1 = 4.155e-1
f2 = 8.116e-1
w2 = 9.0432e-1
aD = 5.133e-2


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


def _twostate_Cp(T_arr, dE, coef_list):
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


def _holzapfel_debye_Cp(T_arr, thetaD):
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


def _melt_Cp(T_arr, T_melt, a, b, c):
    # following the approach of Chen and Sundman (2001)
    # TODO: Try extending the linear/bcm and using this contribution to correct the total Cp
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


def _linear_Cp(T_arr, alpha, T_melt):
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


def _bent_cable_Cp(T_arr, beta_1, beta_2, tau, gamma, T_melt):
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

        return ((temp - tau + gamma) ** 2 / (4 * gamma)) * _indfunc1(T, tau, gamma) + (
            temp - tau
        ) * _indfunc2(temp, tau, gamma)

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


def _einstein_Cp(temp, theta):
    if isinstance(temp, list):
        temp = np.array(temp)
    res = (
        3
        * R
        * (theta / temp) ** 2
        * np.exp(theta / temp)
        / (np.exp(theta / temp) - 1) ** 2
    )
    res = np.nan_to_num(res)  # replace nan with 0
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


def _debye_Cp(T_arr, theta):
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


def _xiong_Cp(T_arr, beta, p, Tc):
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


def _calc_RSE(model_array, Cp_array):
    num_SR_params = 5  # from the paper
    return np.sqrt(np.sum(np.square(np.nan_to_num((model_array - Cp_array)))))


def _fit_segmented_regression(x, arg_dict):
    """
    Function to perform the segmented regression to optimize all modeling parameters.
    """

    temperature_array = arg_dict["temperature_array"]
    Cp_array = arg_dict["Cp_array"]
    models = arg_dict["models"]
    melt_base_Cp = None
    if "melt" in list(models.keys()):
        # Need to pad the Cp array to include some liquid temperatures and target Cp
        melt_dict = models["melt"]
        temperature_array = np.append(temperature_array, melt_dict["T_melt"][0] + 500)
        if "liquid_Cp" in list(melt_dict.keys()):
            liq_Cp = melt_dict["liquid_Cp"][0]
        else:
            liq_Cp = Cp_array[-1]
        Cp_array = np.append(Cp_array, liq_Cp)
        while len(temperature_array[temperature_array > melt_dict["T_melt"][0]]) < 5:
            # if np.max(temperature_array) < melt_dict["T_melt"][0]:
            #     temperature_array = np.append(
            #         temperature_array, melt_dict["T_melt"][0] + 500
            #     )
            # else:
            temperature_array = np.append(
                temperature_array, np.max(temperature_array) + 100
            )
            Cp_array = np.append(Cp_array, liq_Cp)

    model_Cp = np.zeros(len(Cp_array))
    if "einstein" in list(models.keys()):
        model_params = []
        einstein_dict = models["einstein"]
        theta_list = einstein_dict["theta"]
        if "fit" in theta_list:
            idx = theta_list[-1]
            model_Cp += _einstein_Cp(temperature_array, x[idx])
        else:
            model_Cp += _einstein_Cp(temperature_array, theta_list[0])
    elif "holzapfel" in list(models.keys()):
        model_params = []
        holzapfel_dict = models["holzapfel"]
        theta_list = holzapfel_dict["theta"]
        if "fit" in theta_list:
            idx = theta_list[-1]
            model_Cp += _holzapfel_debye_Cp(temperature_array, x[idx])
        else:
            model_Cp += _holzapfel_debye_Cp(temperature_array, theta_list[0])
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
        model_Cp += _xiong_Cp(temperature_array, beta, structure_factor, critical_temp)
    if "bcm" in list(models.keys()):
        bcm_dict = models["bcm"]
        model_params = []
        for param in ["beta_1", "beta_2", "tau", "gamma", "T_melt"]:
            param_list = bcm_dict[param]
            if "fit" in param_list:
                idx = param_list[-1]
                model_params.append(x[idx])
            else:
                model_params.append(param_list[0])
        model_Cp += _bent_cable_Cp(temperature_array, *model_params)
    if "two-state" in list(models.keys()):
        model_params = []
        param_list = models["two-state"]["dE"]
        if "fit" in param_list:
            if isinstance(param_list[0], (float, int)):
                idx = param_list[-1]
                model_params.append(x[idx])
            else:
                idx_list = param_list[-1]
                for j in idx_list:
                    model_params.append(x[j])
        else:
            if isinstance(param_list[0], (float, int)):
                model_params.append(param_list[0])
            else:
                for param in param_list[0]:
                    model_params.append(param)
        model_Cp += _twostate_Cp(
            temperature_array, model_params, models["two-state"]["dE"][1]
        )
    if "linear" in list(models.keys()):
        linear_dict = models["linear"]
        model_params = []
        for param in ["alpha", "T_melt"]:
            param_list = linear_dict[param]
            if "fit" in param_list:
                idx = param_list[-1]
                model_params.append(x[idx])
            else:
                model_params.append(param_list[0])
        model_Cp += _linear_Cp(temperature_array, *model_params)

    if "melt" in list(models.keys()):
        # Now add the high-temp heat capacity model
        melt_dict = models["melt"]
        model_params = []
        for param in ["T_melt", "a", "b", "c"]:
            param_list = melt_dict[param]
            if "fit" in param_list:
                idx = param_list[-1]
                model_params.append(x[idx])
            else:
                model_params.append(param_list[0])
        if "liquid_Cp" not in list(melt_dict.keys()):
            melt_base_Cp = np.max(model_Cp)
        else:
            melt_base_Cp = melt_dict["liquid_Cp"][0]
        assert melt_base_Cp is not None, (
            "Need to specify a base melt heat capacity in another method."
        )
        model_Cp += _melt_Cp(temperature_array, *model_params)
    return _calc_RSE(model_Cp, Cp_array)


def fit_segmented_regression(
    datasets,
    models,
):
    # TODO: Use the pycalphad DB for loading model properties.
    implemented_models = [
        "bcm",
        "einstein",
        "holzapfel",
        "linear",
        "melt",
        "offset",
        "two-state",
        "xiong",
    ]
    for model in list(models.keys()):
        if model not in implemented_models:
            raise NotImplementedError(
                f"{model} not implemented. Options are {implemented_models}"
            )

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
    df = df.sort_values("temperature")

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
                    bounds[-1] = (1e-5, bcm_dict["T_melt"][0] - 300)

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

    arg_dict["models"] = models
    # optimize the model parameters
    # initialize with values for iron, not sure how sensitive it will be
    bounds = tuple(bounds)

    if len(params) > 0:
        min_fits = minimize(
            _fit_segmented_regression,
            x0=params,
            args=arg_dict,
            bounds=bounds,
            method="nelder-mead",
        )
    else:
        min_fits = _fit_segmented_regression(params, arg_dict)
    # update the model dict to access the parameters
    for model, mdict in models.items():
        for param, param_list in mdict.items():
            if "fit" in param_list:
                # update the provided parameter with the fitted parameter
                if isinstance(param_list[-1], int):
                    param_list[0] = min_fits.x[param_list[-1]]
                else:
                    for j in range(len(param_list[0])):
                        param_list[0][j] = min_fits.x[param_list[-1][j]]
    return min_fits, models


def get_segmented_regression_Cp(T_arr, Cp_fits, use_einstein=False, xiong_params=None):
    if xiong_params is not None:
        for param in ["Tc", "p", "beta"]:
            assert param in list(xiong_params.keys()), (
                f"Xiong params dict missing {param}."
            )
    if isinstance(T_arr, (int, float)):
        Cp = 0
    else:
        Cp = np.zeros(len(T_arr))
    if use_einstein:
        Cp += _einstein_Cp(T_arr, Cp_fits.x[0])
    else:
        Cp += _holzapfel_debye_Cp(T_arr, Cp_fits.x[0])
    if xiong_params is not None:
        Cp += _xiong_Cp(
            T_arr, xiong_params["beta"], xiong_params["p"], xiong_params["Tc"]
        )
    Cp += _bent_cable_Cp(T_arr, *Cp_fits.x[1:])
    return Cp


def calc_enthalpy(T_arr, Cp_fits, xiong_params=None, use_einstein=False):
    if isinstance(T_arr, (int, float)):
        enthalpy = 0
    else:
        enthalpy = np.zeros(len(T_arr))
    if use_einstein:
        raise NotImplementedError("Einstein enthalpy calculation not yet implemented.")

    enthalpy += _holzapfel_enthalpy(T_arr, *Cp_fits.x[:1])
    if xiong_params is not None:
        enthalpy += _xiong_enthalpy(T_arr, *(list(xiong_params.values())))
    enthalpy += _bent_cable_enthalpy(T_arr, *Cp_fits.x[1:])
    return enthalpy


def calc_entropy(T_arr, Cp_fits, xiong_params=None):
    if isinstance(T_arr, (int, float)):
        entropy = 0
    else:
        entropy = np.zeros(len(T_arr))
    entropy += _holzapfel_entropy(T_arr, *Cp_fits.x[:1])
    if xiong_params is not None:
        entropy += _xiong_entropy(T_arr, *list(xiong_params.values()))
    entropy += _bent_cable_entropy(T_arr, *Cp_fits.x[1:])
    return entropy


def calc_gibbs_energy(T_arr, Cp_fits, use_einstein=False, xiong_params=None):
    if use_einstein:
        raise NotImplementedError("Einstein Gibbs energy not yet implemented.")

    if isinstance(T_arr, (int, float)):
        ret_arr = calc_enthalpy(T_arr, Cp_fits, xiong_params) - T_arr * calc_entropy(
            T_arr, Cp_fits, xiong_params
        )
    else:
        ret_arr = np.array([])
        for temp in T_arr:
            gibbs = calc_enthalpy(temp, Cp_fits, xiong_params) - temp * calc_entropy(
                temp, Cp_fits, xiong_params
            )
            ret_arr = np.append(ret_arr, gibbs)
    if xiong_params is not None:
        ret_arr += _xiong_gibbs(T_arr, *list(xiong_params.values()))
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
