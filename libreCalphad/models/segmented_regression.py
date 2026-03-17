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

from espei.datasets import load_datasets, recursive_glob
from libreCalphad.databases.db_utils import load_database
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pycalphad import Workspace, as_property, calculate, variables as v
from scipy.integrate import quad
from scipy.optimize import curve_fit, minimize
from tinydb import where
import yaml

R = 8.314472  # J/mol/K
N = 6.02214076e23
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


def _holzapfel_debye_Cp(T_arr, thetaD):
    def _calc_holzapfel_Cp(x):
        term1 = 4 / aD * x**3 / (1 + x**3 / (aD * w0))
        term2 = -3 / (aD**2 * w0) * x**6 / (1 + x**3 / (aD * w0)) ** 2
        term3 = w1 * (f1 / x) ** 2 * np.exp(-f1 / x) / (1 - np.exp(-f1 / x)) ** 2
        term4 = w2 * (f2 / x) ** 2 * np.exp(-f2 / x) / (1 - np.exp(-f2 / x)) ** 2
        return 3 * R * np.sum([term1, term2, term3, term4])

    if isinstance(T_arr, float):
        x = T_arr / thetaD
        ret_arr = _calc_holzapfel_Cp(x)

    else:
        ret_arr = np.array([])
        for T in T_arr:
            x = T / thetaD
            Cp = _calc_holzapfel_Cp(x)
            ret_arr = np.append(ret_arr, Cp)
    return ret_arr


def _holzapfel_enthalpy(T_arr, thetaD):
    def _calc_holzapfel_enthalpy(x, thetaD):
        term0 = 3 / 8
        term1 = x / aD * x**3 / (1 + x**3 / (aD * w0))
        term2 = w1 * f1 * np.exp(-f2 / x) / (1 - np.exp(-f1 / x))
        term3 = w2 * f2 * np.exp(-f2 / x) / (1 - np.exp(-f1 / x))
        return 3 * R * thetaD * np.sum([term0, term1, term2, term3])

    if isinstance(T_arr, float):
        x = T_arr / thetaD
        ret_arr = _calc_holzapfel_enthalpy(x, thetaD)
    else:
        ret_arr = np.array([])
        for T in T_arr:
            x = T / thetaD
            H_holzapfel = _calc_holzapfel_enthalpy(x, thetaD)
            ret_arr = np.append(ret_arr, H_holzapfel)

    return ret_arr


def _holzapfel_entropy(T_arr, thetaD):
    def _calc_holzapfel_entropy(x, thetaD):
        term1 = w0 / 3 * np.log(1 + x**3 / (aD * w0))
        term2 = 1 / aD * x**3 / (1 + x**3 / (aD * w0))
        term3 = -w1 * np.log(1 - np.exp(-f1 / x))
        term4 = w1 * f1 / x * np.exp(-f1 / x) / (1 - np.exp(-f1 / x))
        term5 = -w2 * np.log(1 - np.exp(-f2 / x))
        term6 = w2 * f2 / x * np.exp(-f2 / x) / (1 - np.exp(-f2 / x))
        return 3 * R * np.sum([term1, term2, term3, term4, term5, term6])

    if isinstance(T_arr, float):
        x = T_arr / thetaD
        ret_arr = _calc_holzapfel_entropy(x, thetaD)
    else:
        ret_arr = np.array([])
        for T in T_arr:
            x = T / thetaD
            S_holzapfel = _calc_holzapfel_entropy(x, thetaD)
            ret_arr = np.append(ret_arr, S_holzapfel)
    return ret_arr


def _bent_cable_Cp(T_arr, beta_1, beta_2, tau, gamma):
    def _q(T, tau, gamma):
        def _indfunc1(T, tau, gamma):
            if np.abs(T - tau) <= gamma:
                return 1
            else:
                return 0

        def _indfunc2(T, tau, gamma):
            if T > tau + gamma:
                return 1
            else:
                return 0

        return ((T - tau + gamma) ** 2 / (4 * gamma)) * _indfunc1(T, tau, gamma) + (
            T - tau
        ) * _indfunc2(T, tau, gamma)

    if isinstance(T_arr, float):
        return beta_1 * T_arr + beta_2 * _q(T_arr, tau, gamma)
    else:
        ret_arr = np.array([])
        for T in T_arr:
            Cp = beta_1 * T + beta_2 * _q(T, tau, gamma)
            ret_arr = np.append(ret_arr, Cp)
        return ret_arr


def _bent_cable_enthalpy(T_arr, beta_1, beta_2, tau, gamma):
    a2h = -beta_2 / (12 * gamma) * (tau - gamma) ** 3
    a3h = beta_2 / 2 * (gamma**2 / 3 + tau**2)
    c1h = beta_1 / 2
    c2h = beta_1 / 2 - beta_2 / (4 * gamma) * (tau - gamma)
    c3h = (beta_1 + beta_2) / 2
    b2h = beta_2 / (4 * gamma) * (tau - gamma) ** 2
    b3h = -beta_2 * tau
    d2h = beta_2 / (12 * gamma)

    def _calc_bcm_enthalpy(T, c1h, a2h, b2h, c2h, d2h, a3h, b3h, c3h):
        if T < tau - gamma:
            return c1h * T**2
        elif all([T >= tau - gamma, T <= tau + gamma]):
            return a2h + b2h * T + c2h * T**2 + d2h * T**3
        else:
            return a3h + b3h * T + c3h * T**2

    if isinstance(T_arr, float):
        ret_arr = _calc_bcm_enthalpy(T_arr, c1h, a2h, b2h, c2h, d2h, a3h, b3h, c3h)
    else:
        ret_arr = np.array([])
        for T in T_arr:
            H_bcm = _calc_bcm_enthalpy(T, c1h, a2h, b2h, c2h, d2h, a3h, b3h, c3h)
            ret_arr = np.append(ret_arr, H_bcm)
    return ret_arr


def _bent_cable_entropy(T_arr, beta_1, beta_2, tau, gamma):
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

    def _calc_bcm_entropy(T, c1s, b2s, b3s, c2s, c3s, d2s, e2s, e3s):
        if T < tau - gamma:
            res = c1s * T**2
        elif all([T >= tau - gamma, T <= tau + gamma]):
            res = b2s * T + c2s * T**2 + d2s * T**3 + e2s * T * np.log(T)
        else:
            res = b3s * T + c3s * T**2 + e3s * T * np.log(T)
        return res / T

    if isinstance(T_arr, float):
        ret_arr = _calc_bcm_entropy(T_arr, c1s, b2s, b3s, c2s, c3s, d2s, e2s, e3s)
    else:
        ret_arr = np.array([])
        for T in T_arr:
            S_bcm = _calc_bcm_entropy(T, c1s, b2s, b3s, c2s, c3s, d2s, e2s, e3s)
            ret_arr = np.append(ret_arr, S_bcm)
    return ret_arr


def _einstein_Cp(T, theta):
    return 3 * R * (theta / T) ** 2 * np.exp(theta / T) / (np.exp(theta / T) - 1) ** 2


def _debye_Cp(T_arr, theta):
    def _integrand(x):
        return x**4 * np.exp(x) / (np.exp(x) - 1) ** 2

    if isinstance(T_arr, float):
        ret_arr = 9 * R * (T_arr / theta) ** 3 * quad(_integrand, 0, theta / T_arr)
    else:
        ret_arr = np.array([])
        for T in T_arr:
            Cp = 9 * R * (T / theta) ** 3 * quad(_integrand, 0, theta / T)[0]
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

    if isinstance(T_arr, float):
        tau = T_arr / Tc
        return R * tau * _g(tau, p) * np.log(beta + 1)
    else:
        ret_arr = np.array([])
        for T in T_arr:
            tau = T / Tc
            Cp = R * tau * _g(tau, p) * np.log(beta + 1)
            ret_arr = np.append(ret_arr, Cp)
        return ret_arr


def _xiong_enthalpy(T_arr, beta, p, Tc):
    # Incorrect. I think I messed up the integral
    def _D(p):
        return 0.33471979 + 0.49649686 * (1 / p - 1)

    def _g_int(T, p, Tc):
        tau = T / Tc
        if tau <= 0:
            return 0
        elif tau <= 1:
            coef = 0.63570895 / _D(p)
            term1 = T**4 / (2 * Tc**3)
            term2 = T**10 / (15 * Tc**9)
            term3 = T**16 / (40 * Tc**15)
            term4 = T**21 / (77 * Tc**21)
            return coef * np.sum([term1, term2, term3, term4])
        else:
            coef = 1 / _D(p)
            term1 = -(T**-6) / (3 * Tc**-7)
            term2 = -(T**-20) / (30 * Tc**-21)
            term3 = -(T**-34) / (85 * Tc**-35)
            term4 = -(T**-48) / (168 * Tc**-49)
            return coef * np.sum([term1, term2, term3, term4])

    if isinstance(T_arr, float):
        return R / (2 * Tc) * T_arr**2 * np.log(beta + 1) * _g_int(T_arr, p, Tc)
    else:
        ret_arr = np.array([])
        for T in T_arr:
            H_xiong = R / (2 * Tc) * T**2 * np.log(beta + 1) * _g_int(T, p, Tc)
            ret_arr = np.append(ret_arr, H_xiong)
        return ret_arr


def _xiong_entropy(T_arr, beta, p, Tc):
    # Incorrect. I think I messed up the integral
    def _D(p):
        return 0.33471979 + 0.49649686 * (1 / p - 1)

    def _g_int(T, p, Tc):
        tau = T / Tc
        if tau <= 0:
            return 0
        elif tau <= 1:
            coef = 0.63570895 / _D(p)
            term1 = T**4 / (2 * Tc**3)
            term2 = T**10 / (15 * Tc**9)
            term3 = T**16 / (40 * Tc**15)
            term4 = T**21 / (77 * Tc**21)
        else:
            coef = 1 / _D(p)
            term1 = -(T**-6) / (3 * Tc**-7)
            term2 = -(T**-20) / (30 * Tc**-21)
            term3 = -(T**-34) / (85 * Tc**-35)
            term4 = -(T**-48) / (168 * Tc**-49)
        return coef * np.sum([term1, term2, term3, term4])

    if isinstance(T_arr, float):
        return R / Tc * np.log(beta + 1) * _g_int(T_arr, p, Tc)
    else:
        ret_arr = np.array([])
        for T in T_arr:
            S_xiong = R / Tc * np.log(beta + 1) * _g_int(T, p, Tc)
            ret_arr = np.append(ret_arr, S_xiong)
        return ret_arr


def _xiong_gibbs(T_arr, beta, p, Tc):
    # From Hao2024
    def _D(p):
        return 0.33471979 + 0.49649686 * (1 / p - 1)

    def _g_int(T, p, Tc):
        tau = T / Tc
        if T / Tc <= 1:
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

    def _gmdo(T, beta, p, Tc):
        return -R * np.log(beta + 1) * (T - 0.38438376 * Tc / (p * _D(p)))

    if isinstance(T_arr, float):
        ret_arr = -R * T_arr / _D(p) * np.log(beta + 1) * _g_int(T_arr, p, Tc)
        +_gmdo(T_arr, beta, p, Tc)
    else:
        ret_arr = np.array([])
        for T in T_arr:
            G_mag = -R * T / _D(p) * np.log(beta + 1) * _g_int(T, p, Tc)
            +_gmdo(T, beta, p, Tc)
            ret_arr = np.append(ret_arr, G_mag)
    return ret_arr


def calc_RSE(model_array, Cp_array):
    num_SR_params = 5  # from the paper
    return np.sqrt(
        np.sum(np.square(model_array - Cp_array))
        / (len(model_array) - num_SR_params - 1)
    )


def _fit_segmented_regression(
    x, temperature_array, Cp_array, beta=None, p=None, Tc=None, use_einstein=False
):
    """
    Function to perform the segmented regression to optimize all modeling parameters.
    x[0]: Debye or Einstein temperature
    x[1]: beta_1
    x[2]: beta_2
    x[3]: tau
    x[4]: gamma
    """
    model_Cp = 0
    if not use_einstein:
        model_Cp += _holzapfel_debye_Cp(temperature_array, x[0])
    # if use_debye:
    #     # Debye model
    #     model_Cp = _debye_Cp(temperature_array, x[0])
    else:
        # Einstein model
        model_Cp += _einstein_Cp(temperature_array, x[0])
    if beta is not None:
        # Assume magnetic model included
        model_Cp += np.array([_xiong_Cp(T, beta, p, Tc) for T in temperature_array])
    model_Cp += _bent_cable_Cp(temperature_array, x[1], x[2], x[3], x[4])

    return calc_RSE(model_Cp, Cp_array)


def segmented_regression(
    dataset_folder, components, phase, dbf=None, use_einstein=False
):
    # TODO: Use the pycalphad DB for loading model properties.
    # Load CPM data for modeling
    datasets = load_datasets(recursive_glob(dataset_folder))
    if isinstance(phase, str):
        phase = [phase]
    query = (
        (where("phases") == phase)
        & (where("components") == components)
        & (where("output") == "CPM")
    )
    search_results = datasets.search(query)
    df = pd.DataFrame()
    # CPM plotting
    for result in search_results:
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

    # optimize the model parameters
    # initialize with values for iron
    x0 = [theta_Fe, 8.94e-3, 3.5e-2, 1.7e3, 9.6e1]
    min_fits = minimize(
        _fit_segmented_regression,
        x0=x0,
        args=(
            df["temperature"],
            df["Cp"],
            beta_Fe,
            struct_fact_bcc,
            Tc_Fe,
            use_einstein,
        ),
        bounds=((1, 1500), (0, 0.1), (0, 0.1), (0, np.inf), (0, 500)),
        method="nelder-mead",
    )

    return df, min_fits


def calc_enthalpy(T_arr, Cp_fits, xiong_params=None):
    if isinstance(T_arr, float):
        enthalpy = 0
    else:
        enthalpy = np.zeros(len(T_arr))
    enthalpy += _holzapfel_enthalpy(T_arr, *Cp_fits.x[:1])
    # if xiong_params is not None:
    # enthalpy += _xiong_enthalpy(T_arr, *xiong_params)
    enthalpy += _bent_cable_enthalpy(T_arr, *Cp_fits.x[1:])
    return enthalpy


def calc_entropy(T_arr, Cp_fits, xiong_params=None):
    if isinstance(T_arr, float):
        entropy = 0
    else:
        entropy = np.zeros(len(T_arr))
    entropy += _holzapfel_entropy(T_arr, *Cp_fits.x[:1])
    # if xiong_params is not None:
    # entropy += _xiong_entropy(T_arr, *xiong_params)
    entropy += _bent_cable_entropy(T_arr, *Cp_fits.x[1:])
    return entropy


def calc_gibbs_energy(T_arr, Cp_fits, xiong_params=None):
    if isinstance(T_arr, float):
        ret_arr = calc_enthalpy(T_arr, Cp_fits, xiong_params) - T_arr * calc_entropy(
            T_arr, Cp_fits, xiong_params
        )
    else:
        ret_arr = np.array([])
        for T in T_arr:
            gibbs = calc_enthalpy(T, Cp_fits, xiong_params) - T * calc_entropy(
                T, Cp_fits, xiong_params
            )
            ret_arr = np.append(ret_arr, gibbs)
    if xiong_params is not None:
        ret_arr += _xiong_gibbs(T_arr, *xiong_params)
    return ret_arr
