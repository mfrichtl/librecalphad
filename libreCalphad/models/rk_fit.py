# ==============================================================
# 0️⃣  Imports & global constants
# ==============================================================

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional, Union

# pycalphad imports
from pycalphad import Database, Model, equilibrium
from pycalphad import variables as v

R_J = 8.314462618  # J·mol‑1·K‑1


# ==============================================================
# 1️⃣  CALPHAD reference‑state Gibbs energy (supports custom phase)
# ==============================================================


def ref_gibbs_energy(
    db_path: Union[str, Database],
    element: str,
    temperature: float,
    phase_name: Optional[str] = None,
    pressure: float = 101325.0,
) -> float:
    """
    Return the molar Gibbs energy (J/mol) of a pure element.

    Parameters
    ----------
    db_path : str | pycalphad.Database
        Path to the CALPHAD database or a pre‑loaded Database object.
    element : str
        Chemical symbol, e.g. "Fe".
    temperature : float
        Temperature in Kelvin.
    phase_name : str, optional
        Name of the phase that represents the pure element.
        If None, the first available phase is used.
    pressure : float, optional
        Pressure in Pascal (default = 1 atm).

    Returns
    -------
    g_ref : float
        Molar Gibbs energy of the pure element (J/mol).
    """
    # Load or reuse the database
    if isinstance(db_path, str):
        db = Database(db_path)
    else:
        db = db_path

    # Choose the reference phase
    if phase_name is None:
        phase_name = "ELEMENT" if "ELEMENT" in db.phases else list(db.phases.keys())[0]
    else:
        if phase_name not in db.phases:
            raise KeyError(
                f"Phase '{phase_name}' not found in database. "
                f"Available: {list(db.phases.keys())}"
            )

    # Build a model for the pure element (single sublattice, single species)
    # NOTE: we include a vacancy species ("VA") because many unary databases
    # define the pure element as a sublattice with X(element) + VA.
    comps = [element, "VA"]
    res = equilibrium(
        db,
        comps=comps,
        phases=[phase_name],
        conditions={v.T: temperature, v.P: pressure},
        output="GM",
    )

    # The result is a DataArray with dimensions (T, P).  Squeeze to a scalar.
    g_ref = float(res.GM.squeeze().values)
    return g_ref


# ==============================================================
# 2️⃣  Redlich‑Kister activity‑coefficient calculator (unchanged)
# ==============================================================


def rk_gamma(
    x_a: float, temperature: float, L_vals: List[float]
) -> Tuple[float, float]:
    """Return (γ_A, γ_B) for a binary solid solution."""
    if not (0.0 < x_a < 1.0):
        raise ValueError("x_a must be between 0 and 1.")
    x_b = 1.0 - x_a
    delta = x_a - x_b  # = 2·x_a – 1

    g_ex = sum(Lk * (delta**k) for k, Lk in enumerate(L_vals))
    g_ex *= x_a * x_b

    deriv = sum((k + 1) * Lk * (delta**k) for k, Lk in enumerate(L_vals))

    ln_gamma_a = (g_ex / (R_J * temperature)) + (x_b**2) * deriv / (R_J * temperature)
    ln_gamma_b = (g_ex / (R_J * temperature)) + (x_a**2) * deriv / (R_J * temperature)

    return np.exp(ln_gamma_a), np.exp(ln_gamma_b)


# ==============================================================
# 3️⃣  Convert activities → excess Gibbs (uses CALPHAD refs)
# ==============================================================


def excess_gibbs_from_activities(
    row: pd.Series,
    db_path: Union[str, Database],
    elem_phase_map: Dict[str, str],
    element_A: str,
    element_B: str,
    provisional_L: Optional[List[float]] = None,
) -> float:
    """
    Compute excess Gibbs energy (J/mol) for a single measurement.
    """
    T = float(row["T"])
    x_a = float(row["x_A"])
    x_b = 1.0 - x_a
    a_a = float(row["a_A"])

    # ----- activity of B -------------------------------------------------
    if "a_B" in row and pd.notna(row["a_B"]):
        a_b = float(row["a_B"])
    else:
        if provisional_L is None:
            a_b = x_b  # ideal B
        else:
            _, gamma_b = rk_gamma(x_a, T, provisional_L)
            a_b = gamma_b * x_b

    # ----- Reference Gibbs energies (user‑supplied phases) -------------
    phase_A = elem_phase_map.get(
        element_A
    )  # may be None → fallback inside ref_gibbs_energy
    phase_B = elem_phase_map.get(element_B)

    g_ref_a = ref_gibbs_energy(
        db_path, element=element_A, temperature=T, phase_name=phase_A
    )
    g_ref_b = ref_gibbs_energy(
        db_path, element=element_B, temperature=T, phase_name=phase_B
    )

    # ----- Convert absolute activities to activity coefficients ----------
    gamma_a = a_a / x_a
    gamma_b = a_b / x_b

    # ----- Equation (1): G_ex = RT[ x_a*ln(gamma_a) + x_b*ln(gamma_b) ] ---
    term_a = x_a * np.log(gamma_a) if gamma_a > 0 else 0.0
    term_b = x_b * np.log(gamma_b) if gamma_b > 0 else 0.0
    g_ex = R_J * T * (term_a + term_b)

    # (Optional) absolute mixture Gibbs energy:
    # G_mix = x_a * g_ref_a + x_b * g_ref_b + g_ex

    return g_ex


# ==============================================================
# 4️⃣  Design‑matrix builders
# ==============================================================


def rk_design_matrix(delta_x: np.ndarray, order: int) -> np.ndarray:
    """Columns: [1, Δx, Δx², …, Δx^order] (temperature‑independent)."""
    cols = [delta_x**p for p in range(order + 1)]
    return np.column_stack(cols)


def rk_design_matrix_temp(
    delta_x: np.ndarray, temperature: np.ndarray, order: int
) -> np.ndarray:
    """
    Columns (in order):
        1, Δx, Δx², …, Δx^order,
        T, T·Δx, T·Δx², …, T·Δx^order
    """
    a_cols = [delta_x**p for p in range(order + 1)]
    b_cols = [temperature * (delta_x**p) for p in range(order + 1)]
    return np.column_stack(a_cols + b_cols)


# ==============================================================
# 5️⃣  Load CSV/Excel and rename columns according to user mapping
# ==============================================================


def load_and_rename(filepath: str, col_map: Dict[str, str]) -> pd.DataFrame:
    """
    Read a CSV or Excel file and rename its columns to the canonical names:
    'T', 'x_A', 'x_B', 'a_A', 'a_B'.
    """
    if filepath.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)

    # Invert mapping: actual_name → canonical_name
    inv_map = {v: k for k, v in col_map.items()}
    df = df.rename(columns=inv_map)

    # Auto‑create x_B if missing
    if "x_B" not in df.columns and "x_A" in df.columns:
        df["x_B"] = 1.0 - df["x_A"]
    return df


# ==============================================================
# 6️⃣  Core fitting routine – temperature‑dependent RK
# ==============================================================


def fit_rk_temp_dependent(
    df: pd.DataFrame,
    db_path: Union[str, Database],
    element_A: str,
    element_B: str,
    elem_phase_map: Dict[str, str],
    RK_order: int = 2,
    provisional_L: Optional[List[float]] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Fit L_k(T) = a_k + b_k·T to activity‑derived excess Gibbs data for binary solutions.

    Parameters
    ----------
    df : pandas.DataFrame
        Prepared input data.
    db_path : string | pycalphad Database object
        The thermodynamic database to use for reference states.
    element_A : string
        2-letter string for element A
    element_B : string
        2-letter string for element B
    elem_phase_map : dictionary
        Dictionary with element keys and phase values for element_A and element_B reference states.
    RK_order : int
        The order to use for the Redlich-Kister expansion.

    Returns
    -------
    a_vec : ndarray
        Temperature‑independent coefficients (a_0 … a_N) [J·mol⁻¹].
    b_vec : ndarray
        Temperature‑dependent slopes (b_0 … b_N) [J·mol⁻¹·K⁻¹].
    diagnostics : DataFrame
        Original G_ex, predicted G_ex, residuals, and the instantaneous
        L_k(T) values for each measurement.
    """
    df = df.copy()

    # ----- 1) Compute G_ex for every row (CALPHAD references) ----------
    df["G_ex"] = df.apply(
        lambda r: excess_gibbs_from_activities(
            r, db_path, elem_phase_map, element_A, element_B, provisional_L
        ),
        axis=1,
    )

    # ----- 2) Remove pure‑component points (x_A = 0 or 1) ---------------
    mask = (df["x_A"] > 0) & (df["x_A"] < 1)
    if not mask.all():
        print("Warning: pure‑component points removed from the fit.")
    df = df.loc[mask]

    # ----- 3) Linearised variables ------------------------------------
    df["x_B"] = 1.0 - df["x_A"]
    df["delta_x"] = df["x_A"] - df["x_B"]  # = 2·x_A – 1
    df["y"] = df["G_ex"] / (df["x_A"] * df["x_B"])

    X = rk_design_matrix_temp(
        df["delta_x"].values, df["T"].values, RK_order
    )  # (n, 2*(RK_order+1))
    y = df["y"].values

    # ----- 4) Least‑squares solution -----------------------------------
    beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

    n_coeff = RK_order + 1
    a_vec = beta[:n_coeff]  # a_0 … a_N
    b_vec = beta[n_coeff:]  # b_0 … b_N

    # ----- 5) Re‑construct G_ex_pred using temperature‑dependent L_k(T) ----
    # Compute L_k(T) for each measurement
    L_T = np.zeros((len(df), n_coeff))
    for k in range(n_coeff):
        L_T[:, k] = a_vec[k] + b_vec[k] * df["T"].values

    # Build Δx^k matrix
    delta_pow = np.vstack([df["delta_x"].values ** p for p in range(n_coeff)]).T
    g_ex_pred = (df["x_A"] * df["x_B"]).values * np.sum(L_T * delta_pow, axis=1)

    df["G_ex_pred"] = g_ex_pred
    df["residual"] = df["G_ex"] - df["G_ex_pred"]

    # ----- 6) Attach the instantaneous L_k(T) values to the diagnostics df ----
    coeffs_per_point = pd.DataFrame(
        L_T, columns=[f"L{k}" for k in range(n_coeff)], index=df.index
    )
    coeffs_per_point["T"] = df["T"]
    df = pd.concat([df, coeffs_per_point], axis=1)

    return a_vec, b_vec, df

