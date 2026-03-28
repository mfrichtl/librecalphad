from importlib import resources as impresources
from libreCalphad import databases, tests
from libreCalphad.models.rk_fit import fit_rk_temp_dependent, load_and_rename
import numpy as np
import pandas as pd


def test_temp_dependence():
    synthetic_data = str(impresources.files(tests) / "synthetic_temp_dep.csv")
    col_map = {"T": "Temp_K", "x_A": "MolFrac_A", "a_A": "Act_A", "A_B": "Act_B"}
    synthetic_df = load_and_rename(synthetic_data, col_map)
    elem_phase_map = {"FE": "FCC_A1", "NI": "FCC_A1"}

    ref_db = str(impresources.files(databases) / "unary50.tdb")
    a_fit, b_fit, diagnostics = fit_rk_temp_dependent(
        synthetic_df,
        db_path=ref_db,
        element_A="FE",
        element_B="NI",
        elem_phase_map=elem_phase_map,
        RK_order=2,
    )
    rmse = np.sqrt(np.mean(diagnostics["residual"] ** 2))
    assert np.isclose(a_fit[0], 4502.46, rtol=1e-2)
    assert np.isclose(a_fit[1], -989.95, rtol=1e-2)
    assert np.isclose(a_fit[2], -1059.04, rtol=1e-2)
    assert np.isclose(b_fit[0], 4.9937, rtol=1e-4)
    assert np.isclose(b_fit[1], -1.5293, rtol=1e-4)
    assert np.isclose(b_fit[2], 1.6915, rtol=1e-4)
    assert np.isclose(rmse, 18.60, rtol=1e-2)
