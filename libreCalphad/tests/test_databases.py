from libreCalphad.databases.db_utils import upsert_db_param_from_models
import importlib.resources as impresources
import numpy as np
from pycalphad import Database
import symengine as se
from tinydb import where


def test_upsert_db_param_from_einstein_model():
    db_file = impresources.files("libreCalphad.tests.test_database_files") / "test.tdb"
    dbf = Database(db_file)
    model_dict = {"einstein": {"theta": [300, "fix"]}}
    species_dict = {sp.name: sp for sp in dbf.species}
    constituent_array = ((species_dict["FE"],), (species_dict["VA"],))
    query = (
        (where("phase_name") == "BCC_A2")
        & (where("parameter_type") == "THETA")
        & (where("constituent_array") == constituent_array)
    )
    dbf = upsert_db_param_from_models(dbf, model_dict, "BCC_A2", constituent_array)
    search = dbf.search(query)
    assert len(search) == 1
    assert np.isclose(300, np.exp(float(search[0]["parameter"].args[0])))


def test_upsert_db_param_from_xiong_model():
    db_file = impresources.files("libreCalphad.tests.test_database_files") / "test.tdb"
    dbf = Database(db_file)
    model_dict = {
        "xiong": {"beta": [5, "fix"], "p": [0.27, "fix"], "Tc": [1000, "fix"]}
    }
    species_dict = {sp.name: sp for sp in dbf.species}
    constituent_array = ((species_dict["FE"],), (species_dict["VA"],))
    dbf = upsert_db_param_from_models(dbf, model_dict, "BCC_A2", constituent_array)
    query = (
        (where("phase_name") == "BCC_A2")
        & (where("parameter_type") == "BMAGN")
        & (where("constituent_array") == constituent_array)
    )
    search = dbf.search(query)
    assert np.isclose(5, float(search[0]["parameter"].args[0]))

    query = (
        (where("phase_name") == "BCC_A2")
        & (where("parameter_type") == "TC")
        & (where("constituent_array") == constituent_array)
    )
    search = dbf.search(query)
    assert len(search) == 1
    assert np.isclose(1000, float(search[0]["parameter"].args[0]))


def test_upsert_db_param_from_twostate_model():
    db_file = impresources.files("libreCalphad.tests.test_database_files") / "test.tdb"
    dbf = Database(db_file)
    model_dict = {
        "two-state": {
            "expression": "a + b*T",
            "a": [10000, "fix"],
            "b": [5, "fix"],
            "symbols": ["a", "b"],
        }
    }
    species_dict = {sp.name: sp for sp in dbf.species}
    constituent_array = ((species_dict["FE"],), (species_dict["VA"],))
    dbf = upsert_db_param_from_models(dbf, model_dict, "BCC_A2", constituent_array)
    query = (
        (where("phase_name") == "BCC_A2")
        & (where("parameter_type") == "GD")
        & (where("constituent_array") == constituent_array)
    )
    search = dbf.search(query)
    T = se.symbols("T")
    assert search[0]["parameter"].args[0] == 10000 + 5 * T
