from libreCalphad.models.utilities import convert_conditions
from pycalphad import variables as v

def test_convert_conditions():
    conditions = {v.T: 1000}
    convert_conditions(conditions)