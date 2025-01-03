from libreCalphad.models.utilities import convert_conditions
from pycalphad import variables as v

def test_convert_conditions():
    conditions = {v.T: 1000}
    convert_conditions(conditions)

def test_get_components_with_weights():
    # Originally assumed only molar fractions would be used
    conditions = {v.W('C'): 0.0015}
    components = get_components_from_conditions(conditions, dependent_component='FE')
    assert len(components) == 3