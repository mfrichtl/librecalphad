from libreCalphad.models.martensite_start.martensite_start import get_lath_model_mf
from libreCalphad.models.utilities import convert_conditions, get_components_from_conditions
from numpy.testing import assert_raises
from pycalphad import variables as v

def test_convert_conditions():
    conditions = {'T': 1000}
    convert_conditions(conditions)

def test_get_components_with_weights():
    # Originally assumed only molar fractions would be used
    conditions = {v.W('C'): 0.0015}
    components = get_components_from_conditions(conditions, dependent_component='FE')
    assert len(components) == 3

def test_martensite_start_raises_error_on_weights():
    # Make sure the martensite-start functions promp the user to use molar fractions.
    conditions = {v.W('C'): 0.01}
    with assert_raises(AssertionError):
        get_lath_model_mf(conditions)