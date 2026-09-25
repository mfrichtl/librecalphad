import pytest
from libreCalphad.models.martensite_start.martensite_start import get_lath_model_mf
from libreCalphad.models.utilities import (
    convert_conditions,
    get_components_from_conditions,
    parse_composition,
    trim_conditions,
)
from pycalphad import variables as v


def test_convert_conditions():
    conditions = {"T": 1000}
    convert_conditions(conditions)


def test_get_components_with_weights():
    # Originally assumed only molar fractions would be used
    conditions = {v.W("C"): 0.0015}
    components = get_components_from_conditions(conditions, dependent_component="FE")
    assert len(components) == 3


def test_martensite_start_raises_error_on_weights():
    # Make sure the martensite-start functions prompt the user to use molar fractions.
    conditions = {v.W("C"): 0.01}
    with pytest.raises(AssertionError):
        get_lath_model_mf(conditions)


def test_pass_empty_material_definition():
    # Make sure a proper material definition is passed to the parser function.
    test_row = {"material_at%": "", "material_wt%": ""}
    with pytest.raises(ValueError, match="dependent component"):
        parse_composition(test_row, dependent_element="FE")


def test_trim_conditions_raises_error_when_only_protected_conditions_remain():
    components = ["FE", "C", "VA"]
    conditions = {v.T: 1000, v.P: 101325, v.X("C"): 0.01}

    with pytest.raises(ValueError, match="Cannot reduce conditions"):
        trim_conditions(
            components,
            conditions,
            max_num_conditions=2,
            always_keep_list=["C"],
        )
