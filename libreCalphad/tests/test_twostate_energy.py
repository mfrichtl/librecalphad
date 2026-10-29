"""Two-state Gibbs substitution, symbolic dispatch, and Cp consistency."""

from copy import deepcopy

from libreCalphad.models import energy as en
from libreCalphad.models import heat_capacity as hc
import numpy as np
import pytest
import symengine as se


@pytest.mark.parametrize("symbol_objects", [False, True])
def test_twostate_gibbs_parameter_substitution(symbol_objects):
    T, a, b, c = se.symbols("T a b c")
    expression = a + b * T + c * T * se.log(T)
    symbols = [a, b, c] if symbol_objects else ["a", "b", "c"]
    parameters = {
        "a": [15000, [7500, 17500], "fit"],
        "b": [-5, "fix"],
        "c": [-1, "fix"],
    }
    original = deepcopy(parameters)
    expected = en._twostate_gibbs(1000, 15000 - 5 * T - T * se.log(T))

    actual = en._twostate_gibbs(1000, expression, symbols, **parameters)
    symbolic = en._twostate_gibbs(None, expression, symbols, True, **parameters)

    assert actual == pytest.approx(expected, rel=1e-12)
    assert float(symbolic.subs(T, 1000)) == pytest.approx(expected, rel=1e-12)
    assert symbolic.free_symbols == {T}
    assert parameters == original


@pytest.mark.parametrize(
    "temperatures", [None, 1000, 1000.0, [300, 1000], np.array([1000])]
)
def test_twostate_gibbs_symbolic_dispatch_ignores_temperature(temperatures):
    T = se.Symbol("T")
    gap = 15000 - 5 * T - T * se.log(T)
    expected = -en.R * T * se.log(1 + se.exp(-gap / (en.R * T)))

    # Preserve the positional symbols/ret_expr interface as well as keyword use.
    actual = en._twostate_gibbs(temperatures, gap, [], True)

    assert actual == expected


@pytest.mark.parametrize("scalar_type", [int, float, np.int64, np.float32, np.float64])
def test_twostate_gibbs_scalar_reference(scalar_type):
    temperature = scalar_type(1000)
    temp = float(temperature)
    gap = 15000 - 5 * temp - temp * np.log(temp)
    expected = -en.R * temp * np.logaddexp(0, -gap / (en.R * temp))

    actual = en._twostate_gibbs(temperature, "15000 - 5*T - T*log(T)")

    assert np.isscalar(actual)
    assert actual == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    "temperatures",
    [[], [300, 1000, 1500], (300, 1000), np.array([[300, 800], [1000, 1500]])],
)
def test_twostate_gibbs_arrays_match_independent_formula(temperatures):
    array = np.asarray(temperatures, dtype=float)
    gap = 15000 - 5 * array - array * np.log(array)
    expected = -en.R * array * np.logaddexp(0, -gap / (en.R * array))

    actual = en._twostate_gibbs(temperatures, "15000 - 5*T - T*log(T)")

    assert isinstance(actual, np.ndarray)
    assert actual.shape == array.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-12)


@pytest.mark.parametrize(
    "expression",
    [
        "15000",
        "15000 - 5*T",
        "15000 - 5*T - T*log(T)",
        "-15000 + 5*T + T*log(T)",
        "15000 + 0.01*T**2",
        "0",
    ],
)
def test_twostate_gibbs_derivative_matches_heat_capacity(expression):
    T = se.Symbol("T")
    gibbs = en._twostate_gibbs(None, expression, ret_expr=True)
    capacity = -T * gibbs.diff(T, 2)
    temperatures = np.array([300.0, 1000.0, 1500.0])
    expected = hc._twostate_Cp(
        temperatures, expression=se.sympify(expression), symbols=[]
    )

    actual = [float(capacity.subs(T, temperature)) for temperature in temperatures]

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("ret_expr", [False, True])
def test_twostate_gibbs_missing_parameter_has_actionable_error(ret_expr):
    with pytest.raises(
        ValueError, match="Missing two-state parameter 'a'.*a=\\[value\\]"
    ):
        en._twostate_gibbs(1000, "a - 5*T", symbols=["a"], ret_expr=ret_expr)


@pytest.mark.parametrize("values", [None, 15000, [], np.array(15000)])
def test_twostate_gibbs_invalid_parameter_sequence(values):
    with pytest.raises(ValueError, match="parameter 'a' must be a nonempty sequence"):
        en._twostate_gibbs(1000, "a - 5*T", symbols=["a"], a=values)


def test_twostate_gibbs_requires_expression():
    with pytest.raises(ValueError, match="requires a Gibbs-difference expression"):
        en._twostate_gibbs(1000)


def test_twostate_gibbs_unresolved_parameters_only_allowed_symbolically():
    T, a = se.symbols("T a")
    expression = a - 5 * T

    with pytest.raises(ValueError, match="Unresolved two-state parameters: a"):
        en._twostate_gibbs(1000, expression)

    symbolic = en._twostate_gibbs(None, expression, ret_expr=True)
    assert symbolic.free_symbols == {T, a}
