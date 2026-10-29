"""Two-state energies, symbolic dispatch, and thermodynamic consistency."""

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


@pytest.mark.parametrize("scalar_type", [int, float, np.int64, np.float32, np.float64])
@pytest.mark.parametrize("temperature", [300, 1000, 1500])
def test_twostate_entropy_enthalpy_constant_gap_reference(scalar_type, temperature):
    # Independent population formulas for a constant gap between equal-degeneracy states.
    gap = 15000
    population = 1 / (1 + np.exp(gap / (en.R * temperature)))
    expected_entropy = -en.R * (
        population * np.log(population)
        + (1 - population) * np.log1p(-population)
    )
    expected_enthalpy = population * gap

    entropy = en._twostate_entropy(scalar_type(temperature), gap, [])
    enthalpy = en._twostate_enthalpy(scalar_type(temperature), gap, [])

    assert np.isscalar(entropy)
    assert np.isscalar(enthalpy)
    assert entropy == pytest.approx(expected_entropy, rel=1e-12)
    assert enthalpy == pytest.approx(expected_enthalpy, rel=1e-12)


@pytest.mark.parametrize(
    "expression,gap,derivative",
    [
        ("15000", lambda t: 15000, lambda t: 0),
        ("15000 - 5*T", lambda t: 15000 - 5 * t, lambda t: -5),
        (
            "15000 - 5*T - T*log(T)",
            lambda t: 15000 - 5 * t - t * np.log(t),
            lambda t: -6 - np.log(t),
        ),
        (
            "-15000 + 5*T + T*log(T)",
            lambda t: -15000 + 5 * t + t * np.log(t),
            lambda t: 6 + np.log(t),
        ),
        ("15000 + 0.01*T**2", lambda t: 15000 + 0.01 * t**2, lambda t: 0.02 * t),
        ("0", lambda t: 0, lambda t: 0),
    ],
)
def test_twostate_entropy_enthalpy_temperature_dependent_gap(expression, gap, derivative):
    temperatures = np.array([300.0, 1000.0, 1500.0])
    delta_g = gap(temperatures)
    delta_h = delta_g - temperatures * derivative(temperatures)
    population = 1 / (1 + np.exp(delta_g / (en.R * temperatures)))
    expected_enthalpy = population * delta_h
    expected_entropy = (
        en.R * np.logaddexp(0, -delta_g / (en.R * temperatures))
        + population * delta_h / temperatures
    )

    entropy = en._twostate_entropy(temperatures, expression, [])
    enthalpy = en._twostate_enthalpy(temperatures, expression, [])
    gibbs = en._twostate_gibbs(temperatures, expression)

    np.testing.assert_allclose(entropy, expected_entropy, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(enthalpy, expected_enthalpy, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        enthalpy, gibbs + temperatures * entropy, rtol=1e-12, atol=1e-12
    )

    T = se.Symbol("T")
    symbolic_entropy = en._twostate_entropy(None, expression, [], True)
    symbolic_enthalpy = en._twostate_enthalpy(None, expression, [], True)
    capacity = hc._twostate_Cp(
        temperatures, expression=se.sympify(expression), symbols=[]
    )
    for symbolic, numerical in (
        (symbolic_entropy, entropy),
        (symbolic_enthalpy, enthalpy),
    ):
        evaluated = [float(symbolic.subs(T, temp)) for temp in temperatures]
        np.testing.assert_allclose(evaluated, numerical, rtol=1e-12, atol=1e-12)
    for capacity_expression in (
        symbolic_enthalpy.diff(T),
        T * symbolic_entropy.diff(T),
    ):
        evaluated = [float(capacity_expression.subs(T, temp)) for temp in temperatures]
        np.testing.assert_allclose(evaluated, capacity, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("helper", [en._twostate_entropy, en._twostate_enthalpy])
@pytest.mark.parametrize(
    "temperatures", [[], [300, 1000], (300, 1000), np.array([[300, 800], [1000, 1500]])]
)
def test_twostate_thermal_arrays(helper, temperatures):
    array = np.asarray(temperatures)
    expected = np.array([helper(float(temp), 15000, []) for temp in array.flat])

    actual = helper(temperatures, 15000, [])

    assert isinstance(actual, np.ndarray)
    assert actual.shape == array.shape
    np.testing.assert_allclose(actual, expected.reshape(array.shape), rtol=1e-12)


@pytest.mark.parametrize("helper", [en._twostate_entropy, en._twostate_enthalpy])
@pytest.mark.parametrize("temperatures", [None, 1000, [300, 1000], np.array([1000])])
def test_twostate_thermal_symbolic_dispatch(helper, temperatures):
    T = se.Symbol("T")
    gibbs = en._twostate_gibbs(None, 15000, ret_expr=True)
    expected = (
        -gibbs.diff(T) if helper is en._twostate_entropy else gibbs - T * gibbs.diff(T)
    )

    actual = helper(temperatures, 15000, [], True)

    assert actual == expected


@pytest.mark.parametrize("helper", [en._twostate_entropy, en._twostate_enthalpy])
def test_twostate_thermal_weighted_terms(helper):
    T = se.Symbol("T")
    terms = [15000, -5, -1]
    coefficients = [1, T, T * se.log(T)]
    original = (terms.copy(), coefficients.copy())
    temperatures = [300, 1000, 1500]
    expression = 15000 - 5 * T - T * se.log(T)

    actual = helper(temperatures, terms, coefficients)
    expected = helper(temperatures, expression, [])
    symbolic = helper(None, terms, coefficients, True)
    evaluated = [float(symbolic.subs(T, temp)) for temp in temperatures]

    np.testing.assert_allclose(actual, expected, rtol=1e-12)
    np.testing.assert_allclose(evaluated, expected, rtol=1e-12)
    assert (terms, coefficients) == original
    # Preserve the existing numerical interpretation as a weighted sum, not a polynomial.
    assert helper(1000, [5000, 2500], [2, 2]) == pytest.approx(
        helper(1000, 15000, [])
    )


@pytest.mark.parametrize("helper", [en._twostate_entropy, en._twostate_enthalpy])
@pytest.mark.parametrize("coefficients", [None, [], [1], [1, 2, 3], 1, np.array(1)])
def test_twostate_thermal_invalid_weighted_terms(helper, coefficients):
    with pytest.raises(ValueError, match="coef_list.*one coefficient per.*dE"):
        helper(1000, [15000, -5], coefficients)


@pytest.mark.parametrize("helper", [en._twostate_entropy, en._twostate_enthalpy])
def test_twostate_thermal_unresolved_parameters(helper):
    T, a = se.symbols("T a")
    with pytest.raises(ValueError, match="Unresolved two-state parameters: a"):
        helper(1000, a - T, [])

    symbolic = helper(None, a - T, [], True)
    assert symbolic.free_symbols == {T, a}


def test_twostate_thermal_numerical_derivatives():
    expression = "15000 - 5*T - T*log(T)"
    temperatures = np.array([300.0, 1000.0, 1500.0])
    step = 0.01
    lower_g = en._twostate_gibbs(temperatures - step, expression)
    upper_g = en._twostate_gibbs(temperatures + step, expression)
    lower_h = en._twostate_enthalpy(temperatures - step, expression, [])
    upper_h = en._twostate_enthalpy(temperatures + step, expression, [])
    entropy = en._twostate_entropy(temperatures, expression, [])
    capacity = hc._twostate_Cp(
        temperatures, expression=se.sympify(expression), symbols=[]
    )

    np.testing.assert_allclose(-(upper_g - lower_g) / (2 * step), entropy, rtol=1e-8)
    np.testing.assert_allclose((upper_h - lower_h) / (2 * step), capacity, rtol=1e-8)
