"""Exact Debye energies, independently checked against thermodynamic identities."""

from libreCalphad.models import energy as en
from libreCalphad.models import heat_capacity as hc
import numpy as np
import pytest
import sympy as sp


ENERGY_FUNCTIONS = ("_debye_enthalpy", "_debye_entropy", "_debye_gibbs")


@pytest.mark.parametrize("scalar_type", [int, float, np.int64, np.float32, np.float64])
@pytest.mark.parametrize(
    "temperature,reference_over_R",
    [
        (0, (337.5, 0.0, 337.5)),
        (50, (349.137198710064453, 0.317770787041930664, 333.248659357967920)),
        (300, (944.474007670033212, 4.073687692472504393, -277.632300071718106)),
        (1500, (4508.995717457722276, 8.831311596207312876, -8737.971676853247039)),
    ],
)
def test_debye_energy_reference_values(temperature, reference_over_R, scalar_type):
    # 50-digit quadrature over oscillator frequencies u in [0, 1], theta = 300 K:
    # H/R = 9*theta/8 + 9*theta*Integral(u**3/expm1(theta*u/T), du)
    # G/R = 9*theta/8 + 9*T*Integral(u**2*log(1-exp(-theta*u/T)), du)
    actual = [
        getattr(en, name)(scalar_type(temperature), theta=300)
        for name in ENERGY_FUNCTIONS
    ]

    assert all(np.isscalar(value) for value in actual)
    np.testing.assert_allclose(actual, en.R * np.array(reference_over_R), rtol=1e-12)


@pytest.mark.parametrize("function_name", ENERGY_FUNCTIONS)
@pytest.mark.parametrize(
    "temperatures", [[], [0, 50, 300], (0, 50, 300), np.array([[0, 50], [300, 1500]])]
)
def test_debye_energy_arrays(function_name, temperatures):
    function = getattr(en, function_name)
    temperature_array = np.asarray(temperatures)
    expected = np.array(
        [
            function(float(temperature), theta=300)
            for temperature in temperature_array.flat
        ]
    ).reshape(temperature_array.shape)

    actual = function(temperatures, theta=300)

    assert isinstance(actual, np.ndarray)
    assert actual.shape == temperature_array.shape
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("function_name", ENERGY_FUNCTIONS)
@pytest.mark.parametrize("theta", [0, -300, np.nan, np.inf])
@pytest.mark.parametrize("ret_expr", [False, True])
def test_debye_energy_rejects_invalid_theta(function_name, theta, ret_expr):
    with pytest.raises(ValueError, match="theta must be finite and positive"):
        getattr(en, function_name)(300, theta=theta, ret_expr=ret_expr)


@pytest.mark.parametrize("function_name", ENERGY_FUNCTIONS)
@pytest.mark.parametrize("temperature", [-1, np.nan, np.inf])
def test_debye_energy_rejects_invalid_temperatures(function_name, temperature):
    with pytest.raises(ValueError, match="temperatures must be finite and nonnegative"):
        getattr(en, function_name)([300, temperature], theta=300)


def test_debye_numeric_energy_identity_and_limits():
    temperatures = np.array([0.0, 3.0, 50.0, 300.0, 1500.0, 3e8])
    enthalpy = en._debye_enthalpy(temperatures, theta=300)
    entropy = en._debye_entropy(temperatures, theta=300)
    gibbs = en._debye_gibbs(temperatures, theta=300)

    np.testing.assert_allclose(gibbs, enthalpy - temperatures * entropy, rtol=1e-12)
    assert entropy[0] == 0
    assert enthalpy[0] == gibbs[0] == 9 * en.R * 300 / 8
    assert entropy[1] == pytest.approx(4 * np.pi**4 / 5 * en.R * (3 / 300) ** 3)
    assert enthalpy[-1] == pytest.approx(3 * en.R * temperatures[-1], rel=1e-10)
    assert entropy[-1] == pytest.approx(
        en.R * (4 + 3 * np.log(temperatures[-1] / 300)), rel=1e-10
    )


@pytest.mark.parametrize("theta", [100, 300])
def test_debye_symbolic_energies_match_numeric_and_heat_capacity(theta):
    temperature = sp.Symbol("T")
    enthalpy, entropy, gibbs = [
        getattr(en, name)(None, theta=theta, ret_expr=True)
        for name in ENERGY_FUNCTIONS
    ]
    for expression in (enthalpy, entropy, gibbs):
        assert isinstance(expression, sp.Expr)
        assert expression.has(sp.Integral)
    heat_capacity = -temperature * sp.diff(gibbs, temperature, 2)
    entropy_from_gibbs = -sp.diff(gibbs, temperature)
    enthalpy_from_gibbs = gibbs + temperature * entropy_from_gibbs
    for temp in theta * np.array([0.03, 0.2, 1.0, 5.0]):
        substitutions = {temperature: float(temp)}
        symbolic = [
            float(expression.subs(substitutions).evalf())
            for expression in (enthalpy, entropy, gibbs)
        ]
        numeric = [getattr(en, name)(float(temp), theta) for name in ENERGY_FUNCTIONS]
        np.testing.assert_allclose(symbolic, numeric, rtol=1e-10)
        assert float(heat_capacity.subs(substitutions).evalf()) == pytest.approx(
            hc._debye_Cp(float(temp), theta), rel=1e-10
        )
        assert float(entropy_from_gibbs.subs(substitutions).evalf()) == pytest.approx(
            numeric[1], rel=1e-10
        )
        assert float(enthalpy_from_gibbs.subs(substitutions).evalf()) == pytest.approx(
            numeric[0], rel=1e-10
        )


@pytest.mark.parametrize("function_name", ENERGY_FUNCTIONS)
def test_debye_symbolic_theta_and_zero_limit(function_name):
    temperature = sp.Symbol("T")
    theta = sp.Symbol("theta", positive=True)
    function = getattr(en, function_name)
    expression = function(None, theta, ret_expr=True)

    assert expression.has(theta)
    assert float(expression.subs({temperature: 0, theta: 300})) == pytest.approx(
        function(0, theta=300)
    )
    actual = float(expression.subs({temperature: 300, theta: 300}).evalf())
    assert actual == pytest.approx(function(300, theta=300))
    assert expression.subs({temperature: -1, theta: 300}) is sp.nan
