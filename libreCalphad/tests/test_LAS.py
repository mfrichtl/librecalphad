from calphadOpenDB import databases
from importlib_resources import files
from numpy import allclose
from pandas import read_pickle
from pycalphad import Database, equilibrium


def test_C_FE():
    """
    Test the C-Fe system from Gustafson (1985).
    """


def test_C_CR():
    """
    Test the C-Cr system from Andersson (1987).
    """

    

def test_unary_energies():
    unary_energies = read_pickle('./calphadOpenDB/tests/unary.pkl')
    db = Database('./calphadOpenDB/databases/LAS.tdb')
    phases = list(db.phases.keys())
    components = list(db.elements)

    for phase in phases:
        for component in components:
            if component == '/-' or component == 'VA':
                continue
            for compset in db.phases[phase].constituents:
                if Species(component) in compset:
                    comps = [component, 'VA']
                    calc = calculate(db, comps, phase, T=1000, P=101325, N=1)
                    this_energy = calc.GM.squeeze().values
                    unary_energy = unary_energies.query("phase == @phase & constituent == @component")['GM'].values[0]
                    print(this_energy, unary_energy)
                    assert allclose(this_energy, unary_energy), f"Problem with {phase}, {component}"