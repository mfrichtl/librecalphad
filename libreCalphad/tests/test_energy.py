from importlib import resources as impresources
import json
import libreCalphad.models.energy as en
import numpy as np
import pandas as pd
import symengine as se


def test_symbolic_gibbs():
    T, a = se.symbols("T a")
    Cp_f = a * T
    gibbs = en._symbolic_gibbs(100, [2], Cp_f, ret_expr=True)
    assert np.isclose(gibbs.subs([T, a], [100, 2]), -10000)
    gibbs = en._symbolic_gibbs(100, [2], Cp_f)
    assert np.isclose(gibbs, -10000)
    gibbs = en._symbolic_gibbs([0, 100], [2], Cp_f)
    assert np.isclose(np.sum(gibbs), -10000)
