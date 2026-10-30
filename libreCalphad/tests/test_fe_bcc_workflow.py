"""Exercise BCC fit persistence without running the downstream assessment."""

from copy import deepcopy
from importlib import resources as impresources
import json
from pathlib import Path
import runpy
import sys
from types import ModuleType

from libreCalphad.models import heat_capacity as hc
from libreCalphad import plotting
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.optimize import OptimizeResult
from tinydb import TinyDB
from tinydb.storages import MemoryStorage


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "databases/LC-steels_validation/FE/FE-fit-bcc.py"
)


class AssessmentReached(Exception):
    """Stop before accessing external reference-state files or databases."""


@pytest.fixture
def bcc_workflow(tmp_path, monkeypatch):
    fe_dir = tmp_path / "FE"
    fe_dir.mkdir()
    script = fe_dir / SCRIPT.name
    script.write_text(SCRIPT.read_text())
    models = {
        "BCC_A2": {
            "einstein": {"theta": [250, "fit"]},
            "melt": {"T_melt": [1811, "fix"]},
        }
    }
    params_file = fe_dir / "FE-params.json"
    params_file.write_text(json.dumps(models))
    source = Path(__file__).parent / "test_heat_capacity_files/test_einstein_data.json"
    dataset = json.loads(source.read_text())
    dataset["phases"] = ["BCC_A2"]
    dataset["reference"] = "synthetic BCC"
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    (datasets / "bcc.json").write_text(json.dumps(dataset))
    other_phase = deepcopy(dataset)
    other_phase["phases"] = ["FCC_A1"]
    (datasets / "fcc.json").write_text(json.dumps(other_phase))
    (tmp_path / "run_param_gen.yaml").write_text(
        "system:\n  datasets: " + json.dumps(str(datasets)) + "\n"
    )

    # Avoid discovering machine-specific ESPEI reference-state plugins on import.
    # Use the same TinyDB query API while keeping dataset inputs local to this test.
    espei = ModuleType("espei")
    dataset_module = ModuleType("espei.datasets")

    def load_datasets(paths):
        database = TinyDB(storage=MemoryStorage)
        for path in paths:
            database.insert(json.loads(Path(path).read_text()))
        return database

    dataset_module.load_datasets = load_datasets
    dataset_module.recursive_glob = lambda path: sorted(Path(path).rglob("*.json"))
    espei.datasets = dataset_module
    monkeypatch.setitem(sys.modules, "espei", espei)
    monkeypatch.setitem(sys.modules, "espei.datasets", dataset_module)

    def resource_files(package):
        if package == "libreCalphad.databases":
            return tmp_path
        raise AssessmentReached(package)

    monkeypatch.setattr(impresources, "files", resource_files)
    # The script imports skopt for a commented-out alternative solver only.
    # Isolate that unused optional dependency rather than requiring it in tests.
    skopt = ModuleType("skopt")
    skopt.gp_minimize = None
    monkeypatch.setitem(sys.modules, "skopt", skopt)
    working_dir = tmp_path / "unrelated-working-directory"
    working_dir.mkdir()
    monkeypatch.chdir(working_dir)
    return script, fe_dir, models, working_dir


@pytest.mark.parametrize("success", [True, False])
@pytest.mark.parametrize("plot_failure", [False, True])
def test_bcc_fit_saves_parameters_and_diagnostics_before_plotting(
    bcc_workflow, monkeypatch, success, plot_failure
):
    script, fe_dir, models, working_dir = bcc_workflow
    original = {"FE": {"BCC_A2": {"old": True}, "FCC_A1": {"keep": 1}}, "AL": {}}
    original_diagnostics = {"FE": {"FCC_A1": {"keep": "diagnostics"}}}
    (fe_dir / "_fitted_params.json").write_text(json.dumps(original))
    (fe_dir / "_fit_diagnostics.json").write_text(json.dumps(original_diagnostics))
    fitted = deepcopy(models["BCC_A2"])
    fitted["einstein"]["theta"][0] = 300
    result = OptimizeResult(
        success=np.bool_(success),
        status=np.int64(0 if success else 2),
        message="converged" if success else "iteration limit",
        x=np.array([300.0]),
        fun=np.float64(0.1),
        nit=np.int64(10),
        nfev=np.int64(20),
    )

    def fit(data, model, verbose):
        assert len(data) == 1
        assert data[0]["phases"] == ["BCC_A2"]
        assert data[0]["components"] == ["FE", "VA"]
        assert data[0]["output"] == "CPM"
        assert model == models["BCC_A2"]
        assert verbose
        return result, fitted

    def plot(*args):
        saved = json.loads((fe_dir / "_fitted_params.json").read_text())
        diagnostics = json.loads((fe_dir / "_fit_diagnostics.json").read_text())
        assert saved["FE"]["BCC_A2"] == fitted
        assert diagnostics["FE"]["BCC_A2"]["success"] is success
        if plot_failure:
            raise RuntimeError("plotting failed")
        return plt.subplots()

    monkeypatch.setattr(hc, "fit_heat_capacity", fit)
    monkeypatch.setattr(plotting, "plot_heat_capacity_from_models", plot)
    expected_error = RuntimeError if plot_failure else AssessmentReached
    with pytest.raises(expected_error):
        runpy.run_path(str(script), run_name="__main__")

    saved = json.loads((fe_dir / "_fitted_params.json").read_text())
    assert saved == {**original, "FE": {**original["FE"], "BCC_A2": fitted}}
    diagnostics = json.loads((fe_dir / "_fit_diagnostics.json").read_text())
    assert diagnostics["FE"]["FCC_A1"] == original_diagnostics["FE"]["FCC_A1"]
    assert diagnostics["FE"]["BCC_A2"] == {
        "success": success,
        "status": 0 if success else 2,
        "message": "converged" if success else "iteration limit",
        "iterations": 10,
        "evaluations": 20,
        "parameters": [300.0],
        "objective": 0.1,
        "objective_name": "relative_residual_norm",
        "objective_units": "dimensionless",
        "dataset_count": 1,
        "references": ["synthetic BCC"],
    }
    assert json.loads((fe_dir / "FE-params.json").read_text()) == models
    assert (fe_dir / "FE-BCC-CPM_fits.png").exists() == (not plot_failure)
    assert list(working_dir.iterdir()) == []


@pytest.mark.parametrize("objective", [np.float64(0), np.float64(np.inf)])
def test_bcc_fixed_fit_creates_output_files(bcc_workflow, monkeypatch, objective):
    script, fe_dir, models, working_dir = bcc_workflow
    monkeypatch.setattr(
        hc,
        "fit_heat_capacity",
        lambda *args, **kwargs: (objective, models["BCC_A2"]),
    )

    def plot(*args):
        raise RuntimeError("stop after saving")

    monkeypatch.setattr(plotting, "plot_heat_capacity_from_models", plot)
    with pytest.raises(RuntimeError, match="stop after saving"):
        runpy.run_path(str(script), run_name="__main__")

    assert json.loads((fe_dir / "_fitted_params.json").read_text()) == {"FE": models}
    diagnostics = json.loads((fe_dir / "_fit_diagnostics.json").read_text())
    diagnostics = diagnostics["FE"]["BCC_A2"]
    assert diagnostics["success"] is bool(np.isfinite(objective))
    assert diagnostics["status"] == (0 if np.isfinite(objective) else 1)
    assert diagnostics["objective"] == objective
    assert diagnostics["parameters"] == []
    assert diagnostics["iterations"] == 0
    assert list(working_dir.iterdir()) == []
