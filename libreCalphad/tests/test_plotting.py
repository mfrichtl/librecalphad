from espei.datasets import load_datasets, recursive_glob
from importlib import resources as impresources
from libreCalphad import plotting as lcplt
import matplotlib as mp
from pycalphad import Database, variables as v
from tinydb import where


def test_plot_heat_capacity_from_symbolic():
    model_dict = {
        "symbolic": {
            "expression": "a + b*T",
            "symbols": ["a", "b"],
            "a": [5000, "fix"],
            "b": [20, "fix"],
            "variable_values": [5000, 20],
        }
    }
    dataset_file = (
        impresources.files("libreCalphad.tests.test_datasets")
        / "FE-CPM-BCC_A2-Kollie1969.json"
    )
    datasets = load_datasets([str(dataset_file)])
    query = (
        (where("phases") == ["BCC_A2"])
        & (where("components") == ["FE", "VA"])
        & (where("output") == "CPM")
    )
    search_results = datasets.search(query)
    fig, ax = lcplt.plot_heat_capacity_from_models(model_dict, search_results)
    assert isinstance(fig, mp.figure.Figure)
    assert isinstance(ax, mp.axes._axes.Axes)


def test_plot_calculate_heat_capacity():
    db_file = (
        impresources.files("libreCalphad.tests.test_database_files")
        / "LC-steels-input.xml"
    )
    dbf = Database(db_file)
    components = ["FE", "VA"]
    phases = ["BCC_A2"]
    conditions = {v.N: 1, v.T: (0, 2000, 10), v.P: 101325}
    fig, ax = lcplt.plot_calculated_heat_capacity(dbf, components, phases, conditions)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Isobaric Heat Capacity (J/K/mol-formula)")
    assert isinstance(fig, mp.figure.Figure)
    assert isinstance(ax, mp.axes._axes.Axes)


def test_plot_calculate_heat_capacity_with_dataset_query():
    dataset_file = (
        impresources.files("libreCalphad.tests.test_datasets")
        / "FE-CPM-BCC_A2-Kollie1969.json"
    )
    datasets = load_datasets([str(dataset_file)])
    query = (
        (where("phases") == ["BCC_A2"])
        & (where("components") == ["FE", "VA"])
        & (where("output") == "CPM")
    )
    search_results = datasets.search(query)
    db_file = (
        impresources.files("libreCalphad.tests.test_database_files")
        / "LC-steels-input.xml"
    )
    dbf = Database(db_file)
    components = ["FE", "VA"]
    phases = ["BCC_A2"]
    conditions = {v.N: 1, v.T: (0, 2000, 10), v.P: 101325}
    fig, ax = lcplt.plot_calculated_heat_capacity(
        dbf, components, phases, conditions, datasets=search_results
    )
    assert isinstance(fig, mp.figure.Figure)
    assert isinstance(ax, mp.axes._axes.Axes)


def test_plot_calculate_heat_capacity_with_datasets():
    dataset_file = (
        impresources.files("libreCalphad.tests.test_datasets")
        / "FE-CPM-BCC_A2-Kollie1969.json"
    )
    datasets = load_datasets([str(dataset_file)])
    db_file = (
        impresources.files("libreCalphad.tests.test_database_files")
        / "LC-steels-input.xml"
    )
    dbf = Database(db_file)
    components = ["FE", "VA"]
    phases = ["BCC_A2"]
    conditions = {v.N: 1, v.T: (0, 2000, 10), v.P: 101325}
    fig, ax = lcplt.plot_calculated_heat_capacity(
        dbf, components, phases, conditions, datasets=datasets
    )
    assert isinstance(fig, mp.figure.Figure)
    assert isinstance(ax, mp.axes._axes.Axes)


def test_plot_calculate_gibbs_energies():
    db_file = (
        impresources.files("libreCalphad.tests.test_database_files")
        / "LC-steels-input.xml"
    )
    dbf = Database(db_file)
    components = ["FE", "VA"]
    phases = ["BCC_A2"]
    conditions = {v.N: 1, v.T: (0, 2000, 10), v.P: 101325}
    fig, ax = lcplt.plot_calculated_gibbs_energies(dbf, components, phases, conditions)
    assert isinstance(fig, mp.figure.Figure)
    assert isinstance(ax, mp.axes._axes.Axes)


def test_plot_calculate_gibbs_energies_with_datasets():
    dataset_file = (
        impresources.files("libreCalphad.tests.test_datasets")
        / "FE-CPM-BCC_A2-Kollie1969.json"
    )
    datasets = load_datasets([str(dataset_file)])
    query = (
        (where("phases") == ["BCC_A2"])
        & (where("components") == ["FE", "VA"])
        & (where("output") == "CPM")
    )
    search_results = datasets.search(query)
    db_file = (
        impresources.files("libreCalphad.tests.test_database_files")
        / "LC-steels-input.xml"
    )
    dbf = Database(db_file)
    components = ["FE", "VA"]
    phases = ["BCC_A2"]
    conditions = {v.N: 1, v.T: (0, 2000, 10), v.P: 101325}
    fig, ax = lcplt.plot_calculated_gibbs_energies(
        dbf, components, phases, conditions, datasets=search_results
    )
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Isobaric Heat Capacity (J/K/mol-formula)")
    assert isinstance(fig, mp.figure.Figure)
    assert isinstance(ax, mp.axes._axes.Axes)


def test_plot_calculate_gibbs_energies_with_transition_temperature_printing():
    db_file = (
        impresources.files("libreCalphad.tests.test_database_files")
        / "LC-steels-input.xml"
    )
    dbf = Database(db_file)
    components = ["FE", "VA"]
    phases = ["BCC_A2"]
    conditions = {v.N: 1, v.T: (0, 2000, 10), v.P: 101325}
    fig, ax = lcplt.plot_calculated_gibbs_energies(
        dbf, components, phases, conditions, print_transition_temps=True
    )
    assert isinstance(fig, mp.figure.Figure)
    assert isinstance(ax, mp.axes._axes.Axes)


def test_plot_delta_energies():
    db_file = (
        impresources.files("libreCalphad.tests.test_database_files")
        / "LC-steels-input.xml"
    )
    dbf = Database(db_file)
    dataset_folder = impresources.files("libreCalphad.tests.test_datasets")
    datasets = load_datasets(recursive_glob(str(dataset_folder)))
    components = ["FE", "VA"]
    phases = ["BCC_A2", "FCC_A1"]
    conditions = {v.N: 1, v.T: (0, 2000, 10), v.P: 101325}
    fig, ax = lcplt.plot_delta_energies(
        dbf, components, phases, conditions, datasets, output="GM"
    )
    assert isinstance(fig, mp.figure.Figure)
    assert isinstance(ax, mp.axes._axes.Axes)


def test_plot_delta_energies_with_error():
    db_file = (
        impresources.files("libreCalphad.tests.test_database_files")
        / "LC-steels-input.xml"
    )
    dbf = Database(db_file)
    dataset_folder = impresources.files("libreCalphad.tests.test_datasets")
    datasets = load_datasets(recursive_glob(str(dataset_folder)))
    components = ["FE", "VA"]
    phases = ["BCC_A2", "FCC_A1"]
    conditions = {v.N: 1, v.T: (0, 2000, 10), v.P: 101325}
    fig, ax = lcplt.plot_delta_energies(
        dbf, components, phases, conditions, datasets, output="GM", error=True
    )
    assert isinstance(fig, mp.figure.Figure)
    assert isinstance(ax, mp.axes._axes.Axes)
