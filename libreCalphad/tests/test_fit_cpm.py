import json
from pathlib import Path
import subprocess
import sys


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "databases/LC-steels_validation/FE/fit_cpm.py"
)


def test_fit_fe_cpm_writes_results(tmp_path):
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    source = Path(__file__).parent / "test_heat_capacity_files/test_einstein_data.json"
    (datasets / "einstein.json").write_text(source.read_text())
    params = tmp_path / "params.json"
    params.write_text(json.dumps({"FCC_A1": {"einstein": {"theta": [250, "fit"]}}}))
    output = tmp_path / "results"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--datasets",
            str(datasets),
            "--phase",
            "FCC_A1",
            "--params",
            str(params),
            "--output-dir",
            str(output),
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    fitted = json.loads((output / "FE-FCC_A1-CPM-params.json").read_text())
    assert abs(fitted["FCC_A1"]["einstein"]["theta"][0] - 300) < 1
    assert (output / "FE-FCC_A1-CPM.png").is_file()
    assert json.loads(params.read_text())["FCC_A1"]["einstein"]["theta"] == [
        250,
        "fit",
    ]


def test_fit_fe_cpm_requires_phase_data(tmp_path):
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    params = tmp_path / "params.json"
    params.write_text(json.dumps({"FCC_A1": {"einstein": {"theta": [250, "fit"]}}}))
    output = tmp_path / "results"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--datasets",
            str(datasets),
            "--phase",
            "FCC_A1",
            "--params",
            str(params),
            "--output-dir",
            str(output),
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert result.returncode != 0
    assert "No CPM datasets found" in result.stderr
    assert not output.exists()
