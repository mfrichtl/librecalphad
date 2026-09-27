"""Fit Fe unary heat capacities without updating the thermodynamic database."""

import argparse
import json
from pathlib import Path

from espei.datasets import load_datasets, recursive_glob
from libreCalphad.models.heat_capacity import fit_heat_capacity
from libreCalphad.plotting import plot_heat_capacity_from_models
import matplotlib.pyplot as plt
from tinydb import where


PHASES = ("BCC_A2", "FCC_A1", "LIQUID", "GAS")
PARAMS_FILE = Path(__file__).with_name("FE-params.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", type=Path, required=True, help="ESPEI-datasets directory"
    )
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--params", type=Path, default=PARAMS_FILE)
    args = parser.parse_args(argv)

    if not args.datasets.is_dir():
        parser.error(f"Dataset directory does not exist: {args.datasets}")

    with args.params.open() as f:
        models = json.load(f)[args.phase]

    datasets = load_datasets(recursive_glob(str(args.datasets)))
    components = ["FE", "VA"] if args.phase in ("BCC_A2", "FCC_A1") else ["FE"]
    query = (
        (where("phases") == [args.phase])
        & (where("components") == components)
        & (where("output") == "CPM")
    )
    cpm_data = datasets.search(query)
    if not cpm_data:
        parser.error(
            f"No CPM datasets found for FE {args.phase} ({components}) in {args.datasets}"
        )

    _, fitted_models = fit_heat_capacity(cpm_data, models)
    fig, _ = plot_heat_capacity_from_models(fitted_models, cpm_data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(args.output_dir / f"FE-{args.phase}-CPM.png")
    finally:
        plt.close(fig)
    with (args.output_dir / f"FE-{args.phase}-CPM-params.json").open("w") as f:
        json.dump({args.phase: fitted_models}, f, indent=4)


if __name__ == "__main__":
    main()
