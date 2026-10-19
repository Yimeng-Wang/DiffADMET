import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_io import task_names


def main():
    parser = argparse.ArgumentParser(
        description="Generate artificial labels on small illustrative molecules; no project data are read"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/demo"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(66)
    tasks = task_names()
    molecules = [
        "CC",
        "CCC",
        "CCCC",
        "CCCCC",
        "CCCCCC",
        "CO",
        "CCO",
        "CCCO",
        "CCCCO",
        "CN",
        "CCN",
        "CCCN",
        "CCOC",
        "CCOCC",
        "CC(=O)O",
        "CC(=O)N",
        "c1ccccc1",
        "Cc1ccccc1",
        "Oc1ccccc1",
        "c1ccncc1",
        "C1CCCCC1",
        "C1CCCC1",
        "CC(C)O",
        "CC(C)N",
    ]
    groups = {"train3": molecules[:12], "valid3": molecules[12:20], "test3": molecules[20:]}
    for split, smiles in groups.items():
        values = rng.normal(size=(len(smiles), 135))
        values[:, :108] = (np.arange(len(smiles))[:, None] + np.arange(108)[None, :]) % 2
        values[0, 5] = np.nan
        values[1, 112] = np.nan
        frame = pd.DataFrame(values, columns=tasks)
        frame["SMILES"] = smiles
        if split == "test3":
            frame.to_csv(args.output_dir / "admet135_test3.csv", index=False)
            frame.to_csv(args.output_dir / "admet135_cancer.csv", index=False)
        else:
            for fold in range(5):
                frame.to_csv(args.output_dir / f"{fold}admet135_{split}.csv", index=False)
    pd.DataFrame(np.eye(135, dtype=int), columns=tasks).to_csv(
        args.output_dir / "demo_relations.csv", index=False
    )
    print(
        "Created artificial demonstration CSVs and identity relations. These are not research data or the paper relation matrix."
    )


if __name__ == "__main__":
    main()
