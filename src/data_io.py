import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def task_names():
    schema = json.loads((ROOT / "configs/task_schema.json").read_text(encoding="utf-8"))
    return schema["classification_tasks"] + schema["regression_tasks"]


def read_table(filename):
    filename = Path(filename)
    with filename.open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle))
    if len(header) != len(set(header)):
        raise ValueError(f"{filename.name}: duplicate column names")
    tasks = task_names()
    if set(header) != set(tasks + ["SMILES"]):
        raise ValueError(
            f"{filename.name}: expected SMILES and the 135 tasks in configs/task_schema.json"
        )
    if [name for name in header if name != "SMILES"] != tasks:
        raise ValueError(f"{filename.name}: task order does not match task_schema.json")
    frame = pd.read_csv(filename, encoding="utf-8-sig", dtype={"SMILES": str})
    if frame.empty or frame.SMILES.isna().any() or frame.SMILES.str.strip().eq("").any():
        raise ValueError(f"{filename.name}: empty table or missing SMILES")
    frame[tasks] = frame[tasks].apply(pd.to_numeric, errors="raise")
    values = frame[tasks].to_numpy(dtype=np.float64)
    if np.isinf(values).any():
        raise ValueError(f"{filename.name}: infinite labels are not supported")
    observed = values[:, :108][~np.isnan(values[:, :108])]
    if not np.isin(observed, [0, 1]).all():
        raise ValueError(f"{filename.name}: classification labels must be 0, 1, or blank")
    return frame[tasks + ["SMILES"]]


def read_relations(filename):
    tasks = task_names()
    frame = pd.read_csv(filename, encoding="utf-8-sig")
    if frame.shape == (135, 136):
        if frame.iloc[:, 0].tolist() != tasks:
            raise ValueError("Relation matrix row labels must match task_schema.json")
        frame = frame.iloc[:, 1:]
    if frame.shape != (135, 135) or frame.columns.tolist() != tasks:
        raise ValueError("Expected a 135 x 135 relation CSV with task names as headers")
    values = frame.to_numpy(dtype=np.float32)
    if not np.isfinite(values).all() or not np.isin(values, [0, 1]).all():
        raise ValueError("Relation matrix must contain finite binary values")
    if not (values.sum(axis=0) > 0).all() or not (values.sum(axis=1) > 0).all():
        raise ValueError("Relation matrix needs at least one allowed edge per row and column")
    return values
