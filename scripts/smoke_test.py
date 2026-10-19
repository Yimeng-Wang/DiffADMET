import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    env = dict(os.environ, DGLBACKEND="pytorch", PYTHONHASHSEED="66", PYTHONIOENCODING="utf-8")
    with tempfile.TemporaryDirectory(prefix="diffadmet_smoke_") as temporary:
        work = Path(temporary)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/make_binary_demo.py"),
                "--output-dir",
                str(work / "bin"),
            ],
            check=True,
            env=env,
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/validate_binary_dataset.py"),
                "--data-dir",
                str(work / "bin"),
                "--verify-hashes",
                "--deep",
            ],
            check=True,
            env=env,
        )
        for mode in ["classification", "regression"]:
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "src/train.py"),
                    "--mode",
                    mode,
                    "--data-dir",
                    str(work / "bin"),
                    "--data-format",
                    "bin",
                    "--folds",
                    "0",
                    "--epochs",
                    "1",
                    "--batch-size",
                    "4",
                    "--device",
                    "cpu",
                    "--num-workers",
                    "0",
                    "--threads",
                    "2",
                    "--output-dir",
                    str(work / "outputs"),
                ],
                check=True,
                env=env,
            )
            fold = work / "outputs" / mode / "fold_0"
            suffix = "evalue_c" if mode == "classification" else "evalue_r_denorm"
            expected = {f"0{split}_{suffix}.csv" for split in ["train", "valid", "test", "external"]}
            core = {"best.pth", "normalizer.json", "run_manifest.json", "history.json"}
            assert {path.name for path in fold.iterdir()} == expected | core
            assert (fold / "best.pth").stat().st_size > 0
            history = json.loads((fold / "history.json").read_text(encoding="utf-8"))
            assert len(history) == 1
            assert len(history[0]["train_per_task_loss"]) == 135
            assert len(history[0]["valid_per_task_loss"]) == 135
            run = json.loads((fold / "run_manifest.json").read_text(encoding="utf-8"))
            assert run["config"]["seed"] == 66 and run["fold"] == 0
            assert run["epochs"] == 1 and run["batch_size"] == 4
            assert run["selected_folds"] == [0] and run["normalizer_policy"] == "saved"
            assert set(run["input_sha256"]) == {
                "dataset_manifest",
                "train",
                "valid",
                "test",
                "external",
                "relations",
                "normalizer",
            }
            assert len(run["code_sha256"]) == 6
            assert "versions" not in run
            normalizer = json.loads((fold / "normalizer.json").read_text(encoding="utf-8"))
            assert (
                normalizer["fitted"] and len(normalizer["means"]) == len(normalizer["stds"]) == 27
            )
            import pandas as pd

            for name in expected:
                table = pd.read_csv(fold / name)
                assert len(table) == (108 if mode == "classification" else 27)
                assert table.columns[0] == "Task"
        print(
            "PASS: classification/regression training without library-version queries; retained training artifacts and endpoint-metric CSVs only."
        )


if __name__ == "__main__":
    main()
