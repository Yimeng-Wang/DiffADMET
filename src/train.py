import argparse
import contextlib
import io
import gc
import hashlib
import json
import os
import random
from functools import partial
from pathlib import Path

from data_io import ROOT, read_relations, read_table, task_names


def parse_args():
    parser = argparse.ArgumentParser(
        description="DiffADMET training from anonymous-ID graph/label BIN or user-owned CSV"
    )
    parser.add_argument("--mode", choices=["classification", "regression"], required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/reproduction")
    parser.add_argument("--data-format", choices=["auto", "bin", "csv"], default="auto")
    parser.add_argument("--relation-matrix", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument(
        "--epochs", type=int, default=1000, help="Default: 1000, as in the original study"
    )
    parser.add_argument(
        "--batch-size", type=int, default=2048, help="Default: 2048, as in the original study"
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--verify-data-hashes", action="store_true")
    parser.add_argument("--graph-loading", choices=["preload", "lazy"], default="preload")
    parser.add_argument("--graph-cache-shards", type=int, default=4)
    parser.add_argument("--normalizer-policy", choices=["auto", "saved", "fit"], default="auto")
    parser.add_argument(
        "--limit-samples",
        type=int,
        help="Deterministic per-split subset for checks only; omitted for full training",
    )
    args = parser.parse_args()
    if any(fold not in range(5) for fold in args.folds) or len(set(args.folds)) != len(args.folds):
        parser.error("Choose distinct folds from 0, 1, 2, 3, 4")
    if args.num_workers < 0 or args.threads < 1 or args.graph_cache_shards < 1:
        parser.error("Invalid workers, threads, or graph-cache-shards")
    if args.limit_samples is not None and args.limit_samples < 2:
        parser.error("limit-samples must be at least two")
    return args


def set_seed(seed, torch, dgl, np):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dgl.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def sha256(filename):
    digest = hashlib.sha256()
    with Path(filename).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    args = parse_args()
    config_path = args.config or ROOT / "configs" / f"{args.mode}.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["seed"] != 66:
        raise ValueError("The original training random seed is fixed at 66")
    epochs = args.epochs
    batch_size = args.batch_size
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    binary = args.data_format == "bin" or (
        args.data_format == "auto" and (args.data_dir / "manifest.json").is_file()
    )
    policy = args.normalizer_policy
    if policy == "auto":
        policy = "saved" if binary else "fit"
    bundle = None
    store = None
    if binary:
        from binary_data import BinaryTrainingData

        bundle = BinaryTrainingData(args.data_dir)
        relations = bundle.relations
        if args.verify_data_hashes:
            bundle.verify_hashes()
    else:
        if args.relation_matrix is None:
            raise ValueError("CSV mode needs --relation-matrix; BIN mode uses the bundled matrix")
        if policy == "saved":
            raise ValueError("Saved normalization is available for the supplied BIN dataset only")
        relations = read_relations(args.relation_matrix)
        test_path = args.data_dir / "admet135_test3.csv"
        external_path = args.data_dir / "admet135_cancer.csv"
        test = read_table(test_path) if test_path.exists() else None
        external = read_table(external_path) if external_path.exists() else None
    if not args.validate_only:
        import dgl
        import numpy as np
        import torch
        from dgllife.utils import EarlyStopping
        from torch.utils.data import DataLoader
        from data import MoleculeDataset, RegressionNormalizer, collate_molgraphs
        from model_lightmultipredictor import MultiTaskAdaptiveTransferModel_disc

        if args.mode == "classification":
            from train_learning_c import TrainLearner
        else:
            from train_learning_r import TrainLearner
        if binary:
            from binary_data import BinaryGraphStore, BinaryMoleculeDataset

            store = BinaryGraphStore(bundle, args.graph_loading, args.graph_cache_shards)
        device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
        if device == "auto":
            device = "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        torch.set_num_threads(args.threads)
        set_seed(config["seed"], torch, dgl, np)
    for fold in args.folds:
        if binary:
            profiles = {
                "train": bundle.load_profile(
                    f"fold_{fold}/train", args.limit_samples, config["seed"]
                ),
                "valid": bundle.load_profile(
                    f"fold_{fold}/valid", args.limit_samples, config["seed"] + 1
                ),
            }
            for name in ["test", "external"]:
                if name in bundle.profiles:
                    profiles[name] = bundle.load_profile(
                        name, args.limit_samples, config["seed"] + 2
                    )
            print(
                f"Fold {fold} (BIN): "
                + ", ".join(f"{key}={len(value['labels'])}" for key, value in profiles.items())
            )
        else:
            train_path = args.data_dir / f"{fold}admet135_train3.csv"
            valid_path = args.data_dir / f"{fold}admet135_valid3.csv"
            train, valid = read_table(train_path), read_table(valid_path)
            frames = {"train": train, "valid": valid, "test": test, "external": external}
            if args.limit_samples is not None:
                frames = {
                    name: frame.sample(
                        n=min(len(frame), args.limit_samples), random_state=config["seed"]
                    )
                    if frame is not None
                    else None
                    for name, frame in frames.items()
                }
            print(f"Fold {fold} (CSV): train={len(frames['train'])}, valid={len(frames['valid'])}")
        if args.validate_only:
            continue
        output = args.output_dir / args.mode / f"fold_{fold}"
        output.mkdir(parents=True, exist_ok=False)
        normalizer = RegressionNormalizer()
        if binary and policy == "saved":
            normalizer.load_json(bundle.root / bundle.manifest["normalizers"][fold])
        elif binary:
            normalizer.fit(profiles["train"]["labels"], profiles["train"]["masks"])
        else:
            values = frames["train"][task_names()].to_numpy(dtype=np.float64)
            normalizer.fit(values, ~np.isnan(values))
        normalizer.save(output / "normalizer.json")
        if binary:
            if args.graph_loading == "preload":
                store.preload(
                    np.concatenate([value["molecule_ids"] for value in profiles.values()])
                )
            datasets = {
                name: BinaryMoleculeDataset(profile, store) for name, profile in profiles.items()
            }
            inputs = {"dataset_manifest": bundle.root / "manifest.json"}
            for name in profiles:
                key = f"fold_{fold}/{name}" if name in ["train", "valid"] else name
                inputs[name] = bundle.root / bundle.profiles[key]["path"]
            inputs["relations"] = bundle.root / bundle.manifest["relations"]
            if policy == "saved":
                inputs["normalizer"] = bundle.root / bundle.manifest["normalizers"][fold]
        else:
            datasets = {
                name: MoleculeDataset(frame, args.cache_dir)
                for name, frame in frames.items()
                if frame is not None
            }
            inputs = {"train": train_path, "valid": valid_path, "relations": args.relation_matrix}
            if test is not None:
                inputs["test"] = test_path
            if external is not None:
                inputs["external"] = external_path
        loaders = {
            name: DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=name == "train",
                collate_fn=partial(collate_molgraphs, normalizer=normalizer),
                num_workers=args.num_workers,
                pin_memory=device == "cuda",
            )
            for name, dataset in datasets.items()
        }
        model = MultiTaskAdaptiveTransferModel_disc(
            **config["model"],
            num_tasks=135,
            task_correlations=relations.tolist(),
        ).to(device)
        learner = TrainLearner(model, normalizer=normalizer)
        for group in learner.optimizer.param_groups:
            group["lr"] = config["learning_rate"]
        stopper = EarlyStopping(
            mode="higher", patience=config["patience"], filename=str(output / "best.pth")
        )
        run = {
            "config": config,
            "fold": fold,
            "selected_folds": args.folds,
            "model_forward": "fixed dual-graph input; logits only",
            "epochs": epochs,
            "batch_size": batch_size,
            "device": device,
            "data_format": "bin" if binary else "csv",
            "normalizer_policy": policy,
            "graph_loading": args.graph_loading if binary else "generated CSV graphs",
            "num_workers": args.num_workers,
            "threads": args.threads,
            "code_sha256": {
                name: sha256(ROOT / "src" / name)
                for name in [
                    "train.py",
                    "binary_data.py",
                    "data.py",
                    "model_lightmultipredictor.py",
                    "eval_meter.py",
                    f"train_learning_{'c' if args.mode == 'classification' else 'r'}.py",
                ]
            },
            "limit_samples": args.limit_samples,
            "input_sha256": {key: sha256(value) for key, value in inputs.items()},
            "regression_monitor": "squared Pearson correlation, not coefficient of determination",
        }
        (output / "run_manifest.json").write_text(
            json.dumps(run, indent=2) + "\n", encoding="utf-8"
        )
        suffix = "mc" if args.mode == "classification" else "mr"
        train_epoch = getattr(learner, f"run_a_train_epoch_{suffix}")
        valid_epoch = getattr(learner, f"run_a_valid_epoch_{suffix}")
        history = []
        for epoch in range(epochs):
            score = train_epoch(epoch, loaders["train"], epochs)
            val = valid_epoch(epoch, loaders["valid"], epochs)
            monitor = float(val[0])
            if not np.isfinite(monitor):
                raise ValueError(
                    "Validation monitor is not finite; increase the check subset or inspect label coverage"
                )
            learner.scheduler.step(monitor)
            history.append(
                {
                    "epoch": epoch,
                    "train_metric": float(score[0]),
                    "valid_metric": monitor,
                    "train_total_loss": float(score[1]),
                    "valid_total_loss": float(val[1]),
                    "train_objective_loss": float(score[2]),
                    "valid_objective_loss": float(val[2]),
                    "train_per_task_loss": score[-1],
                    "valid_per_task_loss": val[-1],
                }
            )
            with contextlib.redirect_stdout(io.StringIO()):
                early_stop = stopper.step(monitor, model)
            if epoch == 0 or (epoch + 1) % 10 == 0 or early_stop or epoch + 1 == epochs:
                metric = "AUC" if args.mode == "classification" else "R2"
                print(
                    f"Fold {fold} | epoch {epoch + 1}/{epochs} | "
                    f"train {metric}={float(score[0]):.4f} valid {metric}={monitor:.4f}"
                )
            if early_stop:
                break
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        stopper.load_checkpoint(model)
        evaluate = getattr(learner, f"eval_epoch_{suffix}")
        for name, loader in loaders.items():
            evaluate(fold, loader, task_names(), 108, str(output), name)
        del datasets, loaders, model, learner, stopper
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(
        "Input validation passed." if args.validate_only else "Training and evaluation completed."
    )


if __name__ == "__main__":
    main()
