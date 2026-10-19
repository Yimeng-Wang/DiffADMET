import argparse
import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {
    ".gitattributes",
    ".gitignore",
    "README.md",
    "NOTICE.md",
    "environment.yml",
    "pyproject.toml",
    "LICENSES/Apache-2.0.txt",
    "configs/classification.json",
    "configs/regression.json",
    "configs/task_schema.json",
    "configs/Type.csv",
    "configs/reproduction_manifest.json",
    "docs/DATA_FORMAT.md",
    "docs/BIN_DATA_FORMAT.md",
    "docs/DATA_AVAILABILITY.md",
    "docs/REPRODUCIBILITY.md",
    "docs/cleanup_manifest.json",
    "docs/VALIDATION.md",
    "src/compound_constants.py",
    "src/compound_tools.py",
    "src/data.py",
    "src/binary_data.py",
    "src/data_io.py",
    "src/eval_meter.py",
    "src/featurizers.py",
    "src/model_lightmultipredictor.py",
    "src/mol_to_2d3dgraph.py",
    "src/train.py",
    "src/train_learning_c.py",
    "src/train_learning_r.py",
    "scripts/check_release.py",
    "scripts/build_release.py",
    "scripts/make_demo_data.py",
    "scripts/smoke_test.py",
    "scripts/make_binary_demo.py",
    "scripts/validate_binary_dataset.py",
    ".github/workflows/validate.yml",
}
LOCAL_DIRECTORIES = {
    "data",
    "cache",
    "outputs",
    "private",
    ".venv",
    "venv",
    ".git",
    ".ruff_cache",
    "__pycache__",
}
LOCAL_FILENAMES = {
    "release_manifest.json",
    "DiffADMET-training-source.zip",
    "data_release_archives.json",
}


def is_data_archive(name):
    return bool(re.fullmatch(r"DiffADMET-reproduction-data(?:-part[0-9]{2})?\.zip", name))


def publication_files():
    missing = sorted(name for name in ALLOWED if not (ROOT / name).is_file())
    if missing:
        raise ValueError(f"Missing publication files: {missing}")
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if any(part in LOCAL_DIRECTORIES for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"Symlinks are not allowed: {relative.as_posix()}")
        if (
            path.is_file()
            and relative.as_posix() not in ALLOWED | LOCAL_FILENAMES
            and not is_data_archive(relative.as_posix())
        ):
            raise ValueError(
                f"Unexpected file outside publication allowlist: {relative.as_posix()}"
            )
    for name in sorted(ALLOWED):
        path = ROOT / name
        text = path.read_text(encoding="utf-8-sig")
        if re.search(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U000323af]", name + text):
            raise ValueError(f"Non-English CJK text found in publication file: {name}")
        if name.endswith(".py"):
            compile(ast.parse(text, filename=name), name, "exec")
        if re.search(r"(?i)(?:[A-Z]:[\\/](?:Users|ADMET)|/home/[a-z0-9_-]+/)", text):
            raise ValueError(f"Author-machine absolute path found in {name}")
        if re.search(
            r"(?:gh[pousr]_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{32,}|AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)",
            text,
        ):
            raise ValueError(f"Potential credential found in {name}")
    try:
        probe = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode == 0 and Path(probe.stdout.strip()).resolve() == ROOT.resolve():
            tracked = (
                subprocess.check_output(["git", "-C", str(ROOT), "ls-files", "-z"])
                .decode("utf-8")
                .split("\0")
            )
            unexpected = sorted(
                name for name in tracked if name and name not in ALLOWED | {"release_manifest.json"}
            )
            if unexpected:
                raise ValueError(f"Disallowed files are already tracked by Git: {unexpected}")
    except FileNotFoundError:
        pass
    return sorted(ALLOWED)


def dataset_files(verify_hashes=True):
    directory = ROOT / "data/reproduction"
    metadata_path = directory / "manifest.json"
    manifest = json.loads(metadata_path.read_text(encoding="utf-8"))
    public = json.loads((ROOT / "configs/reproduction_manifest.json").read_text(encoding="utf-8"))
    if manifest != public:
        raise ValueError("Public data inventory and bundle manifest differ")
    expected = {item["path"] for item in manifest["files"]} | {"manifest.json"}
    actual = {
        path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()
    }
    if actual != expected:
        raise ValueError("Unexpected or missing files in the real dataset directory")
    for item in manifest["files"]:
        rel = Path(item["path"])
        if rel.is_absolute() or ".." in rel.parts or rel.suffix not in [".bin", ".json"]:
            raise ValueError("Disallowed data artifact path")
        path = directory / rel
        if path.is_symlink() or path.stat().st_size != item["bytes"]:
            raise ValueError(f"Data artifact size/path mismatch: {rel}")
        if verify_hashes:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != item["sha256"]:
                raise ValueError(f"Data artifact checksum mismatch: {rel}")
    return directory, sorted(expected), manifest


def main():
    parser = argparse.ArgumentParser(
        description="Verify source allowlist and optional authorized real BIN data inventory"
    )
    parser.add_argument("--with-data", action="store_true")
    args = parser.parse_args()
    files = publication_files()
    print(
        f"PASS: {len(files)} allowed text files; syntax, source paths, credential patterns, and Git-tracked content checked."
    )

    if args.with_data:
        _, data, manifest = dataset_files()
        print(
            f"PASS: {len(data)} approved dataset files, {manifest['molecules']} molecules; no CSV/pickle/model-weight files."
        )


if __name__ == "__main__":
    main()
