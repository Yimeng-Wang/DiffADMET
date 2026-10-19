import argparse
import hashlib
import json
import zipfile

from check_release import ROOT, dataset_files, publication_files


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_archive(path, expected):
    with zipfile.ZipFile(path) as archive:
        if set(archive.namelist()) != set(expected):
            raise ValueError("Archive membership mismatch")
        for name, digest in expected.items():
            value = hashlib.sha256()
            with archive.open(name) as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    value.update(block)
            if value.hexdigest() != digest:
                raise ValueError(f"Archive checksum mismatch: {name}")


def main():
    parser = argparse.ArgumentParser(
        description="Build separate source and authorized no-SMILES real-data ZIP assets"
    )
    parser.add_argument("--with-data", action="store_true")
    parser.add_argument(
        "--data-part-mib",
        type=int,
        default=1500,
        help="Maximum uncompressed payload per independent data ZIP",
    )
    args = parser.parse_args()
    if args.data_part_mib < 100:
        parser.error("data-part-mib must be at least 100")
    files = publication_files()
    records = [
        {"path": name, "bytes": (ROOT / name).stat().st_size, "sha256": file_digest(ROOT / name)}
        for name in files
    ]
    manifest = {
        "project": "DiffADMET",
        "distribution": "training source with separately published real graph/label BIN assets",
        "files": records,
        "excluded": "original CSVs, plaintext SMILES mapping, pretrained weights, legacy pickle containers, per-run outputs and service code",
    }
    (ROOT / "release_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    source_path = ROOT / "DiffADMET-training-source.zip"
    with zipfile.ZipFile(source_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in files + ["release_manifest.json"]:
            archive.write(ROOT / name, f"DiffADMET/{name}")
    expected = {f"DiffADMET/{item['path']}": item["sha256"] for item in records}
    expected["DiffADMET/release_manifest.json"] = file_digest(ROOT / "release_manifest.json")
    verify_archive(source_path, expected)
    print(f"Created and verified source ZIP: {len(expected)} files.")
    if not args.with_data:
        return
    directory, names, data_manifest = dataset_files()
    file_hashes = {item["path"]: item["sha256"] for item in data_manifest["files"]}
    file_hashes["manifest.json"] = file_digest(directory / "manifest.json")
    budget = args.data_part_mib * 1024 * 1024
    groups = []
    group = []
    size = 0
    # Put the manifest/auxiliary data in the first part, then graph shards.
    names = sorted(names, key=lambda name: (name.startswith("graphs/"), name))
    for name in names:
        value = (directory / name).stat().st_size
        if group and size + value > budget:
            groups.append(group)
            group, size = [], 0
        group.append(name)
        size += value
    if group:
        groups.append(group)
    archives = []
    for i, group in enumerate(groups):
        suffix = "" if len(groups) == 1 else f"-part{i + 1:02d}"
        path = ROOT / f"DiffADMET-reproduction-data{suffix}.zip"
        with zipfile.ZipFile(
            path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for name in group:
                archive.write(directory / name, "data/reproduction/" + name)
        expected = {"data/reproduction/" + name: file_hashes[name] for name in group}
        verify_archive(path, expected)
        archives.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": file_digest(path),
                "members": len(group),
            }
        )
        print(
            f"Created and verified data asset {path.name}: {len(group)} files, {path.stat().st_size} bytes.",
            flush=True,
        )
    (ROOT / "data_release_archives.json").write_text(
        json.dumps(
            {
                "extract_all_parts": True,
                "dataset_manifest_sha256": file_hashes["manifest.json"],
                "archives": archives,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
