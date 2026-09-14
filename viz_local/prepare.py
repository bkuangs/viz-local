"""Prepare the official Chess RGB/pose subset without interpreting query poses."""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import re
import shutil
import struct
import tempfile
from time import perf_counter
import zipfile
import zlib

from .io import fingerprint, read_json, sha256_file, write_json


ARCHIVE_BYTES = 3079608937
ROLES = ("reference", "confidence_fit", "confidence_calibration", "final_evaluation")
SEQUENCE = re.compile(r"seq-[0-9]{2}")
ARCHIVE_MEMBER = re.compile(r"chess/(seq-[0-9]{2})\.zip")
FRAME_MEMBER = re.compile(r"(seq-[0-9]{2})/frame-([0-9]{6})\.(color\.png|pose\.txt|depth\.png)")


def official_sequences(contents: bytes) -> list[str]:
    sequences = []
    for line in contents.decode("utf-8-sig").splitlines():
        match = re.fullmatch(r"sequence([1-9][0-9]?)", line.strip())
        if match is None:
            raise ValueError(f"Invalid official split line: {line!r}")
        sequences.append(f"seq-{int(match[1]):02d}")
    if not sequences or len(sequences) != len(set(sequences)):
        raise ValueError("Official split is empty or contains duplicate sequences")
    return sorted(sequences)


def validate_roles(roles: dict, train: list[str], test: list[str],
                   actual: list[str]) -> dict[str, str]:
    if set(train) & set(test):
        raise ValueError("Official train/test splits overlap")
    if set(train) | set(test) != set(actual):
        raise ValueError("Official splits do not account for all actual sequences")
    if not isinstance(roles, dict) or set(roles) != set(ROLES):
        raise ValueError(f"Exactly these roles are required: {ROLES}")
    assignments = {}
    for role in ROLES:
        settings = roles[role]
        if not isinstance(settings, dict) or set(settings) != {"sequences", "stride"}:
            raise ValueError(f"Invalid settings for role {role}")
        sequences, stride = settings["sequences"], settings["stride"]
        if type(stride) is not int or stride < 1:
            raise ValueError(f"Stride must be a positive integer for {role}")
        if not isinstance(sequences, list) or not sequences:
            raise ValueError(f"Nonempty sequence list required for {role}")
        allowed = test if role == "final_evaluation" else train
        for sequence in sequences:
            if not isinstance(sequence, str) or not SEQUENCE.fullmatch(sequence):
                raise ValueError(f"Invalid sequence ID: {sequence!r}")
            if sequence in assignments:
                raise ValueError(f"Role sequences overlap: {sequence}")
            if sequence not in allowed:
                raise ValueError(f"Official split leakage: {sequence} in {role}")
            assignments[sequence] = role
    if set(assignments) != set(actual):
        raise ValueError("Role assignments must cover all actual sequences")
    return assignments


def members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    infos = archive.infolist()
    result = {info.filename: info for info in infos}
    if len(result) != len(infos):
        raise ValueError("Duplicate archive member names")
    return result


@contextmanager
def nested_archive(outer: zipfile.ZipFile, info: zipfile.ZipInfo, scratch: Path):
    # ZipFile seeks repeatedly; do not make it seek inside a compressed ZipExtFile.
    with tempfile.NamedTemporaryFile(prefix=".prepare-sequence-", suffix=".zip",
                                     dir=scratch) as temporary:
        with outer.open(info) as source:
            shutil.copyfileobj(source, temporary)
        temporary.flush()
        with zipfile.ZipFile(temporary.name) as nested:
            yield nested


def safe_destination(root: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe destination: {name}")
    destination = root
    for part in relative.parts:
        destination = destination / part
        if destination.is_symlink():
            raise ValueError(f"Refusing symlink destination: {destination}")
    return destination


def validate_file(path: Path, info: zipfile.ZipInfo) -> None:
    if not path.is_file() or path.stat().st_size != info.file_size:
        raise ValueError(f"Asset size/type mismatch: {path}")
    crc = 0
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            crc = zlib.crc32(block, crc)
    if crc != info.CRC:
        raise ValueError(f"Asset CRC32 mismatch: {path}")


def extract_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo,
                   root: Path, relative: str) -> Path:
    path = safe_destination(root, relative)
    if path.exists():
        validate_file(path, info)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".prepare-asset-", dir=path.parent,
                                     delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            with archive.open(info) as source:
                shutil.copyfileobj(source, temporary)
            temporary.flush()
            validate_file(temporary_path, info)
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return path


def validate_png_header(path: Path) -> None:
    # Metadata only: never decode pixels, including final-evaluation RGB images.
    with path.open("rb") as source:
        header = source.read(33)
    if (len(header) != 33 or header[:8] != b"\x89PNG\r\n\x1a\n"
            or header[8:16] != b"\x00\x00\x00\x0dIHDR"):
        raise ValueError(f"Invalid PNG IHDR: {path}")
    width, height = struct.unpack(">II", header[16:24])
    if (width, height) != (640, 480):
        raise ValueError(f"Expected native 640x480 RGB dimensions: {path}")
    if zlib.crc32(header[12:29]) != struct.unpack(">I", header[29:33])[0]:
        raise ValueError(f"Invalid PNG IHDR checksum: {path}")


def sequence_frames(archive: zipfile.ZipFile, sequence: str) -> list[tuple]:
    frames = {}
    for name, info in members(archive).items():
        if name == sequence + "/" and info.is_dir():
            continue
        match = FRAME_MEMBER.fullmatch(name)
        if match is None or match[1] != sequence:
            raise ValueError(f"Unexpected sequence archive member: {name}")
        index, kind = int(match[2]), match[3]
        frames.setdefault(index, {})[kind] = info
    if not frames:
        raise ValueError(f"Empty sequence archive: {sequence}")
    result = []
    for index, assets in sorted(frames.items()):
        if "color.png" not in assets or "pose.txt" not in assets:
            raise ValueError(f"Missing image/pose pair: {sequence}/frame-{index:06d}")
        if not assets["color.png"].file_size or not assets["pose.txt"].file_size:
            raise ValueError(f"Empty image/pose asset: {sequence}/frame-{index:06d}")
        result.append((index, assets["color.png"], assets["pose.txt"]))
    return result


def prepare(config_path: Path) -> dict:
    config_path = Path(config_path)
    config = read_json(config_path)
    required = {"scene", "seed", "archive_path", "data_root", "manifest_path",
                "dataset_url", "roles"}
    if missing := required - set(config):
        raise ValueError(f"Missing preparation configuration fields: {sorted(missing)}")
    if config["scene"] != "chess" or type(config["seed"]) is not int:
        raise ValueError("Preparation requires scene='chess' and an integer seed")
    for key in ("archive_path", "data_root", "manifest_path", "dataset_url"):
        if not isinstance(config[key], str) or not config[key]:
            raise ValueError(f"Nonempty string required for {key}")
    archive_path = config_path.parent / config["archive_path"]
    root = config_path.parent / config["data_root"]
    manifest_path = config_path.parent / config["manifest_path"]
    expected_size = config.get("archive_size_bytes", ARCHIVE_BYTES)
    if type(expected_size) is not int or expected_size <= 0:
        raise ValueError("archive_size_bytes must be a positive integer")
    # Never open an in-progress official download as a ZIP.
    if archive_path.stat().st_size != expected_size:
        raise ValueError(f"Archive is incomplete or wrong size; expected {expected_size} bytes")
    if not zipfile.is_zipfile(archive_path):
        raise ValueError(f"Not a complete ZIP archive: {archive_path}")
    archive_hash = sha256_file(archive_path)
    if "archive_sha256" in config and config["archive_sha256"] != archive_hash:
        raise ValueError("Archive SHA256 mismatch")
    if root.is_symlink() or manifest_path.is_symlink():
        raise ValueError("Data root and manifest must not be symlinks")

    with zipfile.ZipFile(archive_path) as outer:
        outer_members = members(outer)
        sequences = {}
        for name, info in outer_members.items():
            match = ARCHIVE_MEMBER.fullmatch(name)
            if match:
                sequences[match[1]] = info
            elif name not in {"chess/", "chess/chess.png", "chess/TrainSplit.txt",
                              "chess/TestSplit.txt"}:
                raise ValueError(f"Unexpected Chess archive member: {name}")
        split_contents = {}
        for name in ("TrainSplit.txt", "TestSplit.txt"):
            member = outer_members.get("chess/" + name)
            if member is None:
                raise ValueError(f"Missing official split: {name}")
            split_contents[name] = outer.read(member)
        official = {
            "train": official_sequences(split_contents["TrainSplit.txt"]),
            "test": official_sequences(split_contents["TestSplit.txt"]),
        }
        assignments = validate_roles(config["roles"], official["train"], official["test"],
                                     sorted(sequences))
        roles = {role: {"sequences": sorted(config["roles"][role]["sequences"]),
                        "stride": config["roles"][role]["stride"]} for role in ROLES}
        split = {
            "scene": "chess",
            "official_sequences": official,
            "role_sequences": {role: settings["sequences"] for role, settings in roles.items()},
            "assignment_unit": "whole original sequence before subsampling",
            "variant_policy": "All variants inherit the original image_id's role.",
        }
        identity = {
            "schema_version": 1,
            "scene": "chess",
            "seed": config["seed"],
            "archive_path": config["archive_path"],
            "archive_sha256": archive_hash,
            "archive_size_bytes": expected_size,
            "dataset_url": config["dataset_url"],
            "data_root": config["data_root"],
            "manifest_path": config["manifest_path"],
            "roles": roles,
            "split_id": fingerprint(split),
            "selection_rule": "frame_index % role.stride == 0",
            "image_dimensions": {
                "width": 640, "height": 480,
                "policy": "native RGB; no resizing",
                "verification": "Selected PNG IHDR metadata only; unselected dimensions declared.",
            },
            "pose_convention": "camera_to_world",
            "pose_units": "meters",
            "pose_provenance": "original 7-Scenes pose TXT; copied as opaque bytes, no values read",
            "intrinsics": {
                "status": "pending_reference_only_probe",
                "provenance": "RGB uncalibrated; depth intrinsics are not assumed valid for RGB.",
            },
        }
        preparation_id = fingerprint(identity)
        marker = {"preparation_id": preparation_id, "identity": identity}
        marker_path = safe_destination(root, "preparation.json")
        if marker_path.exists():
            if read_json(marker_path) != marker:
                raise ValueError("Preparation cache identity mismatch; use new output paths")
        elif root.exists() and any(root.iterdir()):
            raise ValueError("Nonempty data root without a preparation cache identity")
        existing = read_json(manifest_path) if manifest_path.exists() else None
        if existing is not None:
            body = {key: value for key, value in existing.items() if key != "manifest_id"}
            if existing.get("manifest_id") != fingerprint(body):
                raise ValueError("Existing manifest integrity mismatch")
            if existing.get("preparation_id") != preparation_id:
                raise ValueError("Existing manifest preparation identity mismatch")
        root.mkdir(parents=True, exist_ok=True)
        write_json(marker_path, marker)
        for name in split_contents:
            extract_member(outer, outer_members["chess/" + name], root, name)

        images = []
        counts = {role: {"available": 0, "selected": 0} for role in ROLES}
        selected_bytes = 0
        for sequence, info in sorted(sequences.items()):
            role = assignments[sequence]
            with nested_archive(outer, info, root) as nested:
                for index, color, pose in sequence_frames(nested, sequence):
                    selected = index % roles[role]["stride"] == 0
                    row = {
                        "image_id": f"chess/{sequence}/frame-{index:06d}",
                        "sequence_id": sequence,
                        "frame_index": index,
                        "role": role,
                        "selected": selected,
                        "image_path": color.filename,
                        "pose_path": pose.filename,
                        "width": 640,
                        "height": 480,
                        "image_size_bytes": color.file_size,
                        "pose_size_bytes": pose.file_size,
                        "image_crc32": f"{color.CRC:08x}",
                        "pose_crc32": f"{pose.CRC:08x}",
                    }
                    images.append(row)
                    counts[role]["available"] += 1
                    counts[role]["selected"] += int(selected)
                    if selected:
                        image_path = extract_member(nested, color, root, color.filename)
                        validate_png_header(image_path)
                        extract_member(nested, pose, root, pose.filename)
                        selected_bytes += color.file_size + pose.file_size
        if any(count["selected"] == 0 for count in counts.values()):
            raise ValueError("Every role must have at least one selected original")
        manifest = {
            **identity,
            "preparation_id": preparation_id,
            "split": split,
            "official_sequences": official,
            "official_split_contents": {name: value.decode("utf-8-sig")
                                        for name, value in split_contents.items()},
            "path_base": "data_root (resolved relative to the experiment config)",
            "counts": counts,
            "selected_asset_bytes": selected_bytes,
            "selected_asset_files": 2 * sum(count["selected"] for count in counts.values()),
            "images": images,
        }
        manifest["manifest_id"] = fingerprint(manifest)
        if existing is not None and existing != manifest:
            raise ValueError("Existing manifest differs from the current archive inventory")
        write_json(manifest_path, manifest)
        return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiment.json"))
    args = parser.parse_args()
    start = perf_counter()
    manifest = prepare(args.config)
    print(json.dumps({
        "manifest_path": manifest["manifest_path"],
        "manifest_id": manifest["manifest_id"],
        "split_id": manifest["split_id"],
        "archive_sha256": manifest["archive_sha256"],
        "archive_size_bytes": manifest["archive_size_bytes"],
        "counts": manifest["counts"],
        "selected_asset_bytes": manifest["selected_asset_bytes"],
        "elapsed_seconds": round(perf_counter() - start, 3),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
