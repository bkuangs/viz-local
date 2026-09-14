import copy
import io
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import zlib

from viz_local.io import fingerprint, read_json, sha256_file, write_json
from viz_local.prepare import official_sequences, prepare, validate_roles


ROLES = {
    "reference": {"sequences": ["seq-01", "seq-02"], "stride": 2},
    "confidence_fit": {"sequences": ["seq-04"], "stride": 3},
    "confidence_calibration": {"sequences": ["seq-06"], "stride": 3},
    "final_evaluation": {"sequences": ["seq-03", "seq-05"], "stride": 2},
}
TRAIN = ["seq-01", "seq-02", "seq-04", "seq-06"]
TEST = ["seq-03", "seq-05"]
ALL = sorted(TRAIN + TEST)


def png(width=640, height=480):
    def chunk(kind, payload):
        return (struct.pack(">I", len(payload)) + kind + payload
                + struct.pack(">I", zlib.crc32(kind + payload)))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress((b"\x00" + bytes(width * 3)) * height))
            + chunk(b"IEND", b""))


class SplitTests(unittest.TestCase):
    def test_official_split_parsing(self):
        self.assertEqual(official_sequences(b"sequence2\r\nsequence1\r\n"), TRAIN[:2])
        for value in (b"", b"sequence1\nsequence1\n", b"../sequence1\n", b"sequence0\n"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                official_sequences(value)

    def test_overlap_leakage_and_coverage(self):
        self.assertEqual(validate_roles(ROLES, TRAIN, TEST, ALL)["seq-03"],
                         "final_evaluation")
        variants = []
        overlap = copy.deepcopy(ROLES)
        overlap["confidence_fit"]["sequences"] = ["seq-01"]
        variants.append(overlap)
        leakage = copy.deepcopy(ROLES)
        leakage["confidence_fit"]["sequences"] = ["seq-03"]
        leakage["final_evaluation"]["sequences"] = ["seq-04", "seq-05"]
        variants.append(leakage)
        missing = copy.deepcopy(ROLES)
        missing["reference"]["sequences"] = ["seq-01"]
        variants.append(missing)
        for stride in (0, -1, 1.5, True):
            bad_stride = copy.deepcopy(ROLES)
            bad_stride["reference"]["stride"] = stride
            variants.append(bad_stride)
        for roles in variants:
            with self.subTest(roles=roles), self.assertRaises(ValueError):
                validate_roles(roles, TRAIN, TEST, ALL)
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_roles(ROLES, TRAIN + ["seq-03"], TEST, ALL)
        with self.assertRaisesRegex(ValueError, "actual sequences"):
            validate_roles(ROLES, TRAIN, TEST, ALL + ["seq-07"])


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config_path = self.root / "experiment.json"
        self.archive = self.root / "chess.zip"
        self.config = {
            "scene": "chess", "seed": 17, "dataset_url": "synthetic://chess",
            "archive_path": "chess.zip", "data_root": "data/chess",
            "manifest_path": "artifacts/data_manifest.json", "roles": copy.deepcopy(ROLES),
        }
        self.make_archive()

    def make_archive(self, missing=None, extra=None, image=None):
        color = png() if image is None else image
        with zipfile.ZipFile(self.archive, "w", zipfile.ZIP_DEFLATED) as outer:
            outer.writestr("chess/TrainSplit.txt", "sequence1\r\nsequence2\r\nsequence4\r\nsequence6\r\n")
            outer.writestr("chess/TestSplit.txt", "sequence3\r\nsequence5\r\n")
            for sequence in ALL:
                contents = io.BytesIO()
                with zipfile.ZipFile(contents, "w", zipfile.ZIP_DEFLATED) as nested:
                    nested.writestr(sequence + "/", b"")
                    # Gaps establish index-modulo selection, not list-position slicing.
                    for index in (1, 2, 4, 6):
                        for kind, data in (("color.png", color), ("depth.png", b"not extracted"),
                                           ("pose.txt", b"opaque pose; must never be parsed")):
                            name = f"{sequence}/frame-{index:06d}.{kind}"
                            if name != missing:
                                nested.writestr(name, data)
                    if extra and sequence == "seq-01":
                        nested.writestr(extra, b"unsafe")
                outer.writestr(f"chess/{sequence}.zip", contents.getvalue())
        self.config["archive_size_bytes"] = self.archive.stat().st_size
        self.config["archive_sha256"] = sha256_file(self.archive)
        write_json(self.config_path, self.config)

    def test_deterministic_inventory_selection_and_resume(self):
        first = prepare(self.config_path)
        manifest_path = self.root / self.config["manifest_path"]
        before = manifest_path.read_bytes()
        selected = [row for row in first["images"] if row["selected"]]
        asset = self.root / self.config["data_root"] / selected[0]["image_path"]
        mtime = asset.stat().st_mtime_ns
        self.config["probe"] = {"unrelated_future_setting": True}
        write_json(self.config_path, self.config)
        self.assertEqual(prepare(self.config_path), first)
        self.assertEqual(manifest_path.read_bytes(), before)
        self.assertEqual(asset.stat().st_mtime_ns, mtime)
        self.assertEqual(len(first["images"]), 24)
        self.assertEqual(first["counts"]["reference"], {"available": 8, "selected": 6})
        self.assertEqual(first["counts"]["confidence_fit"], {"available": 4, "selected": 1})
        self.assertEqual(first["counts"]["final_evaluation"], {"available": 8, "selected": 6})
        self.assertEqual([row["frame_index"] for row in selected if row["sequence_id"] == "seq-01"],
                         [2, 4, 6])
        for row in first["images"]:
            for key in ("image_path", "pose_path"):
                self.assertEqual((self.root / self.config["data_root"] / row[key]).exists(),
                                 row["selected"])
        self.assertFalse(list((self.root / "data").rglob("*.depth.png")))
        self.assertFalse(list((self.root / "data").rglob(".prepare-*")))
        self.assertEqual((self.root / "data/chess/TestSplit.txt").read_bytes(),
                         b"sequence3\r\nsequence5\r\n")
        self.assertTrue(self.archive.exists())

    def test_stride_changes_do_not_reassign_originals_or_variants(self):
        first = prepare(self.config_path)
        self.config["data_root"] = "different-data"
        self.config["manifest_path"] = "different-manifest.json"
        for settings in self.config["roles"].values():
            settings["stride"] = 1
        write_json(self.config_path, self.config)
        second = prepare(self.config_path)
        self.assertEqual(first["split_id"], second["split_id"])
        self.assertNotEqual(first["manifest_id"], second["manifest_id"])
        self.assertEqual({row["image_id"]: row["role"] for row in first["images"]},
                         {row["image_id"]: row["role"] for row in second["images"]})
        self.assertIn("inherit", second["split"]["variant_policy"])

    def test_incomplete_archive_is_not_opened(self):
        self.config["archive_size_bytes"] += 1
        write_json(self.config_path, self.config)
        with patch("viz_local.prepare.zipfile.is_zipfile") as is_zip:
            with self.assertRaisesRegex(ValueError, "incomplete"):
                prepare(self.config_path)
            is_zip.assert_not_called()
        self.assertFalse((self.root / "data").exists())

    def test_archive_hash_mismatch(self):
        self.config["archive_sha256"] = "0" * 64
        write_json(self.config_path, self.config)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            prepare(self.config_path)

    def test_exact_size_non_zip_is_rejected(self):
        self.archive.write_bytes(b"not a zip")
        self.config["archive_size_bytes"] = self.archive.stat().st_size
        write_json(self.config_path, self.config)
        with self.assertRaisesRegex(ValueError, "complete ZIP"):
            prepare(self.config_path)

    def test_missing_selected_asset_resumes_but_truncation_fails(self):
        manifest = prepare(self.config_path)
        row = next(row for row in manifest["images"] if row["selected"])
        asset = self.root / self.config["data_root"] / row["pose_path"]
        contents = asset.read_bytes()
        asset.unlink()
        self.assertEqual(prepare(self.config_path), manifest)
        self.assertEqual(asset.read_bytes(), contents)
        asset.write_bytes(contents[:-1])
        with self.assertRaisesRegex(ValueError, "size/type mismatch"):
            prepare(self.config_path)

    def test_corrupt_existing_assets_are_not_overwritten(self):
        manifest = prepare(self.config_path)
        row = next(row for row in manifest["images"] if row["selected"])
        for key in ("image_path", "pose_path"):
            asset = self.root / self.config["data_root"] / row[key]
            original = asset.read_bytes()
            corrupt = bytes([original[0] ^ 1]) + original[1:]
            asset.write_bytes(corrupt)
            with self.subTest(asset=key), self.assertRaisesRegex(ValueError, "CRC32"):
                prepare(self.config_path)
            self.assertEqual(asset.read_bytes(), corrupt)
            asset.write_bytes(original)
        self.assertFalse(list((self.root / "data").rglob(".prepare-*")))

    def test_manifest_and_cache_mismatches(self):
        manifest = prepare(self.config_path)
        path = self.root / self.config["manifest_path"]
        corrupt = copy.deepcopy(manifest)
        corrupt["images"][0]["role"] = "final_evaluation"
        write_json(path, corrupt)
        with self.assertRaisesRegex(ValueError, "manifest integrity"):
            prepare(self.config_path)
        body = {key: value for key, value in corrupt.items() if key != "manifest_id"}
        corrupt["manifest_id"] = fingerprint(body)
        write_json(path, corrupt)
        with self.assertRaisesRegex(ValueError, "archive inventory"):
            prepare(self.config_path)
        write_json(path, manifest)
        self.config["roles"]["reference"]["stride"] = 1
        write_json(self.config_path, self.config)
        with self.assertRaisesRegex(ValueError, "cache identity"):
            prepare(self.config_path)
        self.assertEqual(read_json(path), manifest)

    def test_missing_image_pose_and_unsafe_member(self):
        for name in ("seq-01/frame-000001.pose.txt", "seq-01/frame-000001.color.png"):
            self.make_archive(missing=name)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Missing image/pose"):
                prepare(self.config_path)
            # Each incompatible synthetic archive uses a fresh, explicitly named output root.
            self.config["data_root"] += "-next"
        self.make_archive(extra="seq-01/../../escape.pose.txt")
        with self.assertRaisesRegex(ValueError, "Unexpected sequence archive member"):
            prepare(self.config_path)
        self.assertFalse((self.root / "escape.pose.txt").exists())
        self.assertFalse(list(self.root.rglob(".prepare-*")))

    def test_wrong_dimensions(self):
        self.make_archive(image=png(320, 240))
        with self.assertRaisesRegex(ValueError, "640x480"):
            prepare(self.config_path)

    def test_symlink_asset_is_rejected(self):
        manifest = prepare(self.config_path)
        row = next(row for row in manifest["images"] if row["selected"])
        asset = self.root / self.config["data_root"] / row["pose_path"]
        outside = self.root / "outside.txt"
        outside.write_bytes(asset.read_bytes())
        asset.unlink()
        asset.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "symlink"):
            prepare(self.config_path)


if __name__ == "__main__":
    unittest.main()
