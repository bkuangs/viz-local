import unittest
from pathlib import Path
import tempfile

import h5py
import numpy as np
import pycolmap
from hloc.utils.parsers import names_to_pair

from viz_local.geometry import reference_model
from viz_local.probe import (
    independent_reference_reprojection, reference_groups, residual_summary,
    sampson_residuals, select_references,
)


class ReferenceSelectionTests(unittest.TestCase):
    def test_only_selected_reference_roles_enter_probe(self):
        images = [
            {
                "image_path": f"{sequence}/frame-{index:06d}.color.png",
                "sequence_id": sequence,
                "role": role,
                "selected": selected,
            }
            for sequence, role, selected in [
                ("seq-01", "reference", True),
                ("seq-02", "reference", True),
                ("seq-03", "final_evaluation", True),
                ("seq-04", "confidence_fit", True),
                ("seq-05", "reference", False),
                ("seq-06", "confidence_calibration", True),
            ]
            for index in range(8)
        ]
        result = select_references({"images": list(reversed(images))}, 2)
        self.assertEqual(len(result), 8)
        self.assertEqual({row["sequence_id"] for row in result}, {"seq-01", "seq-02"})
        self.assertTrue(all(row["role"] == "reference" for row in result))
        self.assertEqual(result[1]["image_path"], "seq-01/frame-000002.color.png")

    def test_empty_reference_set_or_invalid_stride_is_an_error(self):
        for stride in (0, 1, -1):
            with self.subTest(stride=stride):
                with self.assertRaises(ValueError):
                    select_references({"images": []}, stride)

    def test_calibration_holdout_contains_no_fit_image(self):
        rows = [
            {"sequence_id": sequence, "image_path": f"{sequence}/{index}.png"}
            for sequence in ("seq-01", "seq-02") for index in range(13)
        ]
        fit, holdout = reference_groups(rows)
        self.assertEqual(len(fit), 20)
        self.assertEqual(len(holdout), 6)
        self.assertFalse({row["image_path"] for row in fit} &
                         {row["image_path"] for row in holdout})

    def test_sampson_residuals_use_pixel_intrinsics_and_pose_direction(self):
        intrinsics = np.array([500.0, 530.0, 320.0, 240.0])
        camera = pycolmap.Camera(
            model="PINHOLE", width=640, height=480, params=intrinsics
        )
        angle = 0.15
        rotation = np.array([
            [np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ])
        transform = pycolmap.Rigid3d(pycolmap.Rotation3d(rotation), [0.2, 0.05, 0.03])
        points = np.random.default_rng(17).uniform([-0.8, -0.6, 2], [0.8, 0.6, 4], (32, 3))
        pair = {
            "points0": camera.img_from_cam(points),
            "points1": camera.img_from_cam(transform * points),
            "essential": pycolmap.essential_matrix_from_pose(transform),
        }
        self.assertLess(np.max(sampson_residuals(intrinsics, [pair])), 1e-10)
        self.assertGreater(np.median(sampson_residuals(
            np.array([600, 600, 320, 240]), [pair]
        )), 0.01)

    def test_undefined_residuals_are_not_success_shaped(self):
        for values in (np.array([]), np.array([float("nan")])):
            with self.assertRaises(ValueError):
                residual_summary(values)


class IndependentReprojectionTests(unittest.TestCase):
    def test_deduplication_and_heldout_errors_cannot_refit_structure(self):
        pose_b = np.eye(4)
        pose_b[0, 3] = 0.1
        model = reference_model(
            {"fit-a.png": np.eye(4), "fit-b.png": pose_b},
            np.array([600, 600, 320, 240]),
        )
        point3d = np.array([0.0, 0.0, 2.0])
        for image in model.images.values():
            xy = model.cameras[1].img_from_cam(image.cam_from_world() * point3d)
            image.points2D = pycolmap.Point2DList([pycolmap.Point2D(xy)])
        point_id = model.add_point3D(point3d, pycolmap.Track([
            pycolmap.TrackElement(1, 0), pycolmap.TrackElement(2, 0),
        ]))
        held_pose = np.eye(4)
        held_pose[0, 3] = 0.05
        rows = [{"image_path": "held.png", "sequence_id": "seq-01"}]
        config = {"probe": {
            "calibration": {"min_match_score": 0.2},
            "independent_reprojection": {
                "median_max_px": 2, "p90_max_px": 5,
                "min_correspondences_per_image": 1,
                "min_occupied_4x4_cells_per_image": 1,
                "allow_nonpositive_depth": False,
            },
        }}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features, matches = root / "features.h5", root / "matches.h5"
            with h5py.File(matches, "w") as stream:
                for name in ("fit-a.png", "fit-b.png"):
                    group = stream.create_group(names_to_pair("held.png", name))
                    group.create_dataset("matches0", data=np.array([0], dtype=np.int16))
                    group.create_dataset("matching_scores0", data=[0.9])
            for offset in (0, 10):
                with h5py.File(features, "w") as stream:
                    stream.create_dataset("held.png/keypoints", data=[[304.5 + offset, 239.5]])
                result = independent_reference_reprojection(
                    config, rows, {"held.png": held_pose}, model,
                    features, matches, root / "output",
                )
                image_result = result["per_image"]["held.png"]
                self.assertEqual(image_result["candidate_pair_matches"], 2)
                self.assertEqual(image_result["deduplicated_2d3d_correspondences"], 1)
                self.assertAlmostEqual(result["reprojection"]["median_px"], offset)
                self.assertEqual(result["passed"], offset == 0)
                np.testing.assert_array_equal(model.points3D[point_id].xyz, point3d)
            with self.assertRaisesRegex(ValueError, "contributed to the fit map"):
                independent_reference_reprojection(
                    config, [{"image_path": "fit-a.png"}], {}, model,
                    features, matches, root / "leak",
                )


if __name__ == "__main__":
    unittest.main()
