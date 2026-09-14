import unittest

import numpy as np

from viz_local.geometry import (
    camera_center, invert_pose, pose_errors, reference_model, validate_pose,
)


class PoseConventionTests(unittest.TestCase):
    def test_rotated_camera_center_is_not_extrinsic_translation(self):
        camera_to_world = np.array(
            [[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]],
            dtype=float,
        )
        world_to_camera = invert_pose(camera_to_world)
        np.testing.assert_allclose(world_to_camera[:3, 3], [-2, 1, -3])
        np.testing.assert_allclose(camera_center(world_to_camera), [1, 2, 3])
        np.testing.assert_allclose(invert_pose(world_to_camera), camera_to_world)
        self.assertEqual(pose_errors(world_to_camera, camera_to_world), (0.0, 0.0))

    def test_known_translation_and_rotation_errors(self):
        estimate_c2w = np.array(
            [[0, -1, 0, 0.03], [1, 0, 0, 0.04], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=float,
        )
        translation, rotation = pose_errors(invert_pose(estimate_c2w), np.eye(4))
        self.assertAlmostEqual(translation, 0.05)
        self.assertAlmostEqual(rotation, 90)

    def test_rotation_geodesic_at_half_turn(self):
        half_turn = np.diag([-1.0, -1.0, 1.0, 1.0])
        self.assertEqual(pose_errors(half_turn, np.eye(4)), (0.0, 180.0))

    def test_invalid_poses_are_errors(self):
        reflection = np.diag([-1.0, 1.0, 1.0, 1.0])
        scale = np.diag([2.0, 1.0, 1.0, 1.0])
        bad_row = np.eye(4)
        bad_row[3, 0] = 1
        for pose in [np.zeros((3, 4)), np.full((4, 4), np.nan), reflection, scale, bad_row]:
            with self.subTest(pose=pose):
                with self.assertRaises(ValueError):
                    validate_pose(pose)

    def test_colmap_model_preserves_reference_pose_and_camera_sharing(self):
        pose = np.array(
            [[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]],
            dtype=float,
        )
        model = reference_model({"second.png": np.eye(4), "first.png": pose},
                                np.array([600, 600, 320, 240]))
        self.assertEqual(model.num_cameras(), 1)
        self.assertEqual(model.num_reg_images(), 2)
        image = model.find_image_with_name("first.png")
        np.testing.assert_allclose(
            image.cam_from_world().matrix(), invert_pose(pose)[:3], atol=1e-12
        )
        np.testing.assert_allclose(image.projection_center(), pose[:3, 3])


if __name__ == "__main__":
    unittest.main()
