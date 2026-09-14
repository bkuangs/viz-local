from pathlib import Path

import numpy as np
import pycolmap


def validate_pose(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("A pose must be a finite 4 x 4 matrix")
    if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-8, rtol=0):
        raise ValueError("Invalid homogeneous pose row")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3, rtol=0):
        raise ValueError("Pose rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1, atol=1e-3, rtol=0):
        raise ValueError("Pose rotation must have determinant +1")
    return pose


def load_pose(path: Path) -> np.ndarray:
    return validate_pose(np.loadtxt(path))


def invert_pose(pose: np.ndarray) -> np.ndarray:
    pose = validate_pose(pose)
    inverse = np.eye(4)
    inverse[:3, :3] = pose[:3, :3].T
    inverse[:3, 3] = -pose[:3, :3].T @ pose[:3, 3]
    return inverse


def camera_center(world_to_camera: np.ndarray) -> np.ndarray:
    return invert_pose(world_to_camera)[:3, 3]


def pose_errors(
    estimated_world_to_camera: np.ndarray, ground_truth_camera_to_world: np.ndarray
) -> tuple[float, float]:
    estimate = validate_pose(estimated_world_to_camera)
    truth = validate_pose(ground_truth_camera_to_world)
    translation_m = np.linalg.norm(camera_center(estimate) - truth[:3, 3])
    relative_rotation = estimate[:3, :3] @ truth[:3, :3]
    cosine = np.clip((np.trace(relative_rotation) - 1) / 2, -1, 1)
    rotation_deg = np.degrees(np.arccos(cosine))
    return float(translation_m), float(rotation_deg)


def reference_model(
    camera_to_world_poses: dict[str, np.ndarray], intrinsics: np.ndarray
) -> pycolmap.Reconstruction:
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if (intrinsics.shape != (4,) or not np.isfinite(intrinsics).all()
            or np.any(intrinsics[:2] <= 0)):
        raise ValueError("PINHOLE intrinsics must be finite [fx, fy, cx, cy] with positive focal lengths")
    if len(camera_to_world_poses) < 2:
        raise ValueError("Reference triangulation needs at least two images")
    model = pycolmap.Reconstruction()
    camera = pycolmap.Camera(
        camera_id=1, model="PINHOLE", width=640, height=480,
        params=intrinsics, has_prior_focal_length=True,
    )
    model.add_camera(camera)
    rig = pycolmap.Rig(rig_id=1)
    rig.add_ref_sensor(camera.sensor_id)
    model.add_rig(rig)
    for image_id, (name, pose) in enumerate(sorted(camera_to_world_poses.items()), 1):
        image = pycolmap.Image(
            name=name, camera_id=1, image_id=image_id, frame_id=image_id
        )
        frame = pycolmap.Frame(
            frame_id=image_id, rig_id=1,
            rig_from_world=pycolmap.Rigid3d(invert_pose(pose)[:3]),
        )
        frame.add_data_id(image.data_id)
        model.add_frame(frame)
        model.add_image(image)
        model.register_frame(image_id)
    return model
