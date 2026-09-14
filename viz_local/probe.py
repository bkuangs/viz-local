import argparse
import copy
import itertools
import json
import os
import random
import zipfile
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
import pycolmap
import torch
from hloc import extract_features, match_features, triangulation
from hloc.utils.io import get_keypoints, get_matches, read_image
from lightglue import LightGlue
from scipy.optimize import least_squares

from .environment import dependency_identity
from .geometry import invert_pose, load_pose, reference_model
from .io import fingerprint, read_json, sha256_file, write_json
from .prepare import safe_destination, validate_file


def select_references(manifest: dict, stride: int) -> list[dict]:
    if stride < 1:
        raise ValueError("Probe reference stride must be positive")
    references = [
        row for row in manifest["images"]
        if row["role"] == "reference" and row["selected"]
    ]
    sequences = sorted({row["sequence_id"] for row in references})
    selected = []
    for sequence in sequences:
        members = sorted(
            (row for row in references if row["sequence_id"] == sequence),
            key=lambda row: row["image_path"],
        )
        selected.extend(members[::stride])
    if len(selected) < 8:
        raise ValueError("The geometry probe requires at least eight reference images")
    return selected


def prepare_features(config: dict, references: list[dict], output: Path) -> tuple:
    settings = config["local_features"]
    if (settings["configuration"] != "superpoint_aachen"
            or settings["matcher"] != "superpoint+lightglue"
            or settings["storage"] != "float32" or settings["matching_batch_size"] != 1):
        raise ValueError("The approved probe uses float32 SuperPoint and single-pair LightGlue")
    root = Path(config["data_root"])
    feature_conf = copy.deepcopy(extract_features.confs["superpoint_aachen"])
    feature_conf["model"]["max_keypoints"] = config["local_features"]["max_keypoints"]
    feature_conf["preprocessing"]["resize_max"] = config["local_features"]["resize_max"]
    match_conf = copy.deepcopy(match_features.confs["superpoint+lightglue"])
    names = [row["image_path"] for row in references]
    pairs = list(itertools.combinations(names, 2))
    identity = {
        "dependencies": dependency_identity(),
        "seed": config["seed"],
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "features": feature_conf,
        "feature_storage": "float32; hloc keypoints use pixel-center origin (0,0)",
        "matches": match_conf,
        "images": {
            row["image_path"]: {
                "rgb_sha256": sha256_file(root / row["image_path"]),
                "pose_sha256": sha256_file(root / row["pose_path"]),
            } for row in references
        },
    }
    cache = output / "features" / fingerprint(identity)[:16]
    cache.mkdir(parents=True, exist_ok=True)
    identity_path = cache / "identity.json"
    if identity_path.exists() and read_json(identity_path) != identity:
        raise RuntimeError("Feature cache identity mismatch")
    write_json(identity_path, identity)
    features = cache / "features.h5"
    matches = cache / "matches.h5"
    pairs_path = cache / "pairs.txt"
    pairs_path.write_text("".join(f"{first} {second}\n" for first, second in pairs))
    evidence_path = cache / "evidence.json"
    if evidence_path.exists():
        evidence = read_json(evidence_path)
        for path in (features, matches):
            if sha256_file(path) != evidence["cache_sha256"][path.name]:
                raise RuntimeError(f"Corrupt completed feature cache: {path}")
        for name, digest in evidence["weight_sha256"].items():
            if sha256_file(Path(name)) != digest:
                raise RuntimeError(f"Changed pretrained weights: {name}")
        return features, matches, evidence

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = perf_counter()
    extract_features.main(
        feature_conf, root, image_list=names, feature_path=features, as_half=False
    )
    torch.cuda.synchronize()
    extraction_seconds = perf_counter() - start
    start = perf_counter()
    match_features.main(match_conf, pairs_path, features, matches=matches)
    torch.cuda.synchronize()
    matching_seconds = perf_counter() - start
    with h5py.File(features) as stream:
        counts = {name: len(stream[name]["keypoints"]) for name in names}
        for name in names:
            if list(stream[name]["image_size"][:]) != [640, 480]:
                raise ValueError(f"Unexpected image dimensions: {name}")
            if not 0 < counts[name] <= config["local_features"]["max_keypoints"]:
                raise RuntimeError(f"Invalid extracted feature count for {name}")
    pair_counts = {
        f"{first} {second}": len(get_matches(matches, first, second)[0])
        for first, second in pairs
    }
    if not any(count >= 32 for count in pair_counts.values()):
        raise RuntimeError("No real reference pair produced at least 32 matches")
    weight_paths = [
        Path("external/hloc/third_party/SuperGluePretrainedNetwork/models/weights/superpoint_v1.pth"),
        Path(torch.hub.get_dir()) / "checkpoints" /
        f"{LightGlue.features['superpoint']['weights']}_{LightGlue.version.replace('.', '-')}.pth",
    ]
    evidence = {
        "identity": identity,
        "image_count": len(names),
        "pair_count": len(pairs),
        "keypoint_counts": counts,
        "match_counts": pair_counts,
        "extraction_wall_seconds": extraction_seconds,
        "matching_wall_seconds": matching_seconds,
        "timing_scope": "Reference probe only; includes setup/decoding and any weight download, "
                        "may resume partial caches; not warmed full-query latency",
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "cache_sha256": {path.name: sha256_file(path) for path in (features, matches)},
        "weight_sha256": {str(path): sha256_file(path) for path in weight_paths},
    }
    write_json(evidence_path, evidence)
    return features, matches, evidence


def save_pair_example(root: Path, features: Path, matches: Path, evidence: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hloc.utils.viz import plot_images, plot_matches

    pair = next(pair for pair, count in sorted(evidence["match_counts"].items()) if count >= 32)
    first, second = pair.split()
    indices, _ = get_matches(matches, first, second)
    indices = indices[np.linspace(0, len(indices) - 1, min(80, len(indices)), dtype=int)]
    plot_images([read_image(root / first), read_image(root / second)],
                titles=[first, second])
    plot_matches(get_keypoints(features, first)[indices[:, 0]],
                 get_keypoints(features, second)[indices[:, 1]], lw=0.6, ps=2)
    plt.savefig(output / "reference_pair.png", dpi=120)
    plt.close()


def reference_groups(references: list[dict]) -> tuple[list[dict], list[dict]]:
    fit, holdout = [], []
    for sequence in sorted({row["sequence_id"] for row in references}):
        rows = [row for row in references if row["sequence_id"] == sequence]
        for index, row in enumerate(rows):
            (holdout if index % 4 == 2 else fit).append(row)
    return fit, holdout


def calibration_pairs(
    rows: list[dict], poses: dict, features: Path, matches: Path, settings: dict
) -> tuple[list[dict], list[dict]]:
    observations, accounting = [], []
    for first, second in itertools.combinations(rows, 2):
        name0, name1 = first["image_path"], second["image_path"]
        pose0, pose1 = poses[name0], poses[name1]
        baseline = float(np.linalg.norm(pose0[:3, 3] - pose1[:3, 3]))
        indices, scores = get_matches(matches, name0, name1)
        if not np.isfinite(scores).all():
            raise ValueError(f"Non-finite match scores for {name0} and {name1}")
        selected = np.flatnonzero(scores >= settings["min_match_score"])
        selected = selected[np.argsort(-scores[selected], kind="stable")]
        selected = selected[:settings["max_matches_per_pair"]]
        reason = "used"
        if baseline < settings["min_baseline_m"]:
            reason = "short_baseline"
        elif len(selected) < settings["min_matches_per_pair"]:
            reason = "insufficient_matches"
        accounting.append({
            "pair": [name0, name1], "baseline_m": baseline, "raw_matches": len(indices),
            "selected_matches": len(selected), "status": reason,
        })
        if reason != "used":
            continue
        indices = indices[selected]
        relative_pose = (
            pycolmap.Rigid3d(invert_pose(pose1)[:3])
            * pycolmap.Rigid3d(invert_pose(pose0)[:3]).inverse()
        )
        observations.append({
            "essential": pycolmap.essential_matrix_from_pose(relative_pose),
            # hloc stores centers at (0,0); COLMAP camera coordinates start at (0.5,0.5).
            "points0": get_keypoints(features, name0)[indices[:, 0]].astype(float) + 0.5,
            "points1": get_keypoints(features, name1)[indices[:, 1]].astype(float) + 0.5,
        })
    return observations, accounting


def sampson_residuals(intrinsics: np.ndarray, pairs: list[dict]) -> np.ndarray:
    fx, fy, cx, cy = intrinsics
    inverse_k = np.linalg.inv(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]))
    errors = [
        pycolmap.compute_squared_sampson_error(
            pair["points0"], pair["points1"], inverse_k.T @ pair["essential"] @ inverse_k
        )
        for pair in pairs
    ]
    residuals = np.sqrt(np.concatenate(errors))
    if not np.isfinite(residuals).all():
        raise ValueError("Reference epipolar residuals are undefined")
    return residuals


def residual_summary(values: np.ndarray) -> dict:
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Cannot summarize empty or non-finite geometry residuals")
    return {
        "count": len(values),
        "median_px": float(np.median(values)),
        "p90_px": float(np.percentile(values, 90)),
        "fraction_within_4px": float(np.mean(values <= 4)),
    }


def calibrate_reference_camera(
    config: dict, references: list[dict], features: Path, matches: Path
) -> tuple[dict, dict]:
    settings = config["probe"]["calibration"]
    policy = config["probe"]["acceptance_policy"]
    if settings["model"] != "PINHOLE":
        raise ValueError("Only the declared effective PINHOLE calibration is implemented")
    root = Path(config["data_root"])
    poses = {row["image_path"]: load_pose(root / row["pose_path"]) for row in references}
    fit, holdout = reference_groups(references)
    fit_pairs, fit_accounting = calibration_pairs(fit, poses, features, matches, settings)
    holdout_pairs, holdout_accounting = calibration_pairs(
        holdout, poses, features, matches, settings
    )
    if len(fit_pairs) < 5 or len(holdout_pairs) < 3:
        raise RuntimeError("Too few eligible reference pairs for calibration and its holdout")
    solutions = [
        least_squares(
            sampson_residuals, np.asarray(initial, dtype=float), args=(fit_pairs,),
            bounds=(settings["lower_bounds"], settings["upper_bounds"]),
            loss="soft_l1", f_scale=1.0, x_scale="jac", max_nfev=200,
        )
        for initial in settings["initializations"]
    ]
    solution = solutions[0]
    stable = all(
        np.max(np.abs(other.x - solution.x)) <= policy["initialization_max_parameter_difference_px"]
        for other in solutions[1:]
    )
    singular_values = np.linalg.svd(solution.jac, compute_uv=False)
    condition_number = (
        float(singular_values[0] / singular_values[-1]) if singular_values[-1] > 0 else None
    )
    fit_summary = residual_summary(sampson_residuals(solution.x, fit_pairs))
    holdout_summary = residual_summary(sampson_residuals(solution.x, holdout_pairs))
    checks = {
        "optimizers_converged": all(result.success for result in solutions),
        "initialization_stability": stable,
        "no_bound_hit": all(not np.any(result.active_mask) for result in solutions),
        "full_rank_well_conditioned_jacobian": bool(
            np.linalg.matrix_rank(solution.jac) == 4
            and condition_number is not None
            and condition_number <= policy["maximum_jacobian_condition_number"]
        ),
        "heldout_median": holdout_summary["median_px"] <= policy["holdout_sampson_median_max_px"],
        "heldout_fraction_within_4px":
            holdout_summary["fraction_within_4px"] >= policy["holdout_sampson_fraction_within_4px_min"],
    }
    report = {
        "camera_model": "PINHOLE",
        "parameter_order": ["fx", "fy", "cx", "cy"],
        "params": solution.x.tolist(),
        "image_size": [640, 480],
        "coordinate_origin": "COLMAP pixel coordinates; image corner (0,0)",
        "provenance": "Effective camera fitted only to reference RGB matches and original "
                      "reference poses; not a manufacturer or RGB/depth extrinsic calibration",
        "pose_assumption": "Original provided camera poses approximate RGB camera poses; "
                           "unknown RGB/depth extrinsics and tracking errors remain",
        "fit_images": [row["image_path"] for row in fit],
        "holdout_reference_images": [row["image_path"] for row in holdout],
        "fit_pair_accounting": fit_accounting,
        "holdout_pair_accounting": holdout_accounting,
        "fit_sampson": fit_summary,
        "holdout_sampson": holdout_summary,
        "initial_holdout_sampson": residual_summary(
            sampson_residuals(np.asarray(settings["initializations"][0]), holdout_pairs)
        ),
        "optimizer": {"method": "trf", "loss": "soft_l1", "f_scale": 1.0,
                      "x_scale": "jac", "max_nfev": 200, "settings": settings},
        "solutions": [
            {"params": result.x.tolist(), "success": bool(result.success),
             "message": result.message, "evaluations": result.nfev, "cost": result.cost}
            for result in solutions
        ],
        "jacobian_singular_values": singular_values.tolist(),
        "jacobian_condition_number": condition_number,
        "acceptance_policy": policy,
        "checks": checks,
        "accepted_for_triangulation_probe": all(checks.values()),
    }
    return report, poses


def triangulate_reference_group(
    config: dict, rows: list[dict], poses: dict, params: list,
    features: Path, matches: Path, output: Path,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    names = [row["image_path"] for row in rows]
    initial = reference_model({name: poses[name] for name in names}, np.asarray(params))
    initial_path = output / "input_model"
    initial_path.mkdir(exist_ok=True)
    initial.write(initial_path)
    pairs_path = output / "pairs.txt"
    pairs_path.write_text("".join(
        f"{first} {second}\n" for first, second in itertools.combinations(names, 2)
    ))
    options = {
        "num_threads": 4, "random_seed": config["seed"], "fix_existing_frames": True,
        "ba_refine_focal_length": False, "ba_refine_principal_point": False,
        "ba_refine_extra_params": False, "ba_refine_sensor_from_rig": False,
        "triangulation": {"ignore_two_view_tracks": False, "random_seed": config["seed"]},
    }
    start = perf_counter()
    model = triangulation.main(
        output / "model", initial_path, Path(config["data_root"]), pairs_path,
        features, matches, estimate_two_view_geometries=True, mapper_options=options,
    )
    elapsed = perf_counter() - start
    pose_drift = max(
        float(np.max(np.abs(
            model.find_image_with_name(name).cam_from_world().matrix()
            - initial.find_image_with_name(name).cam_from_world().matrix()
        ))) for name in names
    )
    if pose_drift > 1e-10 or not np.allclose(model.cameras[1].params, params, atol=1e-10, rtol=0):
        raise RuntimeError("Triangulation changed the fixed reference poses or intrinsics")
    residuals, depths, tracks, counts = [], [], [], {}
    for point in model.points3D.values():
        tracks.append(point.track.length())
    for image in model.images.values():
        counts[image.name] = image.num_points3D
        camera = model.cameras[image.camera_id]
        for point2d in image.points2D:
            if not point2d.has_point3D():
                continue
            point_cam = image.cam_from_world() * model.points3D[point2d.point3D_id].xyz
            depths.append(float(point_cam[2]))
            residuals.append(float(np.linalg.norm(camera.img_from_cam(point_cam) - point2d.xy)))
    if not residuals:
        raise RuntimeError("Reference-only triangulation produced no 3D observations")
    summary = residual_summary(np.asarray(residuals))
    points = np.asarray([point.xyz for point in model.points3D.values()])
    policy = config["probe"]["acceptance_policy"]
    checks = {
        "point_count": model.num_points3D() >= policy["triangulated_points_min_per_group"],
        "median_reprojection": summary["median_px"] <= policy["reprojection_median_max_px"],
        "p90_reprojection": summary["p90_px"] <= policy["reprojection_p90_max_px"],
        "positive_depth_fraction": bool(np.mean(np.asarray(depths) > 0) >= policy["positive_depth_fraction_min"]),
        "images_with_points": sum(count > 0 for count in counts.values())
            >= policy["images_with_points_fraction_min"] * len(names),
        "fixed_reference_poses": pose_drift <= 1e-10,
    }
    return {
        "images": names, "registered_images": model.num_reg_images(),
        "points3D": model.num_points3D(), "observations": len(residuals),
        "reprojection": summary, "observations_per_image": counts,
        "track_length_median": float(np.median(tracks)),
        "depth_median_m": float(np.median(depths)),
        "depth_p90_m": float(np.percentile(depths, 90)),
        "point_bounds_m": [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
        "max_pose_matrix_change": pose_drift, "wall_seconds": elapsed,
        "mapper_options": options,
        "effective_mapper_defaults": pycolmap.IncrementalPipelineOptions(options).summary(),
        "geometric_verification": "hloc estimation_and_geometric_verification; "
                                  "RANSAC max_num_trials=20000, min_inlier_ratio=0.1",
        "model_path": str(output / "model"),
        "checks": checks, "passed": all(checks.values()),
    }


def run_geometry(
    config: dict, references: list[dict], features: Path, matches: Path,
    evidence: dict, output: Path,
) -> dict:
    identity = {
        "feature_identity": evidence["identity"],
        "cache_sha256": evidence["cache_sha256"],
        "weight_sha256": evidence["weight_sha256"],
        "probe_settings": config["probe"],
        "data_manifest_sha256": sha256_file(Path(config["manifest_path"])),
        "code_sha256": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in ("probe.py", "geometry.py")
        },
    }
    run_dir = output / "geometry" / fingerprint(identity)[:16]
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        report = read_json(summary_path)
        if report["identity"] != identity:
            raise RuntimeError("Geometry cache identity mismatch")
        for name, digest in report["model_sha256"].items():
            if sha256_file(Path(name)) != digest:
                raise RuntimeError(f"Changed completed geometry artifact: {name}")
    else:
        start = perf_counter()
        calibration, poses = calibrate_reference_camera(config, references, features, matches)
        write_json(run_dir / "calibration.json", calibration)
        groups = {}
        if calibration["accepted_for_triangulation_probe"]:
            fit, holdout = reference_groups(references)
            for name, rows in (("fit", fit), ("holdout", holdout)):
                groups[name] = triangulate_reference_group(
                    config, rows, poses, calibration["params"], features, matches, run_dir / name
                )
        report = {
            "identity": identity,
            "calibration": calibration,
            "triangulation": groups,
            "geometry_wall_seconds": perf_counter() - start,
            "model_sha256": {
                str(path): sha256_file(path)
                for group in groups.values()
                for path in sorted(Path(group["model_path"]).glob("*.bin"))
            },
            "passed": calibration["accepted_for_triangulation_probe"]
                      and len(groups) == 2 and all(group["passed"] for group in groups.values()),
            "scope": "Reference-only geometry plausibility, not query localization or "
                     "a metric-accuracy guarantee; no reference/query alignment applied",
        }
        write_json(summary_path, report)
    write_json(output / "latest.json", {"summary_path": str(summary_path), "passed": report["passed"]})
    if not report["passed"]:
        raise RuntimeError(f"RGB geometry probe did not pass; review {summary_path} before proceeding")
    return report


def checkpoint_summary(manifest: dict, evidence: dict, report: dict, output: Path) -> dict:
    calibration = report["calibration"]
    counts = np.asarray(list(evidence["keypoint_counts"].values()))
    pair_counts = np.asarray(list(evidence["match_counts"].values()))
    return {
        "scope": report["scope"],
        "data_manifest_id": manifest["manifest_id"],
        "split_id": manifest["split_id"],
        "geometry_report": read_json(output / "latest.json")["summary_path"],
        "reference_pair_visualization": str(output / "reference_pair.png"),
        "geometry_passed": report["passed"],
        "query_attempts_run": 0,
        "features": {
            "images": evidence["image_count"], "pairs": evidence["pair_count"],
            "keypoints_min_median_max": [int(counts.min()), float(np.median(counts)), int(counts.max())],
            "matches_min_median_max": [int(pair_counts.min()), float(np.median(pair_counts)), int(pair_counts.max())],
            "extraction_wall_seconds": evidence["extraction_wall_seconds"],
            "matching_wall_seconds": evidence["matching_wall_seconds"],
            "cuda_peak_allocated_bytes": evidence["cuda_peak_allocated_bytes"],
            "cuda_peak_reserved_bytes": evidence["cuda_peak_reserved_bytes"],
            "timing_scope": evidence["timing_scope"],
            "weight_sha256": evidence["weight_sha256"],
        },
        "calibration": {
            key: calibration[key] for key in (
                "camera_model", "parameter_order", "params", "image_size", "provenance",
                "pose_assumption", "fit_images", "holdout_reference_images",
                "fit_sampson", "holdout_sampson", "initial_holdout_sampson",
                "solutions", "jacobian_condition_number", "checks", "acceptance_policy",
            )
        },
        "triangulation": {
            name: {key: group[key] for key in (
                "registered_images", "points3D", "observations", "reprojection",
                "track_length_median", "depth_median_m", "depth_p90_m",
                "max_pose_matrix_change", "wall_seconds", "model_path", "checks", "passed",
            )} for name, group in report["triangulation"].items()
        },
        "geometry_wall_seconds": report["geometry_wall_seconds"],
        "model_sha256": report["model_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("experiment.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/chunk1/probe"))
    parser.add_argument("--features-only", action="store_true")
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    if args.summary is not None and args.features_only:
        parser.error("--summary requires the complete geometry probe")
    config = read_json(args.config)
    config["data_root"] = str(args.config.parent / config["data_root"])
    config["manifest_path"] = str(args.config.parent / config["manifest_path"])
    if not torch.cuda.is_available():
        raise RuntimeError("Chunk 1 extraction/matching must run on the intended CUDA device")
    torch.hub.set_dir("weights/torch/hub")
    torch.set_num_threads(4)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    pycolmap.set_random_seed(config["seed"])
    manifest = read_json(Path(config["manifest_path"]))
    body = {key: value for key, value in manifest.items() if key != "manifest_id"}
    if manifest["manifest_id"] != fingerprint(body):
        raise ValueError("Data manifest integrity mismatch")
    if manifest["pose_convention"] != "camera_to_world" or manifest["pose_units"] != "meters":
        raise ValueError("The probe requires original camera-to-world poses in meters")
    if (manifest["roles"] != config["roles"]
            or manifest["archive_sha256"] != config["archive_sha256"]):
        raise ValueError("Prepare the configured split and archive before running the probe")
    references = select_references(manifest, config["probe"]["reference_subsample"])
    for row in references:
        for kind in ("image", "pose"):
            info = zipfile.ZipInfo(row[f"{kind}_path"])
            info.file_size = row[f"{kind}_size_bytes"]
            info.CRC = int(row[f"{kind}_crc32"], 16)
            validate_file(safe_destination(Path(config["data_root"]), info.filename), info)
        load_pose(Path(config["data_root"]) / row["pose_path"])
    features, matches, evidence = prepare_features(config, references, args.output)
    save_pair_example(Path(config["data_root"]), features, matches, evidence, args.output)
    print(json.dumps({
        key: evidence[key] for key in
        ("image_count", "pair_count", "extraction_wall_seconds", "matching_wall_seconds",
         "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes")
    }, indent=2))
    if not args.features_only:
        report = run_geometry(config, references, features, matches, evidence, args.output)
        if args.summary is not None:
            write_json(args.summary, checkpoint_summary(manifest, evidence, report, args.output))
        print(json.dumps({
            "geometry_passed": report["passed"],
            "intrinsics": report["calibration"]["params"],
            "heldout_reference_sampson": report["calibration"]["holdout_sampson"],
            "triangulation": {
                name: {key: group[key] for key in ("points3D", "observations", "reprojection", "passed")}
                for name, group in report["triangulation"].items()
            },
        }, indent=2))


if __name__ == "__main__":
    main()
