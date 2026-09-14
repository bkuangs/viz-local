import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from .io import sha256_file, write_json


def git_revision(path: Path) -> str:
    status = subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain"], text=True
    ).strip()
    if status:
        raise RuntimeError(f"Pinned dependency has local changes: {path}\n{status}")
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def dependency_identity() -> dict:
    lightglue_source = importlib.metadata.distribution("lightglue").read_text(
        "direct_url.json"
    )
    if lightglue_source is None:
        raise RuntimeError("LightGlue must be installed from its pinned Git revision")
    lightglue_revision = json.loads(lightglue_source)["vcs_info"]["commit_id"]
    return {
        "python": platform.python_version(),
        "uv_lock_sha256": sha256_file(Path("uv.lock")),
        "hloc_revision": git_revision(Path("external/hloc")),
        "superpoint_revision": git_revision(
            Path("external/hloc/third_party/SuperGluePretrainedNetwork")
        ),
        "lightglue_revision": lightglue_revision,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "torchvision", "pycolmap", "hloc", "lightglue",
                         "numpy", "scipy", "h5py", "opencv-python", "pillow")
        },
    }


def inspect_environment() -> dict:
    import pycolmap
    import torch
    import torchvision

    if not torch.cuda.is_available():
        raise RuntimeError("Chunk 1 requires CUDA; refusing a silent CPU fallback")
    start = perf_counter()
    torch.manual_seed(17)
    left = torch.randn(128, 128, device="cuda")
    product = left @ left.T
    if not torch.isfinite(product).all().item():
        raise RuntimeError("CUDA matrix multiplication produced non-finite values")
    boxes = torch.tensor([[0, 0, 2, 2], [0, 0, 2, 2]], device="cuda", dtype=torch.float32)
    kept = torchvision.ops.nms(boxes, torch.tensor([0.9, 0.8], device="cuda"), 0.5)
    if kept.tolist() != [0]:
        raise RuntimeError("torchvision CUDA NMS returned an unexpected result")
    torch.cuda.synchronize()
    camera = pycolmap.Camera(
        model="PINHOLE", width=640, height=480, params=[600, 600, 320, 240]
    )
    if list(camera.img_from_cam([0.0, 0.0, 1.0])) != [320.0, 240.0]:
        raise RuntimeError("PyCOLMAP camera projection did not match its convention")
    gpu = torch.cuda.get_device_properties(0)
    disk = shutil.disk_usage(".")
    host_disk = shutil.disk_usage("/mnt/c") if Path("/mnt/c").is_mount() else None
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dependencies": dependency_identity(),
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
        "ram_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
        "filesystem_free_bytes": disk.free,
        "windows_c_free_bytes": None if host_disk is None else host_disk.free,
        "gpu": {
            "name": gpu.name,
            "total_memory_bytes": gpu.total_memory,
            "compute_capability": [gpu.major, gpu.minor],
            "torch_cuda_runtime": torch.version.cuda,
            "driver_report": subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                text=True,
            ).strip(),
        },
        "smoke_operations": ["CUDA matrix multiplication", "torchvision CUDA NMS",
                             "PyCOLMAP pinhole projection"],
        "smoke_elapsed_seconds": perf_counter() - start,
        "scope": "Compatibility smoke operations, not localization or query latency",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/chunk1/environment.json"))
    args = parser.parse_args()
    report = inspect_environment()
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
