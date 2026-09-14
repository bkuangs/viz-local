# viz-local

Failure-aware visual localization: when should a system trust an estimated camera pose?

**Status: Chunk 1 in progress under the approved Balanced budget.** Budget reconnaissance and approval checkpoints are recorded in [PLAN.md](PLAN.md). No localization pipeline, experiments, or measured localization results exist yet. Chunk 2 is not authorized.

## The question

Can the geometric structure of 2D-3D correspondences identify incorrect camera poses beyond what inlier count, inlier ratio, and reprojection error already reveal?

This is an empirical hypothesis, not a promised improvement or a claim of research novelty. The interesting cases are poses that PnP-RANSAC returns with apparently convincing support but that disagree with ground truth.

## Agreed direction

- Python, PyTorch, COLMAP/PyCOLMAP, and hloc; SuperPoint + LightGlue first.
- 7-Scenes: one-scene pilot, expanding to two or three scenes only if justified.
- Fixed reference-only maps, with separate confidence-development and held-out evaluation queries.
- A small logistic-regression confidence score using conventional signals and correspondence geometry.
- Three threshold baselines plus a conventional-signals-only regression ablation.
- Synthetic blur, brightness reduction, and occlusion; real camera viewpoints for viewpoint analysis.
- NVIDIA RTX 3060 Ti as the intended experiment GPU.
- SIFT pipeline comparison only as an explicitly approved stretch goal.

The project owns the confidence logic, experimental protocol, and failure analysis. It reuses localization infrastructure rather than rebuilding it.

## Intended deliverables

One reproducible offline experiment, result tables and plots, and a short report with inspectable failure cases. Report pose errors, localization success, failure-detection AUROC, accuracy at stated acceptance rates, and runtime.

No web application, custom SLAM system, feature-network training, or large testing framework is planned.

## Before implementation

Read the checkpoint status and selected budget in [PLAN.md](PLAN.md). Balanced allows 35-50 hands-on hours and 40-60 GB working space, starting with Chess only. The current host is Ubuntu on WSL2 with an 8 GB RTX 3060 Ti and about 16 GiB RAM. Chunk 1 must establish actual environment compatibility and pilot cost. Implement only the approved chunk, present its evidence, and pause before continuing.

Dataset archives, extracted images, maps, weights, and intermediate outputs stay outside Git. Dataset and model licenses still apply; a public repository does not make third-party assets redistributable.

## Chunk 1 environment

Run from the repository root on the existing Linux/WSL2 GPU host, with Python
3.10 and `uv` available:

```bash
bash scripts/bootstrap.sh
uv run --locked python -m viz_local.environment
```

The bootstrap keeps an unmodified, commit-pinned hloc checkout in ignored
`external/hloc`; only its required SuperPoint source/weights submodule is
initialized. LightGlue is pinned to a Git revision in `pyproject.toml` and
`uv.lock`. PyTorch 2.7.1 and torchvision 0.22.1 use CUDA 12.6 wheels;
PyCOLMAP is pinned to 3.13.0. All transitive Python dependencies are locked.
Do not install a second CUDA toolkit or replace the Windows GPU driver from
inside WSL. COLMAP geometry runs on CPU; neural extraction/matching runs on GPU.

The environment command fails rather than silently falling back to CPU and
records actual versions, hardware, and compatibility operations in
`artifacts/chunk1/environment.json`. The initial observed snapshot is in
`results/chunk1/environment.json`; it is not a localization benchmark.

## References

- [hloc](https://github.com/cvg/Hierarchical-Localization)
- [LightGlue](https://github.com/cvg/LightGlue)
- [7-Scenes dataset and terms](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)
