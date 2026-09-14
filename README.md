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

## Chunk 1 data and pose conventions

The [7-Scenes license](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/7-scenes-msr-la-dataset-7-scenes.rtf)
permits non-commercial uses, including personal experimentation and academic
research, subject to its terms. Read it before downloading. SuperPoint's
upstream code/weights also have restrictive terms; LightGlue uses Apache-2.0.
Dataset images, pose files, pretrained weights, and derived visualizations are
kept local rather than redistributed in Git.

```bash
mkdir -p data/downloads
curl --location --fail --show-error --retry 3 --continue-at - \
  --output data/downloads/chess.zip \
  https://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/chess.zip
curl --location --fail --show-error --retry 3 \
  --output data/downloads/7-scenes-license.rtf \
  https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/7-scenes-msr-la-dataset-7-scenes.rtf
uv run --locked python -m viz_local.prepare --config experiment.json
```

The retained Chess archive is 3,079,608,937 bytes; its observed SHA256 is pinned
in `experiment.json`. This locally recorded digest identifies the download,
not an independently published dataset signature. Preparation uses Python's
standard-library ZIP support; `unzip` is not required. It extracts only selected
RGB/pose files, not depth or dense volumes, and validates existing assets before
reusing them. Skip the download once the complete archive is retained.

The official archive confirms training sequences 01, 02, 04, 06 and test
sequences 03, 05. Proposed roles for checkpoint approval are:

| Role | Whole sequences | Frame-index stride | Selected originals |
| --- | --- | --- | --- |
| Reference map | 01, 02 | 10 | 200 |
| Confidence fit | 04 | 20 | 50 |
| Confidence calibration | 06 | 20 | 50 |
| Final evaluation | 03, 05 | 5 | 400 |

All 6,000 originals receive a role before subsampling; variants must inherit
that role. The final count is provisional, not a completed evaluation.
`artifacts/chunk1/data_manifest.json` records IDs, roles, selection, original
pose conventions, and per-asset size/CRC32. The original poses are
camera-to-world transforms in meters; `viz_local.geometry` explicitly converts
them to COLMAP world-to-camera poses and measures camera-center error rather
than extrinsic-translation error. Preparation does not interpret query pose
values or decode final-image pixels.

The selected 700 originals occupy about 299 MB across 1,400 RGB/pose files.
`results/chunk1/data.json` records the observed preparation summary. RGB
intrinsics are deliberately pending in the data manifest; they must come from
the separate reference-only probe, never from assumed depth-camera calibration.

## References

- [hloc](https://github.com/cvg/Hierarchical-Localization)
- [LightGlue](https://github.com/cvg/LightGlue)
- [7-Scenes dataset and terms](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)
