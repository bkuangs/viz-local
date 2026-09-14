# viz-local

Failure-aware visual localization: when should a system trust an estimated camera pose?

**Status: planning only.** No localization pipeline, experiments, or measured results exist yet. See [PLAN.md](PLAN.md) for implementation chunks, budget options, and explicit approval checkpoints.

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

Read the checkpoint status and budget options in [PLAN.md](PLAN.md). Choose a budget tier and confirm the GPU machine's operating system and available memory/storage. Implement only the approved chunk, present its evidence, and pause before continuing.

Dataset archives, extracted images, maps, weights, and intermediate outputs stay outside Git. Dataset and model licenses still apply; a public repository does not make third-party assets redistributable.

## References

- [hloc](https://github.com/cvg/Hierarchical-Localization)
- [LightGlue](https://github.com/cvg/LightGlue)
- [7-Scenes dataset and terms](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)
