# Implementation plan

## 1. Goal and working agreement

Build a small, reproducible experiment around this hypothesis:

> Geometric correspondence structure predicts incorrect returned poses beyond inlier count, inlier ratio, and reprojection residuals.

Success means answering that question honestly with held-out evidence and understandable failure cases. It does not require a positive result, a particular AUROC, or a new localization algorithm. Do not fill the resume placeholders with targets or development-set results.

**Current state:** the planning repository exists; implementation has not started. Budget selection and approval for Chunk 1 are pending.

Work one chunk at a time. At every checkpoint, report the artifacts, observed results, time/storage used, blockers, and recommended next decision. Stop and wait for explicit approval before starting another chunk. If a chunk fails its exit criteria, discuss a bounded recovery attempt rather than silently expanding scope.

On resuming, read this file, inspect the actual repository/artifacts, and update the status table only with completed work. Preserve the commands and configuration needed to resume without repeating expensive work.

| Chunk | Status | Approval needed next |
| --- | --- | --- |
| 1. Environment, data, and pose conventions | Not started | Budget tier and permission to begin |
| 2. One-scene localization and failure audit | Not started | Pilot design approved after Chunk 1 |
| 3. Confidence model and development comparison | Not started | Failure evidence and features approved |
| 4. Frozen held-out benchmark | Not started | Protocol and any scene expansion approved |
| 5. Portfolio report and reproducible handoff | Not started | Results and claims reviewed |
| 6. Optional SIFT comparison | Deferred | Separate stretch-goal approval |

## 2. Budget options

These are rough hands-on engineering estimates, not commitments or measured runtime. Dataset downloads and unattended GPU runs are additional elapsed time. Disk numbers are working-space allowances, not dataset download sizes; revise them after the pilot.

| Option | Hands-on budget | Scope ceiling | Final unique query target | Disk allowance |
| --- | --- | --- | --- | --- |
| Lean | 20-30 hours | One scene; complete confidence experiment | 100-200 | 20-30 GB |
| Balanced, recommended | 35-50 hours | Two scenes if the pilot justifies expansion | 300-600 total | 40-60 GB |
| Expanded | 50-70 hours | Up to three scenes; optional SIFT comparison | 600-1,000 total | 60-100 GB |

Targets count unique original final-evaluation images, not their corrupted variants. The final dataset manifest, available sequence splits, and measured cost determine the actual counts. A smaller honest result is preferable to duplicating near-identical frames to inflate N.

Start with the same one-scene pilot under every option. A flexible budget is not permission to run all scenes or implement stretch goals.

The RTX 3060 Ti is a reasonable starting device; confirm actual VRAM, system RAM, OS, driver, and CUDA/PyTorch compatibility in Chunk 1. Prefer native Linux for the experiment machine if available. This Mac can hold the repository, but it is not the assumed CUDA execution environment. Do not add Docker, cloud deployment, or a second execution environment unless compatibility makes one necessary.

## 3. Scope and architecture

The data flow is:

```text
reference images + reference-only camera information
    -> hloc feature extraction/matching -> COLMAP triangulation -> frozen map

query image -> reference-image retrieval -> SuperPoint/LightGlue matching
    -> 2D-3D correspondences -> PyCOLMAP PnP-RANSAC/refinement
    -> pose or explicit no-pose outcome -> confidence features -> accept/reject

ground-truth query pose + recorded outcome -> offline evaluation only
```

Use hloc as an external, pinned dependency. Do not fork it or rewrite feature extraction, matching, retrieval, SfM, or PnP. Start with at most 2,048 local features per image, single-pair matching, and an existing hloc retrieval configuration with top-10 reference candidates. These are pilot defaults, not tuned final settings.

Keep the project flat: one small Python package, one experiment configuration, a few runnable modules for data preparation, localization, confidence, and evaluation, and a short report. Add each module only when its chunk needs it. No plugin registry, generalized dataset framework, service, dashboard, experiment-tracking server, or hyperparameter-search system.

Cache reference features, reference retrieval descriptors, the map, and per-query localization records. Score experiments should reuse those records without rerunning matching. Include the upstream versions, map/split identifiers, preprocessing settings, seeds, and effective solver settings in each run manifest so incompatible caches cannot be reused unnoticed.

### Dataset, map, and leakage rules

Use Chess as the provisional pilot scene. Select any later scene using development evidence and sequence availability, not its final-test score. Repetitive scenes such as Stairs are candidates, not guaranteed sources of useful failures.

Assign four roles before creating corruptions: reference-map images, confidence-fit queries, confidence-calibration queries, and final-evaluation queries. Prefer whole sequences. Use official training sequences for reference/fit/calibration roles and reserve official test sequences for final evaluation. If a scene lacks enough sequences, propose disjoint temporal blocks with exclusion gaps and disclose the weaker separation before proceeding. Do not silently fall back to random adjacent-frame splits.

All variants of an original image retain its role. Subsample video frames with a recorded stride or deterministic rule. Fit preprocessing and regression only on confidence-fit data. Select score thresholds and development decisions using calibration data. No query ground truth enters retrieval, PnP, confidence features, or acceptance decisions.

The initial map strategy is pose-anchored sparse triangulation using reference images and their provided reference poses. This is reuse of reference ground truth for mapping, not estimation of the reference trajectory from scratch. The query problem remains RGB relocalization. Resolve a defensible RGB camera model in Chunk 1: 7-Scenes documents uncalibrated RGB/depth cameras, and its listed depth intrinsics are not automatically valid RGB calibration. Any intrinsics estimation must use reference data only.

Do not blindly use the downloadable SfM model as a leakage-free reference. The hloc 7-Scenes helper filters query images out of an existing model, which does not establish whether those queries influenced its original poses, calibration, or structure. Build from reference-only inputs or establish the supplied artifact's provenance. If pose-anchored triangulation is unsuitable, discuss reference-only SfM with a single reference-derived similarity alignment to the dataset frame. Never align using query poses or align each estimated query separately.

Use the original dataset poses as the declared evaluation ground truth and disclose their tracking/calibration limitations. Do not substitute SfM-derived query poses without explicitly changing and labeling the evaluation protocol. Avoid raw or rendered dense-depth integration in the MVP.

### Preserve real solver outcomes

The inspected hloc `localize_sfm.main` non-clustered path substitutes the nearest reference camera pose when PnP returns no result. That fallback is not a successful pose estimate.

Reuse the lower-level solver helpers or inspect their solver return directly. Record `pose_estimated` versus `no_pose`, including missing retrieval and insufficient correspondences. Never count fallback poses as returned PnP estimates. Programming errors and missing/corrupt assets must surface as run errors, not quietly become `no_pose`. A final run must account for every scheduled query attempt.

## 4. Confidence experiment

Use only measurements available at inference. Compute correspondence statistics on the final solver inlier set, with the inlier-ratio denominator equal to the actual deduplicated 2D-3D correspondence set supplied to PnP. Count the same query-keypoint/3D-point pair only once across reference views.

| Signal | Initial definition |
| --- | --- |
| Inlier support | `log1p(inlier_count)` |
| Inlier ratio | Inliers / solver input correspondences |
| Reprojection residual | `log1p(median final inlier residual in pixels)` at the declared coordinate scale |
| Image coverage | Fraction of cells in a fixed 4 x 4 image grid occupied by distinct inlier query keypoints |
| 3D spread | Two eigenvalue ratios, `lambda_2 / lambda_1` and `lambda_3 / lambda_1`, of centered distinct inlier map points, with descending eigenvalues |

Record useful residual quantiles for analysis, but do not add every possible statistic to the model. Define undefined/degenerate feature handling explicitly; do not silently replace NaNs with plausible confidence.

The eigenvalue ratios describe configuration, not a proof that PnP is unobservable. Planarity alone is not necessarily failure. Viewing-ray spread, depth spread, map track quality, and PnP conditioning are possible follow-ups only if the initial failure analysis identifies a specific need. There is no per-query triangulation step, so do not call these features "triangulation degeneracy."

Compare five scorers on identical saved poses:

1. Inlier count alone.
2. Inlier ratio alone.
3. Median reprojection residual alone, with lower being better.
4. Standardized, L2-regularized logistic regression using the three conventional signals.
5. The same regression configuration with image coverage and the two 3D-spread features added.

Use a fixed initial regularization setting, not a parameter sweep. Fit feature scaling on fit data only. Give each original image equal total training weight across its returned variants so one heavily corrupted image does not dominate. The logistic output is a ranking score; do not claim calibrated safety probabilities.

Comparison 5 versus 4 isolates the additional geometry information. Comparison 5 versus 1 supports the proposed resume bullet. If geometry does not help, report that instead of endlessly adding features.

## 5. Benchmark protocol and metric definitions

### Query conditions

Start with clean images and two severities each of Gaussian blur, brightness attenuation, and a seeded rectangular occlusion. This produces seven attempts per original image, not seven independent originals. Choose and record exact parameters on development data, then freeze them. Brightness attenuation is a controlled darkness proxy, not a physically complete low-light sensor simulation. Do not add corruption combinations in the MVP.

Viewpoint change uses actual held-out camera poses. Report results against reference-relative translation/orientation bins chosen on development data; query ground truth is used only to assign evaluation bins. Viewpoint is a slice of the existing queries, not a synthetic eighth corruption.

For an end-to-end corruption claim, run retrieval and local extraction on each corrupted image. Do not silently reuse the clean query's retrieved candidates. Keep reference maps unchanged.

### Correctness and accounting

Use camera-center translation error in meters and relative-rotation geodesic error in degrees. Convert camera-to-world and world-to-camera conventions explicitly; do not compare extrinsic translation vectors as if they were camera positions.

The proposed primary correctness criterion is translation error <= 0.05 m **and** rotation error <= 5 degrees. Confirm it after the calibration/pose-convention pilot and freeze it before final evaluation. Also report continuous errors and a predeclared gross-error slice, provisionally translation > 0.5 m or rotation > 20 degrees. A marginal threshold miss is not automatically a "dangerously incorrect" pose.

For a declared set of query attempts, let Q be scheduled attempts, P returned PnP poses, K accepted poses, and W incorrect accepted poses. No-pose outcomes are always rejected.

| Metric | Definition and interpretation |
| --- | --- |
| Pose return rate | P / Q |
| Localization success rate | Correct returned poses / Q |
| Rotation/translation error | Median and 90th percentile over returned poses, with P and no-pose count shown |
| Acceptance / coverage | K / Q |
| Accepted-pose accuracy | (K - W) / K |
| Accepted-pose risk | W / K |
| Incorrect acceptances per query | W / Q |
| Failure-detection AUROC | Incorrect versus correct returned poses, with incorrect as the positive class and low confidence as high failure score |

Report conditional AUROC alongside the no-pose rate; do not inflate it by treating obvious no-pose failures as easy score-ranking successes. AUROC is undefined when either returned-pose class is absent. Accepted accuracy/risk is undefined at zero acceptance.

Plot risk against coverage. Compare scorers at the same integer K on the same query set, with stable, label-independent tie-breaking. Consider 50%, 75%, and 90% coverage only where achievable; do not accept no-pose outcomes to reach a target. Matched-K ranking is an offline comparison, not a deployable threshold. Separately report calibration-selected frozen thresholds and the coverage they actually achieve on final data.

At equal K, report absolute risk difference and relative reduction:

```text
X = 100 * (risk_inlier_count - risk_geometry) / risk_inlier_count
```

If the baseline risk is zero, relative reduction is undefined. Show W and K so a dramatic percentage cannot conceal a one-error difference. Negative improvement is a valid result.

Report clean and per-corruption results, per-scene results, and a clearly labeled pooled result. Keep the same condition mix for all scorers. Add lightweight image-group bootstrap intervals for the main AUROC and matched-coverage risk difference, resampling original images with all their variants together. Note residual temporal dependence and the limited number of scenes; do not claim unseen-scene generalization from held-out queries in the same mapped scenes.

### Runtime

Separate offline map construction from online query latency. For online measurements, time query extraction, retrieval, matching, PnP, confidence, and their total on the stated hardware. Declare whether image decoding is included, use warm-up and CUDA synchronization, and report median/p90 with the image size, feature budget, and reference top-k. Reference caches are allowed; cached query features/matches are not allowed in a claimed full-query latency. Report confidence-only overhead separately.

## 6. Actionable implementation chunks

### Chunk 1: Environment, data, and coordinate sanity (about 4-6 hours)

Confirm budget tier and the GPU host's OS/RAM/VRAM/storage. Pin a compatible Python/PyTorch/PyCOLMAP/hloc/LightGlue combination rather than assuming their latest releases work together. Use one environment and download only the pilot scene and required pretrained assets, respecting licenses.

Inspect official sequence lists and define the four data roles. Establish the RGB camera model and map-coordinate strategy. Create a manifest with image IDs, sequence IDs, role, intrinsics provenance, pose convention, and units.

**Exit evidence:** one real image pair passes through extraction/matching on the intended device; known transforms confirm camera-center and rotation-error calculations; reference/query roles are disjoint; a small reference-only triangulation/reprojection probe is geometrically plausible. Save exact versions and commands.

**Pause:** approve the split, calibration/ground-truth caveats, initial feature budget, estimated cost, and Chunk 2. If RGB geometry is not defensible, stop here and discuss a dataset or mapping adjustment.

### Chunk 2: Working one-scene localization and failure audit (about 8-12 hours)

Build and freeze a useful reference-only sparse map. Start with roughly 150-250 reference images and 50-100 development originals, subject to sequence coverage and the pilot cost. Keep fit and calibration query roles separate. Use one retrieval/matcher/PnP configuration and preserve explicit solver outcomes.

Save one record per attempt: original/variant ID, scene/sequence/role, retrieval candidates, solver status, estimated pose, solver correspondence and inlier information, confidence inputs, errors for offline analysis, and stage timings. Keep large correspondence arrays in ignored artifacts and only small summary tables in Git.

Run clean development queries and the bounded corruption pilot. Visually inspect correct poses, no-pose failures, incorrect returned poses, and especially gross/high-inlier failures. Define the "high-inlier" reporting cutoff using development data and freeze it.

**Exit evidence:** sensible clean poses in the declared coordinate frame, complete outcome accounting, and inspectable failure examples. Aim for at least 20 distinct development originals with incorrect returned variants and 20 with correct returned variants, distributed across fit/calibration roles; this is a practical go/no-go heuristic, not a statistical-power guarantee.

**Pause:** show whether errors are mostly no-pose, marginal threshold misses, or genuinely misleading estimates. Approve proceeding only if the confidence question is measurable. Otherwise agree on one bounded adjustment, such as another development scene or different reference coverage. Do not weaken RANSAC solely to manufacture errors, inspect final queries to hunt failures, or automatically expand the benchmark.

### Chunk 3: Confidence model and development comparison (about 6-9 hours)

Implement the six initial features and five scorers. Fit on confidence-fit records only, using returned poses. Select operating thresholds on calibration data. Produce calibration risk-coverage curves, conditional failure AUROC, and several paired examples where scorers disagree.

**Exit evidence:** geometry-versus-conventional ablation on calibration data, explicit error counts, and a defensible explanation of at least one success or limitation. A negative result passes if the experiment is sound. Add only focused checks for feature degeneracy/deduplication, leakage, metric denominators, and tie behavior; do not test upstream neural networks or build a large mocked integration suite.

**Pause:** decide whether the score and hypothesis are ready for final evaluation. Approve the exact final protocol and any second/third scene, or agree to stop at one scene. Added scenes require their own reference/fit/calibration/test roles and development pass before the next freeze; they are not thrown directly into final evaluation.

### Chunk 4: Freeze and run the held-out benchmark (about 5-8 hours, plus compute)

Freeze reference maps, manifests, scorer weights/scaling, solver/retrieval settings, correctness thresholds, corruption parameters/seeds, viewpoint bins, comparison coverages, and reporting definitions. For multiple scenes, use one pooled confidence model by default and report per-scene performance; avoid separate per-scene model tuning.

Run only the approved final queries, keeping their labels out of fitting and operating-point selection. Produce the result table, risk-coverage plot, AUROC with grouped uncertainty, per-condition/scene breakdown, and runtime summary.

**Exit evidence:** every scheduled attempt accounted for, actual N unique final originals and S scenes stated separately from Q variant attempts, fixed-threshold and matched-K results distinguished, and no hidden tuning on final data.

**Pause:** review the results and limitations. Final outcomes may motivate future work, but not retroactive tuning while still calling the same evaluation held out. A substantive method change requires a fresh holdout or an explicitly exploratory label.

### Chunk 5: Portfolio report and handoff (about 3-5 hours)

Write a compact report with a pipeline diagram, the hypothesis, protocol, main result table, risk-coverage plot, and two or three visual cases. Include at least one limitation or failure the geometry score misses. Use dataset-derived images publicly only where the relevant terms permit it; keep restricted assets local.

Document the commands to reproduce preparation, localization, confidence fitting, and evaluation, plus the hardware, upstream versions, data sources, and expected runtime/storage from actual runs. Re-run a small saved subset from the documented path and regenerate the headline table from recorded outputs.

**Exit evidence:** a reader can understand what this project contributed, where the numbers came from, and how to reproduce them without undocumented manual steps. Revise README status and draft factual resume bullets. No fabricated gains and no unsupported "safety guarantee" wording.

**Pause:** present the finished core project. Stop here unless the SIFT stretch goal is explicitly approved.

### Chunk 6: Optional SIFT pipeline comparison (additional 8-15 hours)

Choose SIFT + LightGlue or conventional SIFT matching at approval time and label the pairing explicitly. The baseline is not "SIFT alone." Reuse the same reference-image membership, camera/calibration policy, retrieval policy, query manifests, feature budget, solver settings, and timing boundaries.

Different descriptors generally need feature-specific tracks/maps. If rebuilding tracks, hold the reference camera poses fixed and describe the result as a comparison of complete pipelines with fixed maps per pipeline, not an identical-3D-map extractor-only experiment. A cross-feature map-transfer subsystem is out of scope.

Evaluate confidence baselines within each pipeline. If fitting separate confidence models, give both the same development protocol and disclose it; do not claim cross-pipeline transfer. Compare risk only at shared achievable coverages, and report localization success/runtime alongside confidence results.

**Exit evidence:** one fair comparison table with the changed components and remaining confounders stated. Update the report only if this adds useful evidence rather than breadth for its own sake.

**Pause:** review the comparison and close the project.

## 7. Filling the portfolio metrics

| Placeholder | Source |
| --- | --- |
| N | Unique original final-evaluation query images in the frozen manifest |
| S | Scenes actually included in final evaluation |
| X | Relative reduction in incorrectly accepted poses versus inlier count at the same K, with baseline/geometry counts and absolute risk difference |
| A | Incorrect-returned-pose AUROC on the declared final query conditions, with uncertainty and class counts |
| B | Accepted-pose accuracy at the stated operating point |
| C | Accepted attempts / all scheduled attempts, expressed as a percentage |

A, B, and C must describe the same scorer and declared evaluation population. State whether C comes from offline matched-K ranking or a frozen calibration threshold. If N includes corruptions instead of originals, rewrite the bullet to say "query attempts" and report unique originals separately.

If there are too few wrong returned poses or the score does not improve risk, rewrite the contribution around the evaluated hypothesis and documented failure analysis. Do not retain "reducing incorrectly accepted poses" without supporting results.

## 8. Sources and integration notes

- [hloc overview and reusable modules](https://github.com/cvg/Hierarchical-Localization)
- [hloc 7-Scenes instructions and calibration/depth caveats](https://github.com/cvg/Hierarchical-Localization/blob/master/hloc/pipelines/7Scenes/README.md)
- [7-Scenes reference-model filtering helper](https://github.com/cvg/Hierarchical-Localization/blob/master/hloc/pipelines/7Scenes/utils.py)
- [hloc solver and fallback behavior](https://github.com/cvg/Hierarchical-Localization/blob/master/hloc/localize_sfm.py)
- [LightGlue feature support and pretrained matchers](https://github.com/cvg/LightGlue)
- [7-Scenes data format, camera limitations, and terms](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)

These sources informed the plan on 2026-09-14. Resolve and record compatible upstream commit/version pins during Chunk 1; do not treat moving branch documentation as a reproducible environment.
