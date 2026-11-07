# Paper evidence tables / 论文数据说明

This directory is a local research export for the manuscript. Counts describe the
available records, not an externally verified benchmark or a completed 1,232-image
evaluation. Source images, training images, base-model weights and LoRA weights
are not included. Publication rights and public-release permissions are not yet
fully cleared.

## Core records

| File | Contents | Scope and cautions |
| --- | --- | --- |
| `study_20261106_training_full.jsonl` | All 789 available update records from step 10000 through 10788, including the 46 top-level fields and nested sample rows present in the original log. | Paths are sanitized. Style-gradient ratios may be estimates or carried forward; consult the audit and estimated flags. |
| `study_20261106_training_steps.csv` | The same 789 updates, restricted to 16 convenient columns. | This is a selected-field table, not the full raw record; use `study_20261106_training_full.jsonl` for all recorded fields. |
| `study_20261106_training_rows.csv` | All 12,624 logged training-row presentations for those updates. | The 46,046 planned pairs are not 46,046 presentations in this continuation. The rows describe the log, not image assets. |
| `study_20261106_evaluation_full.jsonl` | All 600 archived evaluation records, with recorded fields preserved and local paths sanitized. | 600 of 1,232 planned; the output images are retained in the existing local archive, not copied here. |
| `study_20261106_evaluation_records.csv` | The 600 records flattened into a convenient table with an output-exists flag. | Some historical generation inputs were stretched. |
| `study_20261106_evaluation_coverage.csv` | Counts grouped by task and mode. | The combinations are not independent samples. |
| `study_20261106_summary.json` | Independently recomputed counts, label split and observed training ranges. | Label disjointness does not prove semantic separation or no pretraining exposure. |
| `study_20261106_style_split.json` | Original train/validation/test style-label lists, with 140/30/37 exact labels. | Exact label separation alone cannot establish visual or pretraining independence. |
| `study_20261106_archived_status.json` | Status retained with the historical evaluation archive. | A historical snapshot, not evidence that a process is currently running. |
| `study_20261106_training_configuration.json` | The 9999-resume launch configuration with local paths sanitized. | Configured maxima and target ratios are not necessarily actual measured values. |
| `study_20261106_evaluation_protocol.json` | Planned cases, seeds, modes, sampling protocol and sources, with local paths sanitized. | This is a plan, including a historical `planned_not_started` status; use the 600 actual records for completion. |
| `study_20261106_evaluation_environment.json` | Recorded evaluation software, device and checkpoint roles. | A historical environment record; it does not include model weights. |

## Reference decomposition and controls

| File | Contents |
| --- | --- |
| `study_20261106_component_results.json` | Complete two-arm reference-proxy experiment, including all candidate results and limitations. |
| `study_20261106_component_candidates.csv` | All 24 candidate evaluations across joint and staged paths. |
| `study_20261106_component_responses.csv` | 18 finite-difference proxy rows across two paths, three states and three proxies. |
| `study_20261106_internal_paths.json` | Six fixed-reference spectral-intervention paths. |
| `study_20261106_feedback.json` | Three inference-time feedback decisions with all candidates. |
| `study_20261106_uniform_control.json` | The matched late-stage uniform-strength control. |
| `study_20261106_external_paths.json` | Four earlier external-reference switching paths; a distinct experiment. |

These pilots use one content, one target style and one seed per experiment; they
do not establish an unseen-style generalization rate or independent learned style
atoms. Middle-stage pictures are clean-image estimates, not completed images.

## Full diagnostic measurements

- `study_20261106_weights_matrices.csv` and `study_20261106_weights_factors.csv`:
  280 induced-matrix rows, and 1,120 factor rows covering 560 factors for each
  of checkpoints 9999 and 10788.
- `study_20261106_weights_spectra.npz`: 280-by-32 singular-value array and
  factor histograms.
- `study_20261106_activation_conditions.csv`, `study_20261106_activation_layers.csv`,
  `study_20261106_activation_distributions.npz`, and
  `study_20261106_activation_signedlog_histograms.npz`: 174 conditions,
  layer/slot summaries and binned activation data. The original 128-bin
  histogram covers [-16, 16] and excludes 34.1% of values; the separate
  192-bin signed-log histogram covers all recorded values in this probe.
  Their corresponding metadata are `study_20261106_activation_summary.json`
  and `study_20261106_activation_signedlog_summary.json`. Neither contains
  the full unbinned activation tensors.
- `study_20261106_style_vectors.npz`: all 458 image-level vectors with 1,168
  components, their existing projection arrays and uniquely indexed,
  sanitized record labels. These are computed descriptors, not raw images
  or a proven disentangled style embedding.
- `study_20261106_tensor_pair_distances.csv` and `study_20261106_tensor_style_matrix.csv`:
  weighted-VAE-descriptor pair comparisons.
- `study_20261106_vgg19_reference_distribution_pairs.csv` and
  `study_20261106_styleshot_auxiliary_pairs.csv`: individual comparisons from
  two different feature probes. Their units and preprocessing differ.
- `study_20261106_branch_*.json` and `study_20261106_branch_*.jsonl`: the
  available 400/600-update convolution analyses, dynamic ON/OFF review, and
  retained branch-training records.
- `study_20261106_pilot_*.jsonl`: the four 15-update control/treatment logs for
  reference ranking and paired-texture pilots.
- `study_20261106_diagnostic_*.json`: author-reference runs, VAE roundtrip,
  reference-swap, transpose-invariance, caching and high-frequency diagnostics.

The export replaces absolute local paths with basenames. Some files can therefore
lose path distinctions; the 458 vector labels additionally preserve their original
row order and a unique sequential index. See the local evidence index outside the
paper project for the original, unsanitized files and images. The vector and
activation files contain descriptors and aggregates, not original hidden tensors.

`build_evidence.py` regenerates the tables from explicit paths to the archive,
training log, split manifest and local-convolution branch directory. It refuses to overwrite existing exports. It does
not download data, train a model or verify third-party image rights.
