# Recorded 9999 → 10788 training implementation

This directory contains the three project scripts retained with the 9999-resume run. `launch_parameters.template.json` preserves the recorded hyperparameters while replacing host-specific paths with placeholders. It is an archival reference, **not a ready-to-run environment or a guarantee of exact reproduction**. The original source includes historic `/home/waas/...` defaults; pass explicit arguments and an independent output directory when adapting it.

## Inputs needed before running

- The compatible Anima native `library` package and its dependencies, plus Qwen and T5 tokenizer directories. The original launch used a separate `code_root` containing `library`, `configs/qwen3_06b`, `configs/t5_old` and the style-split manifest. This training directory alone does not contain that full code tree or a validated dependency lockfile.
- Compatible Anima base, Qwen image VAE and Qwen text encoder; record their sources, licenses, versions and file hashes. The matching 9999 **training state** (not only a LoRA file) is needed to reproduce this continuation's optimizer/controller resume.
- The StyleExpert archive and style split manifest. The archived original dataset was deleted after migration; the upstream [dataset card](https://huggingface.co/datasets/HH-LG/StyleExpert) records CC BY-NC 4.0. An upstream re-download is not proven byte-identical to the original archive.
- A CUDA-capable PyTorch environment sized for the job. The run record reports one RTX 6000D with 83.05 GiB visible, PyTorch 2.12.0+cu130 and CUDA 13.0. Its CPU, OS, driver, image digest and full dependency versions were not captured. These observations are not universal minimum requirements.

The run requested 1,000 updates, previews every 100 and checkpoints every 200; 789 updates are retained. Microbatch 1 × effective batch 16, resolution 1024, rank 32, 100-update learning-rate warmup from 1e-5 to 5e-5, seven activation-checkpointed blocks, and the measured sparse gradient-ratio controller are described in the paper. Actual training involved 12,624 recorded sample presentations; this does not mean 12,624 independently verified images or a full pass through the training pool.

## Data construction and checks

The pre-existing archive supplies source/target pairs parsed from names such as `*_src_*` and `extra_*`. The recorded style split has 140/30/37 train/validation/test labels. During training, A is sampled from the same style label as paired target C but a different content key than B/C. A different key is not proof of semantic independence. Ambiguous names, duplicates and orphan records were tracked; the optional pixel-split check was **disabled**, so `passed` in that disabled check is not evidence of image-level deduplication. No additional human annotation or human quality audit is documented. See the paper for exact counts and limitations.

## Adaptation checklist

1. Supply all inputs above, check their upstream licensing and select a clean output directory; preserve the existing checkpoints.
2. Replace every `<SET_LOCAL_PATH>` in the template with real local paths and inspect all settings, including resume-state compatibility. The template is data for a launcher, not a shell command.
3. Build a launcher or translate the JSON keys to the script's CLI options. First inspect the entry point with `python anima_native_style_response_from9280_local.py --help` in the prepared environment. The original native model package, exact dataset revision, and runtime environment must be reconstructed independently.
4. Validate preview production, checkpoint writes, saved step number and sample-level data splits before committing to a long run. These files have not been rerun as an independently packaged training release.

The code's project license and third-party redistribution permissions still require review before publication. See the repository README and the paper's model-identification appendix for SHA-256 values and provenance limits.
