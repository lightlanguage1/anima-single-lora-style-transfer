# Local native-support snapshot

This directory contains only the native Python modules used by the bundled inference path, plus the two tokenizer configuration trees. It contains no model weights or original training images. The files were copied without changing their contents from the project's archived cloud code. They are not claimed to identify the exact upstream revision of every library module.

- `library/anima_models.py` retains the original NVIDIA Apache-2.0 notice and upstream code comments. `library/qwen_image_autoencoder_kl.py` retains the copyright and Apache-2.0 notice for Qwen-Image, Wan and Hugging Face, noting its Diffusers/Musubi-Tuner lineage. The library snapshot was obtained alongside the project's sd-scripts-based native implementation; see the sd-scripts upstream and the attached license. sd-scripts states that some portions of the wider project have separate licenses, so its licensing cannot be generalized to every file outside this included subset.
- `configs/anima_tokenizers_SOURCE.txt` records the upstream sd-scripts commit used for the archived Qwen3 and T5 tokenizer configuration assets. This source record refers to the tokenizer assets and does not prove that all bundled library modules come from that exact commit. The upstream repository is https://github.com/kohya-ss/sd-scripts .
- `LICENSE-sd-scripts-Apache-2.0.txt` preserves the project's archived upstream license. The original file-level notices remain in place. No model-weight or dataset license is granted by this notice.

These files cover the project's bundled ComfyUI inference route; they are not a replacement for the complete upstream training library. External Python packages and user-supplied model files are described in the parent README.
