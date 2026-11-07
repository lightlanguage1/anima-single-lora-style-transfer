# Third-party notices and usage boundaries

## Code

Original project-authored code in this repository is made available under [Apache License 2.0](LICENSE), unless a file or directory says otherwise. Bundled third-party code and tokenizer assets retain their own copyright notices and applicable licenses. The code license does not grant rights to third-party models, datasets or images.

The node uses a compatible snapshot of [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts). Its upstream project states that most code uses Apache-2.0 while individual components can use different licenses. Preserve the original notices in each bundled file and follow the accompanying dependency notice; do not assume a blanket license covers every upstream component. Qwen Image VAE integration includes upstream Qwen/Wan and Hugging Face Diffusers code; its source file retains its own notices.

## Models and datasets

- [Anima model card and license](https://huggingface.co/circlestone-labs/Anima#license): CircleStone Labs Non-Commercial License and applicable third-party terms. Its license explicitly includes LoRAs in the definition of derivatives; distribution requires the notices, license copy and other conditions specified by the rights holder. Base weights are not included in the repository. The 10788 LoRA and its specific license notices are provided as a separate Release asset. Do not infer commercial or production use rights from this repository's code license.
- [StyleExpert dataset](https://huggingface.co/datasets/HH-LG/StyleExpert): its dataset card states CC BY-NC 4.0. The source archive is not included; the exact version of the archive used in the historical run was not verified. Observe source attribution, noncommercial restrictions and any applicable third-party rights.
- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B) and [Qwen-Image model card](https://huggingface.co/Qwen/Qwen-Image): consult upstream model cards and applicable file licenses. Local filenames and SHA-256 values do not establish a particular upstream revision.

The distribution rights and terms applicable to a trained LoRA should be assessed independently from the code. In particular, the applicable CircleStone license describes conditions for distributing derivatives; consult the [10788 notice](model_licenses/NOTICE_10788.md), accompanying license texts and any applicable training-data terms. Built on NVIDIA Cosmos.

## Illustrations and research examples

Recorded image source pages are listed in [data/image_sources.json](data/image_sources.json). The Metropolitan Museum of Art provides its designated public-domain works under [its open-access policy](https://www.metmuseum.org/policies/image-resources); Pexels supplies its photos under the [Pexels license](https://www.pexels.com/license/), including restrictions concerning identifiable persons and endorsement. A source page and file name alone do not prove the identity or clearance of every local asset. Generated results may incorporate reference inputs and are subject to applicable rights of others. Image metadata and attribution should be verified individually before any separate redistribution or reuse.

This summary is a navigation aid, not a replacement for the original licenses.

The historical four-case image containing an unverified plant photograph is excluded from this public snapshot. Other original archives remain local. Pexels portraits must not be used to imply the depicted person's endorsement.
