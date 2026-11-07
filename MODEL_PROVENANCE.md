# 模型文件与 SHA-256 附录

以下 SHA-256 是 2026-11-07 对**本地实际文件**计算的完整值，用于识别文件字节，不代表已核实来源、许可证或训练时的文件版本。请对自己的模型文件单独计算并与预期版本核对。

| 本地文件 | SHA-256 |
|---|---|
| `anima_baseV10.safetensors` | `bd43b7cffe1ed1153d9c41e7beb2f18cb1273eafbaa3af3edd6a173dc90a006e` |
| `qwen_image_vae.safetensors` | `a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f` |
| `qwen_3_06b_base.safetensors` | `cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba` |
| `lora84_010788.safetensors` | `20ab06331e98cc99d02132a84dbb20c9570c0499bb603ea669a42aa289a7fa24` |

前三项是本机现有兼容模型文件，最后一项是本项目保存的单 LoRA。10788 续训记录中的底模及 VAE 摘要记录处于关闭状态，因此上表不能单独证明本地文件就是当时云端加载的同一份字节。确切上游修订、训练时 tokenizer 版本以及已删除的原始数据集归档摘要均未核实。VGG19 与 StyleShot 兼容评测编码器的具体权重文件也尚未确认，因此暂不填写摘要。

参考来源：[Anima 官方模型卡](https://huggingface.co/circlestone-labs/Anima)、[StyleExpert 官方数据集](https://huggingface.co/datasets/HH-LG/StyleExpert)。文件名与上游卡片的对应关系不应被视为具体修订已确认；使用和再分发前请各自核对许可证。Git 仓库不包含底模与训练集；[10788 LoRA Release](https://github.com/lightlanguage1/anima-single-lora-style-transfer/releases/tag/v0.1.0-10788) 提供本项目权重及配套许可。

在 PowerShell 中对自己的文件计算：

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath 'D:\path\to\model.safetensors'
```

Linux/macOS 可运行 `sha256sum /path/to/model.safetensors` 或 `shasum -a 256 /path/to/model.safetensors`。10788 发布附件的摘要为上表所列值；下载后可自行计算核对。
