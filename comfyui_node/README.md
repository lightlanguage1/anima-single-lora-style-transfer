# Anima 双参考画风迁移节点

这是研究项目的 ComfyUI 节点：输入内容图 B 和画风参考图 A，调用项目原生的 `target + B + A` 三槽推理及单个 rank-32 LoRA，返回 1024×1024 图像。它不是标准 LoRA Loader，不能仅用普通的 LoRA 加载器替代。

## 安装

1. 将本目录完整放到 ComfyUI 的 `custom_nodes/AnimaSingleLoRA/`，保留 `runtime/` 和 `native_support/`。重启 ComfyUI 后，在 `Anima/Research` 分类找到 **Anima · 内容 B + 画风 A（单 LoRA）**。
2. 默认使用随节点提供的 `native_support/` 模型代码和 tokenizer；[来源与许可说明](native_support/SOURCE.md)记录了本地快照与上游来源的证据边界。原有外部项目代码目录仍可在 `project_code_directory` 填写并覆盖默认位置。
3. 准备兼容的 Anima baseV10、Qwen Image VAE、Qwen 0.6B 文本编码器和项目训练得到的 rank-32 LoRA。[下载 10788 LoRA 与许可附件](https://github.com/lightlanguage1/anima-single-lora-style-transfer/releases/tag/v0.1.0-10788)；底模和原始测试图片不随节点发布。
4. 使用支持 BF16 的 NVIDIA CUDA 设备。项目的模型库还会导入 torchvision、diffusers、einops、OpenCV，文本编码需要 transformers、sentencepiece、accelerate 和 safetensors。节点读取图像使用 Pillow 和 NumPy。在运行 ComfyUI 的 Python 环境中安装 [requirements.txt](requirements.txt) 中缺少的依赖；PyTorch 和 torchvision 应按该 ComfyUI 安装与 CUDA 环境配置，不要盲目替换。

使用 ComfyUI 自带的 Python 执行安装，例如：

```text
<ComfyUI 的 Python> -m pip install -r <本节点目录>/requirements.txt
```

这份列表用于安装缺失的运行库，不锁定所有版本。本机曾验证的组合为 Python 3.10.11、PyTorch 2.6.0+cu124、torchvision 0.21.0+cu124、diffusers 0.38.0、transformers 4.57.6、accelerate 1.0.1、safetensors 0.8.0rc0、einops 0.8.0、NumPy 1.26.4、Pillow 10.3.0、OpenCV 4.8.1.78、sentencepiece 0.1.99；这是环境记录，不保证其他系统能直接安装相同二进制组合。flash-attn、xformers、Triton 不是此路径的必需项。
## 在画布上运行

1. 把 [workflow.json](workflow.json) 拖入 ComfyUI。左上角选内容图 B，左下角选风格图 A，可直接点两个 LoadImage 节点上传各自的图片。
2. 在中间的节点填写底模、VAE、文本编码器、项目 LoRA 的文件路径，将 `project_code_directory` 留空以使用随节点附带的兼容代码及 tokenizer，或填写自己的兼容代码目录；`runtime_directory` 留空即可使用随节点附带的运行时。
3. 选择支持 BF16 的 GPU，点击“运行”。默认使用固定种子 20261014，按 30 步生成。右侧 SaveImage 展示结果，并写入 ComfyUI 的输出目录。

## 节点连线与参数

`LoadImage (B) → content_B`；`LoadImage (A) → style_A`；节点的 `image → SaveImage`。各模型字段可以指向不同目录：

- `base_model`：Anima 底模文件。
- `vae_model`：Qwen Image VAE 文件。
- `text_encoder`：Qwen 0.6B 文本编码器 safetensors。
- `lora_file`：项目专用 rank-32 LoRA 的 safetensors 或其原生 PT 检查点。
- `project_code_directory`：留空使用随附 `native_support/`；若填写，则使用包含 `library/` 和 `configs/` 的兼容外部项目代码目录。
- `runtime_directory`：留空使用本节点自带的运行时；只有有意切换到兼容的本项目运行时才填写。切换运行时代码目录后需重启 ComfyUI。
- `device`：选择存在、支持 BF16 的 CUDA 设备。
- `aspect_ratio`：`pad` 等比缩放并填灰边；`center_crop` 等比缩放并居中裁剪。两种模式均不拉伸。
- `seed`、`steps`：默认 20261014 和 30。训练时固定任务提示词随原生运行时使用，不提供任意提示词输入。
- `reload_models`：修改整数可强制重新加载本节点持有的模型。模型文件的大小或修改时间改变时，运行时缓存和 ComfyUI 输出缓存也会失效；若只修改文件内容且元数据不变，请手动增加此值。

输入目前每路限 1 张图；透明 RGBA 使用中灰合成；输出为 ComfyUI `IMAGE` 和 `run_info`。模型会缓存在当前进程显存中，路径变化或提高 `reload_models` 会卸载本节点的旧实例。模型与推理源码仍为研究版本，视觉质量不作保证。

公开上游文件入口：[Anima base 模型目录](https://huggingface.co/circlestone-labs/Anima/tree/main/split_files/diffusion_models)、[Qwen Image VAE](https://huggingface.co/circlestone-labs/Anima/tree/main/split_files/vae)、[Qwen 0.6B 文本编码器](https://huggingface.co/circlestone-labs/Anima/blob/main/split_files/text_encoders/qwen_3_06b_base.safetensors)。使用前请查看对应模型卡、许可证与版本；不同版本不保证和已有 LoRA 兼容。Anima 官方模型及其衍生模型受非商业许可和适用的 NVIDIA 模型许可约束。底模与编码器不随代码发布；本项目 LoRA 可从 [10788 Release](https://github.com/lightlanguage1/anima-single-lora-style-transfer/releases/tag/v0.1.0-10788) 获取。
## 路径填写示例

请将以下占位值替换为自己的真实文件路径；模型权重（包括项目 LoRA）不随代码提供：

```text
base_model: <Anima-base-safetensors-路径>
vae_model: <Qwen-Image-VAE-safetensors-路径>
text_encoder: <Qwen-0.6B-safetensors-路径>
lora_file: <本项目-rank32-LoRA-safetensors-路径>
project_code_directory: 留空以使用随附的 native_support
runtime_directory: 留空
device: cuda:0
```
[workflow_api.json](workflow_api.json) 供脚本调用。替换模型路径和两张输入图的文件名，将输入图片放入 ComfyUI 的输入目录，再把整张 API 图作为 `{"prompt": ...}` 的值提交到 `POST /prompt`。返回的 `prompt_id` 可用于 `GET /history/<prompt_id>` 查询执行结果和文件名。

2026-11-07 在 Windows、ComfyUI 0.24.1、Python 3.10.11、PyTorch 2.6.0+cu124 和 RTX 3080（20 GiB）上，以 10788 步权重执行了完整四节点工作流。20 GiB 是本次设备的容量。ComfyUI 从执行开始到成功保存用时 99.8 秒，含本进程首次加载模型。输出为 1024×1024 RGB，固定种子和相同输入的输出与已有本机样例逐像素一致。本次结果 PNG 带有恢复画布所需的 `workflow` 元数据。本机样例图保存在节点目录，其他安装可选用自己的内容图和风格图。

另外，以随附 `native_support/` 作为默认项目代码路径，使用相同内容、参考、权重、seed 20261014 和 30 步，在同一台 RTX 3080 上通过 ComfyUI 的进程内 `PromptExecutor` 执行 `LoadImage B/A → 节点 → SaveImage`。本次含加载耗时 78.64 秒，输出 1024×1024，与外部代码路径的前次结果逐像素一致。此测试没有手工验证将 GUI 工作流拖放到画布的操作；运行时间仅供本机参考。
## 证据边界

节点调用本项目已有运行时及现有 LoRA，属于实验性适配。一个输出证明流程可执行，不等于画风、笔触或内容已经完全解耦。`runtime/` 保存本项目运行时所需脚本；随附的最小代码与 tokenizer 快照用于节点推理，不等于完整上游训练库；模型权重和参考图片需要用户自行准备并确认来源和许可。
