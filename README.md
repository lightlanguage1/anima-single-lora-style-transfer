# Anima 参考画风迁移：单 LoRA 的诊断研究

> Reference-Guided Style Transfer with a Single LoRA — an exploratory diagnostic study of style and texture generalization on Anima.

[正式排版的英文论文 PDF](Anima_Single_LoRA_Research.pdf) · [中文论文审阅版](paper_review_zh.md) · [论文 LaTeX 正文](main.tex) · [实验数据说明](data/README.md) · [图片来源记录](data/image_sources.json) · [项目展示页面](index.html)

本项目研究如何在冻结的 Anima 上，通过单个 LoRA，将参考图 A 的画风迁移到内容图 B。当前结果显示参考条件可以改变输出，但笔触和材质的忠实度不稳定。这里同时保留正例、失败案例、训练记录与测量限制；这不是已通过独立基准验证的最终论文。

## 快速开始

1. 获取本仓库，并把 `comfyui_node` 目录完整放入 ComfyUI 的 `custom_nodes/AnimaSingleLoRA/`，保留其中的 `runtime` 与 `native_support`。
2. 在 ComfyUI 自己的 Python 环境安装[节点说明](comfyui_node/README.md)列出的依赖。准备 Anima Base、Qwen Image VAE、Qwen 0.6B 文本编码器，以及本项目训练得到的 10788 LoRA。**模型权重不包含在本仓库，目前没有公开下载地址**；没有兼容 LoRA 时无法复现本文结果。
3. 重启 ComfyUI，导入 [workflow.json](comfyui_node/workflow.json)，选择内容图 B 与画风图 A，填写四个模型文件路径并运行；`project_code_directory` 留空使用随节点附带的兼容代码和 tokenizer。

[详细安装与故障排查](comfyui_node/README.md) · [模型 SHA-256 与来源](MODEL_PROVENANCE.md) · [许可及第三方声明](THIRD_PARTY_NOTICES.md)
## 做了什么

- **训练主线：**9999 → 10788；计划最多 1,000 次更新，日志实际记录 789 次。1024 分辨率、等效 batch 16、rank 32。
- **生成图分析：**10788 的 3 个内容 × 23 张参考 × 2 个种子，共 138 张匹配输出。换参考、换内容、换种子的差异是描述性特征距离，不是泛化成功率。
- **模型审计：**280 个 LoRA 更新矩阵、560 个因子；另有 174 次受控单步前向。输入槽分开不等于内容和画风已经完全解耦。
- **分支与小试：**卷积、动态增量、参考排序和配对纹理的训练都保留了范围与负结果；动态增量 OFF 仍保留旧卷积与 LoRA。
- **算子反例：**特定方形 latent 的转置使相对 RMS 变化约 1.16，统计画风 loss 却接近 0；这证明该反例下的监督盲区，不能单独确定泛化失败的根因。
- **参考目标分解：**同一油画 A 的颜色、方向与局部纹理代理，用单个 LoRA 分别做同时和阶段拟合；同候选预算下都未明显重现厚涂。该单例未证明独立画风基元或泛化能力。

## 方法概览

A 是画风参考，B 是内容，C 是训练时的配对目标。目标状态、B、A 三组 token 经专门的 `SequenceRoute` 进入冻结 Anima；单个 LoRA 作用于 28 个 block 中的 10 种投影。训练中的关键关系为：

```text
z_t       = (1 - t) z_C + t ε
v_target  = ε - z_C
z_hat_0   = z_t - t v_theta
L         = λ_flow L_flow + λ_style L_style
```

画风项比较预测 latent 与配对目标 C 的多尺度 Gram、通道矩和边缘统计；不直接逐块对齐 A 的笔触。0.15 是加权风格梯度与加权 flow 梯度的范数比控制目标，不是固定的画风 loss 权重。完整定义、示意和实验图请看[项目页面](research_showcase.html#method)。

## 仓库内容

```text
index.html                  GitHub Pages 入口
research_showcase.html      研究展示与图像对照
images/                     页面所用内容、画风参考与生成结果
figures/                    论文和研究页面使用的统计图
main.tex                    英文论文草稿与引用
data/                      已脱去本机绝对路径的实验汇总和图像来源
comfyui_node/               ComfyUI 接入代码与说明
training/                   续训脚本、参数模板与复现边界
```

显示的图片保持原始宽高比；历史综合生成时，部分输入曾被强制缩放为方形。本地数据目录提供了完整字段但已去除绝对路径的续训日志，以及评测记录和分析表；未脱敏的原始日志、原始训练集、底模与 LoRA 权重不包含在此准备目录中。因此该目录目前**不能单独复现完整训练**。

## 数据、训练与环境

- **数据来源：**[StyleExpert 官方数据集](https://huggingface.co/datasets/HH-LG/StyleExpert)（数据卡标注 CC BY-NC 4.0）。项目按文件名构造成内容 B、配对目标 C 及同风格不同内容键的参考 A；140/30/37 是训练、验证、测试风格标签数，不等于已完成的人工标注。逐像素跨集验证在原记录中是关闭的，不能当作去重通过。
- **训练方式：**冻结 Anima 底模，只训练一个 rank-32 LoRA；每次更新累计 16 条，画风项使用预测 latent 与 C 的多尺度统计，控制器稀疏测量梯度范数比。详见[训练代码与复现说明](training/README.md)、[参数模板](training/launch_parameters.template.json)和[流程图](figures/training_route.png)。
- **已记录设备：**一次续训使用 NVIDIA RTX 6000D（可见显存 83.05 GiB）、PyTorch 2.12.0+cu130、CUDA 13.0。CPU、操作系统版本、驱动及镜像摘要未留有可靠记录，不能据此声称已完整复现原环境。
- **模型哈希：**[模型来源与哈希附录](MODEL_PROVENANCE.md)及论文附录列出本地 Anima 底模、Qwen VAE、文本编码器和 10788 LoRA 的完整 SHA-256；续训时底模哈希记录被禁用，本地哈希不能单独证明其与云端当时加载的文件一致。
- **复现范围：**脚本与脱敏配置已纳入本目录；节点随附的 native 代码和 tokenizer 仅覆盖推理路径，完整训练依赖、模型、原始数据、样本清单和 9999 完整训练状态仍需另行准备，完整独立复现尚未通过验证。
## ComfyUI 节点

[节点安装、模型路径与工作流说明](comfyui_node/README.md)。节点保留 `target + B + A` 三槽路由与单个项目 LoRA；按 `LoadImage B/A → AnimaNativeStyleTransfer → SaveImage` 连接，不等同于普通 LoRA Loader。

已在本地 ComfyUI 的 PromptExecutor 中，用 10788、种子 20261014、30 步和等比补边生成 1024×1024 图像；RTX 3080 20 GiB 的本次最新执行包含冷加载共约 99.8 秒。这是一次运行记录，不是通用速度或画风质量结论。所需的 Anima 基模、VAE、文本编码器与 LoRA 权重需自行准备；节点附带经过整理的兼容推理代码和 tokenizer。

## 本地查看与 GitHub Pages

直接打开 `index.html` 可查看页面；标题的可选字体来自 Google Fonts，无法连接时会使用系统后备字体。GitHub Pages 可在仓库设置中选择默认分支的仓库根目录作为发布源；是否启用以仓库实际设置为准。

本地 10788 LoRA 文件约 175 MiB，不能直接作为普通 Git 文件推送；仓库的 `.gitignore` 排除了模型权重。若以后决定公开权重，需要另选 Git LFS、Release 或其他托管方式，并核对底模和数据的许可。

## 证据与局限

- 综合评测计划 1,232 条，本地档案记录 600 条；138 条只是其中的匹配分析子集。
- 23 个参考标签各对应一张参考图；底模是否见过这些画风未知，不能将其写成严格封存的未见风格基准。
- 逐步训练数据表覆盖 789 次更新及其 12,624 行已记录样本呈现；25 次稀疏梯度审计不能扩展成逐步真实测量。
- 最新 A 分解对照只有一张 A、一张 B 和一个种子；代理下降不代表笔触迁移成功，原版与拟合组的推理计算量也不同。
- VGG 与 StyleShot 兼容编码器在个别参考上的相对判断相反；它们的定义、尺度和裁剪不同，不能跨代理比较数值。
- 与其他方法尚未按同任务、数据和推理协议完成公平重跑。
- 第三方参考图片来源列于 [data/image_sources.json](data/image_sources.json)；正式公开前需逐项核对图片授权与署名要求。项目原创代码使用 Apache-2.0；模型、数据与图片适用各自的条款，详见[第三方授权说明](THIRD_PARTY_NOTICES.md)。

相关工作：[Gatys 等](https://arxiv.org/abs/1508.06576) · [LoRA](https://arxiv.org/abs/2106.09685) · [Flow Matching](https://arxiv.org/abs/2210.02747) · [StyleShot](https://arxiv.org/abs/2407.01414) · [USO](https://arxiv.org/abs/2508.18966)。
