"""Run fixed content/style separation probes against existing Anima LoRAs.

Examples:
  python evaluate_anima_disentanglement.py --mode plan --preview_plan RUN/generalization_preview_plan.json
  python evaluate_anima_disentanglement.py --checkpoint RUN/resume_latest.json --ces_dir /home/waas/shuju/ces
  python evaluate_anima_disentanglement.py --checkpoint OLD.pt NEW.pt --manifest separation_cases.json

No optimizer, model download, checkpoint modification or training-process control.
Inference time: O(checkpoints * seeds * cases * sampling_steps * model_cost).
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from html import escape
import json
from pathlib import Path
import tarfile
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch

import anima_native_style_response_from9280_local as native
from anima_disentanglement import (
    TestCase, add_identity_controls, aggregate_sample_metrics, ces_cases, compare_content_swaps,
    expand_contents, manifest_cases, matched_style_metrics, measure_sample, metric_backend,
    preview_cases, quick_controlled_matrix, summarize_samples, write_plan,
)
from anima_disentanglement_depth import (
    add_source_style_probes, save_separation_report, separation_coverage, summarize_separation,
)

MODEL_DEFAULTS = {
    'code_root': '/home/waas/code', 'base_ckpt': '/home/waas/models/anima_baseV10.safetensors',
    'vae_ckpt': '/home/waas/models/qwen_image_vae.safetensors',
    'text_encoder': '/home/waas/models/qwen_3_06b_base.safetensors',
    'qwen_tokenizer': '/home/waas/code/configs/qwen3_06b',
    't5_tokenizer': '/home/waas/code/configs/t5_old',
    'tar_path': '/home/waas/shuju/StyleExpert/style_expert40k.tar',
    'rank': 32, 'res': 1024,
}


def add_input_arguments(parser: argparse.ArgumentParser) -> None:
    """Declare mutually exclusive data sources and a fixed sampling budget."""
    parser.add_argument('--mode', choices=['plan', 'run'], default='run')
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--preview_plan', type=Path)
    inputs.add_argument('--manifest', type=Path)
    inputs.add_argument('--ces_dir', type=Path)
    parser.add_argument('--content', nargs='*', default=[])
    parser.add_argument('--checkpoint', nargs='+', default=[])
    parser.add_argument('--seeds', nargs='+', type=int, default=[20261014])
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--cross_contents', action='store_true')
    parser.add_argument('--preset', choices=['quick', 'all'], default='quick',
                        help='Preview-plan quick preset uses 2 contents/2 styles/2 refs; manifests and ces retain all rows')
    parser.add_argument('--no_identity', action='store_true')
    parser.add_argument('--separation_probes', action='store_true',
                        help='Also use known paired C images as B style variants; increases inference cases')
    parser.add_argument('--output_dir', type=Path,
                        default=Path('/home/waas/exp/separation_test_' + time.strftime('%Y%m%d_%H%M%S')))


def parse_arguments() -> argparse.Namespace:
    """Validate a standalone evaluation request without initializing CUDA."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_input_arguments(parser)
    parser.add_argument('--run_dir', type=Path, help='Read model locations from an existing training run')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--cpu_threads', type=int, default=4)
    for key in MODEL_DEFAULTS:
        parser.add_argument('--' + key, default=None, type=int if key in ('rank', 'res') else str)
    arguments = parser.parse_args()
    if min(arguments.steps, arguments.cpu_threads) < 1:
        parser.error('steps and cpu_threads must be positive')
    if not arguments.seeds or len(set(arguments.seeds)) != len(arguments.seeds):
        parser.error('seeds must be nonempty and unique')
    if arguments.mode == 'run' and not arguments.checkpoint:
        parser.error('run mode requires --checkpoint; use plan mode to inspect the test budget')
    return arguments


def build_cases(arguments: argparse.Namespace) -> list[TestCase]:
    """Build requested controls without making assumptions about style labels."""
    if arguments.preview_plan:
        cases = preview_cases(json.loads(arguments.preview_plan.read_text(encoding='utf-8')))
        pool = list(cases)
        if arguments.preset == 'quick':
            cases = quick_controlled_matrix(cases)
    elif arguments.manifest:
        payload = json.loads(arguments.manifest.read_text(encoding='utf-8'))
        cases = manifest_cases(payload, arguments.manifest.resolve().parent)
        pool = list(cases)
    else:
        cases = ces_cases(arguments.ces_dir.resolve(), arguments.content)
        pool = list(cases)
    if arguments.cross_contents:
        cases = expand_contents(cases)
    if getattr(arguments, 'separation_probes', False):
        cases = add_source_style_probes(cases, pool)
    if not arguments.no_identity:
        cases = add_identity_controls(cases)
    if not cases:
        raise ValueError('No evaluation cases selected')
    return cases


def model_arguments(arguments: argparse.Namespace) -> SimpleNamespace:
    """Reuse trained model locations, allowing explicit local overrides."""
    settings = dict(MODEL_DEFAULTS)
    if arguments.run_dir:
        for name in ('resume_arguments_84g.json', 'arguments_84g.json'):
            path = arguments.run_dir / name
            if path.is_file():
                inherited = json.loads(path.read_text())
                settings.update({key: inherited[key] for key in settings if key in inherited})
                break
    for key in settings:
        explicit = getattr(arguments, key)
        if explicit is not None:
            settings[key] = explicit
    if settings['rank'] != 32 or settings['res'] != 1024:
        raise ValueError('This evaluator uses the current rank32/1024 Anima route')
    return SimpleNamespace(**settings)


class ImageAssets:
    """Read only requested files or tar members; keep test latents in CPU RAM."""

    def __init__(self, settings: SimpleNamespace) -> None:
        self.settings = settings
        self.archive: tarfile.TarFile | None = None
        self.images: dict[str, Image.Image] = {}
        self.latents: dict[str, torch.Tensor] = {}

    def image(self, name: str) -> Image.Image:
        """Load a requested image with the training run's EXIF/RGB/square policy."""
        if name not in self.images:
            if name.startswith('tar:'):
                if self.archive is None:
                    self.archive = tarfile.open(self.settings.tar_path, 'r:*')
                stream = self.archive.extractfile(name[4:])
                if stream is None:
                    raise FileNotFoundError(name)
                with stream:
                    self.images[name] = self.prepare_image(stream)
            else:
                with Path(name).open('rb') as stream:
                    self.images[name] = self.prepare_image(stream)
        return self.images[name]

    def prepare_image(self, stream: Any) -> Image.Image:
        """Apply the existing image transform using Pillow functions."""
        with Image.open(stream) as image:
            return ImageOps.exif_transpose(image).convert('RGB').resize(
                (self.settings.res, self.settings.res), Image.Resampling.BICUBIC)

    def latent(self, name: str) -> torch.Tensor:
        """Return an already encoded CPU latent; no inference side effects."""
        return self.latents[name]

    def close(self) -> None:
        """Release the optional archive without touching its contents."""
        if self.archive is not None:
            self.archive.close()


def configure_tokenizers(context: SimpleNamespace) -> None:
    """Load the same local tokenizers and task prompt as the training code."""
    from transformers import AutoTokenizer, T5TokenizerFast
    settings = context.args
    context.qt = AutoTokenizer.from_pretrained(settings.qwen_tokenizer, local_files_only=True)
    if context.qt.pad_token is None:
        context.qt.pad_token = context.qt.eos_token
    directory = Path(settings.t5_tokenizer)
    context.tt = T5TokenizerFast(vocab_file=str(directory / 'spiece.model'),
                                tokenizer_file=str(directory / 'tokenizer.json'))
    native.encode_text(context)


def configure_vae(context: SimpleNamespace) -> None:
    """Load the existing official VAE with its native latent normalization."""
    from library.qwen_image_autoencoder_kl import load_vae
    context.vae = load_vae(context.args.vae_ckpt, device='cpu', disable_mmap=True)
    context.vae.requires_grad_(False).eval().to(device=context.device, dtype=torch.bfloat16)
    context.mean = torch.tensor(context.vae.latents_mean, device=context.device).view(1, 16, 1, 1, 1)
    context.std = torch.tensor(context.vae.latents_std, device=context.device).view(1, 16, 1, 1, 1)
    if not torch.isfinite(context.std).all() or not (context.std > 0).all():
        raise ValueError('Invalid VAE latent normalization')


def read_adapter(checkpoint: str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load native PT/safetensors/JSON-pointer weights without an optimizer or sidecar."""
    path = Path(checkpoint).resolve()
    if path.suffix == '.json':
        pointer = json.loads(path.read_text())
        path = (path.parent / pointer['state_file']).resolve()
    if path.suffix == '.safetensors':
        from safetensors import safe_open
        from safetensors.torch import load_file
        with safe_open(str(path), framework='pt', device='cpu') as handle:
            metadata = handle.metadata() or {}
        return load_file(str(path), device='cpu'), {'path': str(path), 'step': metadata.get('step')}
    state = torch.load(path, map_location='cpu', weights_only=True)
    return state['adapter'], {'path': str(path), 'step': state.get('step')}


def copy_adapter(named: dict[str, torch.Tensor], weights: dict[str, torch.Tensor]) -> None:
    """Replace all adapter tensors so checkpoint comparisons cannot accumulate weights."""
    if set(named) != set(weights):
        raise ValueError('Checkpoint adapter tensor names do not match the current route')
    with torch.no_grad():
        for name, parameter in named.items():
            value = weights[name]
            if value.shape != parameter.shape or not torch.isfinite(value).all():
                raise ValueError('Invalid adapter tensor: ' + name)
            parameter.copy_(value.to(parameter))
            parameter.requires_grad_(False)


class AnimaTestRuntime:
    """Offline frozen inference using the exact training route and sampler."""

    def __init__(self, arguments: argparse.Namespace, output: Path) -> None:
        settings = model_arguments(arguments)
        device = torch.device(arguments.device)
        if device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('Full inference requires CUDA; plan mode and unit tests run on CPU')
        torch.cuda.set_device(device)
        library, attention = native.import_native(settings.code_root)
        base, _ = native.load_base(settings, library, device)
        self.context = SimpleNamespace(args=settings, out=output, device=device, base=base)
        configure_vae(self.context)
        configure_tokenizers(self.context)
        self.context.route = native.SequenceRoute(base, attention)
        self.context.route.use_checkpoint = False
        self.assets = ImageAssets(settings)
        self.steps = arguments.steps
        self.adapter_scope = native.lora_scope(base, settings.rank)
        self.named = self.adapter_scope.__enter__()

    @torch.inference_mode()
    def encode_assets(self, cases: list[TestCase]) -> None:
        """Encode only selected B/A/C images; never stage a training dataset."""
        names = sorted({name for case in cases for name in (case.source, case.reference, case.target) if name})
        context = self.context
        for name in names:
            pixels = np.array(self.assets.image(name), dtype=np.float32, copy=True)
            tensor = torch.from_numpy(pixels).permute(2, 0, 1)[None, :, None].to(context.device) / 127.5 - 1
            with native.amp(context.device):
                encoded = context.vae.encode(tensor.to(torch.bfloat16))
            posterior = encoded['latent_dist'] if isinstance(encoded, dict) else encoded.latent_dist
            latent = ((posterior.mode().float() - context.mean) / context.std).to(torch.bfloat16)
            if not torch.isfinite(latent).all():
                raise FloatingPointError('Nonfinite input latent: ' + name)
            self.assets.latents[name] = latent.detach().cpu()

    def load(self, checkpoint: str) -> dict[str, Any]:
        """Load one existing checkpoint; no optimizer or checkpoint writes."""
        weights, metadata = read_adapter(checkpoint)
        copy_adapter(self.named, weights)
        return metadata

    @torch.inference_mode()
    def generate(self, case: TestCase, seed: int) -> tuple[torch.Tensor, Image.Image]:
        """Use identical noise for all references with the requested seed."""
        context = self.context
        content = self.assets.latent(case.source).to(context.device)
        reference = self.assets.latent(case.reference).to(context.device)
        output = native.sample(lambda noisy, timestep, text: context.route(
            noisy, timestep, text, [content, reference]), content, context.context['task'], seed, self.steps)
        image = native.tensor_image(native.decode_tensor(context, output))
        return output.detach().cpu(), image

    def close(self) -> None:
        """Release only this evaluator's model and file handles."""
        self.adapter_scope.__exit__(None, None, None)
        self.assets.close()


def render_overview(directory: Path, samples: list[dict], assets: ImageAssets) -> None:
    """Render B/C/A/output rows; show a missing C as missing rather than copying B."""
    tile, row_height = 224, 262
    for seed in dict.fromkeys(sample['seed'] for sample in samples):
        selected = [sample for sample in samples if sample['seed'] == seed]
        canvas = Image.new('RGB', (tile * 4, row_height * len(selected)), 'white')
        drawing = ImageDraw.Draw(canvas)
        for row, sample in enumerate(selected):
            case = sample['case']
            drawing.text((4, row * row_height), f'{case.content} | {case.style} | seed {seed}'[:120], fill='black')
            pictures = [assets.image(case.source), assets.image(case.target) if case.target else None,
                        assets.image(case.reference), sample['image']]
            for column, (label, picture) in enumerate(zip(['B content', 'C target / unavailable', 'A reference', 'output'], pictures)):
                drawing.text((column * tile + 4, row * row_height + 18), label, fill='black')
                if picture is not None:
                    canvas.paste(picture.resize((tile, tile), Image.Resampling.LANCZOS), (column * tile, row * row_height + 38))
        canvas.save(directory / f'overview_seed_{seed}.png')


def save_metrics(directory: Path, samples: list[dict], summary: dict) -> None:
    """Write raw diagnostic values and a blank human-review sheet."""
    records = [{**asdict(sample['case']), 'seed': sample['seed'], 'checkpoint': sample['checkpoint'],
                'output': sample['output'], **sample['metrics']} for sample in samples]
    native.write_json(directory / 'metrics.json', {'summary': summary, 'records': records})
    columns = list(dict.fromkeys(key for record in records for key in record))
    with (directory / 'metrics.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else value
                          for key, value in record.items()} for record in records)
    with (directory / 'visual_review.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        columns = ['output', 'content_preserved', 'style_matches_A', 'reference_content_leakage', 'notes']
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({'output': sample['output']} for sample in samples)


def summary_table(summary: dict) -> str:
    """Show separate diagnostic dimensions without inventing a pass threshold."""
    labels = {'mean_content_layout_ssim_proxy': '内容布局相似度（灰度 SSIM）',
              'mean_content_over_reference_layout_advantage': '更接近内容图还是参考图的布局',
              'mean_matched_style_group_advantage_proxy': '正确画风组相对其他画风组的优势',
              'mean_matched_target_advantage_proxy': '正确配对目标相对其他目标的优势',
              'between_style_response_mean': '换画风时的输出变化',
              'within_style_reference_sensitivity_mean': '同画风换参考内容时的输出变化'}
    rows = ''.join(f'<tr><td>{label}</td><td>{summary.get(key) if summary.get(key) is not None else "缺少对应数据"}</td></tr>'
                   for key, label in labels.items())
    return '<table border="1" cellpadding="8">' + rows + '</table>'


def save_report(output: Path, entries: list[dict], plan: dict) -> None:
    """Build an offline report with image comparisons and explicit diagnostic limits."""
    sections = []
    for entry in entries:
        images = ''.join(f'<a href="{escape(entry["directory"])}/overview_seed_{seed}.png">'
                         f'<img src="{escape(entry["directory"])}/overview_seed_{seed}.png"></a>' for seed in plan['seeds'])
        sections.append(f'<h2>{escape(entry["checkpoint"])}</h2>{summary_table(entry["summary"])}{images}'
                        f'<p><a href="{escape(entry["directory"])}/separation_depth.html">画风 / 内容解耦深度报告（四组干扰测试）</a></p>'
                        f'<p><a href="{escape(entry["directory"])}/metrics.csv">Metrics CSV</a> · '
                        f'<a href="{escape(entry["directory"])}/visual_review.csv">Visual review sheet</a></p>')
    description = ('本报告固定种子、提示词和采样设置，检查：同内容换画风、同画风换参考内容、同参考换内容，以及 A=B 自身参考对照。'
                   '布局相似度是粗略的灰度结构指标；画风距离使用 latent 统计。应结合图片检查人物、姿势和背景是否被参考图带跑。'
                   '不同画风之间发生变化不等于迁移正确；同画风的变化很小也可能是忽略了参考。C 列留空表示没有配对目标，不会拿 B 冒充 C。'
                   '指标不自动判定“解耦成功”。')
    document = '<!doctype html><meta charset="utf-8"><title>Anima 内容—画风分离测试</title>'
    document += '<style>body{font:16px system-ui;max-width:1300px;margin:30px auto;padding:16px}img{max-width:100%;display:block}pre{white-space:pre-wrap}</style>'
    document += '<h1>Anima 内容—画风分离测试</h1><p>' + description + '</p>'
    document += '<pre>' + escape(json.dumps(plan['coverage'], indent=2, ensure_ascii=False)) + '</pre>'
    (output / 'report.html').write_text(document + ''.join(sections), encoding='utf-8')


class EvaluationRunner:
    """Run controlled samples, preserving every image before aggregate diagnostics."""

    def __init__(self, runtime: Any, cases: list[TestCase], arguments: argparse.Namespace) -> None:
        self.runtime, self.cases, self.arguments = runtime, cases, arguments

    def sample_case(self, case: TestCase, settings: tuple) -> dict:
        """Save one sampled image immediately and collect its diagnostic proxies."""
        directory, checkpoint, seed, position = settings
        latent, image = self.runtime.generate(case, seed)
        filename = f'output_{position:04d}_seed_{seed}.png'
        image.save(directory / filename)
        sample = {'case': case, 'checkpoint': checkpoint, 'seed': seed,
                  'latent': latent, 'image': image, 'output': filename}
        sample['metrics'] = measure_sample(sample, self.runtime.assets, native._multiscale_style_loss)
        sample['metrics'].update(matched_style_metrics(sample, self.cases, self.runtime.assets, native._multiscale_style_loss))
        native.append_json(directory / 'progress.jsonl', {'case': asdict(case), 'seed': seed, 'output': filename})
        print(f'[sample] {checkpoint} {position + 1} B={case.content} style={case.style} seed={seed}', flush=True)
        return sample

    def run_checkpoint(self, index: int, checkpoint: str) -> dict[str, Any]:
        """Evaluate each checkpoint with exactly the same cases and noise seeds."""
        directory = self.arguments.output_dir / f'checkpoint_{index:02d}'
        directory.mkdir()
        metadata = self.runtime.load(checkpoint)
        native.write_json(directory / 'checkpoint.json', metadata)
        samples = []
        for seed in self.arguments.seeds:
            for case in self.cases:
                samples.append(self.sample_case(case, (directory, checkpoint, seed, len(samples))))
        originals = [sample for sample in samples if sample['case'].source_variant == 'original']
        summary = summarize_samples(originals, native._multiscale_style_loss)
        summary.update(aggregate_sample_metrics(originals))
        summary['legacy_summary_scope'] = 'original B inputs only; source variants are in separation_depth'
        summary['content_swap_pairs'] = compare_content_swaps(originals, self.runtime.assets)
        separation = summarize_separation(samples, self.runtime.assets, native._multiscale_style_loss)
        summary['separation_depth'] = {key: value for key, value in separation.items() if key != 'pairs'}
        save_separation_report(directory, separation, samples, self.runtime.assets)
        render_overview(directory, samples, self.runtime.assets)
        save_metrics(directory, samples, summary)
        return {'checkpoint': checkpoint, 'directory': directory.name, 'summary': summary}


def main() -> int:
    """Plan on CPU or run a standalone, frozen inference comparison."""
    arguments = parse_arguments()
    torch.set_num_threads(arguments.cpu_threads)
    cases = build_cases(arguments)
    arguments.output_dir = arguments.output_dir.resolve()
    arguments.output_dir.mkdir(parents=True, exist_ok=False)
    settings = {'seeds': arguments.seeds, 'steps': arguments.steps, 'checkpoints': arguments.checkpoint or ['not_selected'],
                'prompt': 'Preserve the content of the first reference image and transfer the artistic style of the second reference image.'}
    plan = write_plan(arguments.output_dir / 'test_plan.json', cases, settings)
    plan['coverage']['factor_specific_pairs_per_seed'] = separation_coverage(cases)
    native.write_json(arguments.output_dir / 'test_plan.json', plan)
    print(json.dumps({'coverage': plan['coverage'], 'output_count': plan['output_count']}, ensure_ascii=False, indent=2), flush=True)
    if arguments.mode == 'plan':
        return 0
    metric_backend()
    runtime = AnimaTestRuntime(arguments, arguments.output_dir)
    try:
        runtime.encode_assets(cases)
        runner = EvaluationRunner(runtime, cases, arguments)
        entries = [runner.run_checkpoint(index, checkpoint) for index, checkpoint in enumerate(arguments.checkpoint)]
        save_report(arguments.output_dir, entries, plan)
        native.write_json(arguments.output_dir / 'result.json', {'status': 'COMPLETED_REQUIRES_IMAGE_REVIEW', 'results': entries})
    finally:
        runtime.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
