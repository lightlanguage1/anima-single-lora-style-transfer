"""Controlled content/style test plans and diagnostics.

Planning is O(cases); pairwise diagnostics are O(samples²). No training occurs.
All distances are diagnostic proxies and require inspection of the saved images.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from itertools import combinations
from functools import lru_cache
from pathlib import Path, PurePosixPath, PureWindowsPath
from statistics import mean
from typing import Any, Callable, Sequence
import json
import sys

from PIL import Image, ImageFilter
import numpy as np
import torch
from torch import Tensor

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
LOCAL_METRIC_DEPENDENCIES = Path(__file__).with_name('_deps')
if LOCAL_METRIC_DEPENDENCIES.is_dir():
    sys.path.insert(0, str(LOCAL_METRIC_DEPENDENCIES))


@dataclass(frozen=True)
class TestCase:
    """One fixed B/A pairing; target is absent for unpaired external images."""
    content: str
    source: str
    style: str
    reference: str
    target: str | None = None
    kind: str = 'external'
    control: str = 'style'
    source_variant: str = 'original'
    reference_content: str | None = None
    variant_evidence: str | None = None
    source_style: str | None = None


def is_absolute_asset(value: str) -> bool:
    """Recognize cloud and Windows absolute paths when planning on either host."""
    return PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()


def resolve_asset(value: str, directory: Path) -> str:
    """Resolve external paths while preserving explicit tar member identifiers."""
    if value.startswith('tar:'):
        return value
    if is_absolute_asset(value):
        return value
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else (directory / path).resolve())


def preview_cases(payload: dict[str, Any]) -> list[TestCase]:
    """Reuse existing split labels and references without loading the dataset."""
    selected = payload.get('cases', payload.get('preview', []))
    if not selected:
        raise ValueError('Preview plan contains no cases or preview rows')
    asset = lambda name: name if name.startswith('tar:') or is_absolute_asset(name) else 'tar:' + name
    return [TestCase(row['B'], asset(row['source']), row['style'], asset(reference),
                     asset(row['target']) if row.get('target') else None, row['kind'])
            for row in selected for reference in dict.fromkeys(row['refs'])]


def manifest_cases(payload: dict[str, Any], directory: Path) -> list[TestCase]:
    """Build the full content/style/reference matrix declared by the user."""
    contents, styles = payload.get('contents', []), payload.get('styles', [])
    if not contents or not styles:
        raise ValueError('Manifest requires nonempty contents and styles lists')
    if len({entry['id'] for entry in contents}) != len(contents):
        raise ValueError('Content ids must be unique')
    if len({entry['id'] for entry in styles}) != len(styles):
        raise ValueError('Style ids must be unique')
    return [case for content in contents for style in styles
            for case in manifest_pair_cases(content, style, directory)]


def manifest_pair_cases(content: dict, style: dict, directory: Path) -> list[TestCase]:
    """Expand explicitly labelled source variants and reference content identities."""
    references = style.get('refs', [])
    if not references:
        raise ValueError(f'Style {style["id"]} has no references')
    variants = [{'id': 'original', 'path': content['path']}] + content.get('variants', [])
    if len({variant['id'] for variant in variants}) != len(variants):
        raise ValueError(f'Duplicate source variant ids for {content["id"]}')
    target = content.get('targets', {}).get(style['id'])
    cases = []
    for variant in variants:
        for reference in references:
            entry = {'path': reference} if isinstance(reference, str) else reference
            cases.append(TestCase(content['id'], resolve_asset(variant['path'], directory),
                style['id'], resolve_asset(entry['path'], directory),
                resolve_asset(target, directory) if target else None,
                kind=style.get('kind', 'external'),
                source_variant=variant['id'], reference_content=entry.get('content_id'),
                variant_evidence=variant.get('evidence', 'declared_same_content') if variant['id'] != 'original' else None,
                source_style=variant.get('style', content.get('source_style'))))
    return list(dict.fromkeys(cases))


def ces_cases(directory: Path, content_names: Sequence[str]) -> list[TestCase]:
    """Use only ces files; never guess which references share the same style."""
    images = sorted(path.resolve() for path in directory.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    selected = [resolve_asset(name, directory) for name in content_names]
    if not selected:
        default = directory / 'QQ20241205-000623.png'
        if not default.is_file():
            raise ValueError('Specify --content filenames for this ces directory')
        selected = [str(default.resolve())]
    available = {str(path) for path in images}
    if not set(selected) <= available:
        raise ValueError('All --content files must be images inside ces_dir')
    references = [str(path) for path in images if str(path) not in selected]
    if not references:
        raise ValueError('No external style references remain after selecting content')
    return [TestCase(Path(source).stem, source, Path(reference).stem, reference)
            for source in selected for reference in references]


def expand_contents(cases: Sequence[TestCase]) -> list[TestCase]:
    """Cross existing references with every content; do not invent paired C targets."""
    sources = {(case.content, case.source): case for case in cases}
    references = {(case.style, case.reference): case for case in cases}
    targets = {(case.content, case.source, case.style): case for case in cases}
    result = []
    for (content, source), source_case in sorted(sources.items()):
        for (style, reference), reference_case in sorted(references.items()):
            template = targets.get((content, source, style),
                replace(source_case, target=None, kind='crossed_unpaired'))
            result.append(replace(template, style=style, reference=reference,
                                  reference_content=reference_case.reference_content))
    return result


def quick_controlled_matrix(cases: Sequence[TestCase]) -> list[TestCase]:
    """Select up to 2 contents × 2 styles × 2 refs with matched style controls."""
    groups = defaultdict(list)
    for case in cases:
        groups[(case.content, case.source)].append(case)
    eligible = [key for key, rows in groups.items() if len({case.style for case in rows}) >= 2]
    if not eligible:
        return list(cases)
    primary = min(eligible, key=lambda key: (not any(case.kind.startswith('seen_') for case in groups[key]), key))
    others = sorted((key for key in groups if key != primary),
                    key=lambda key: (all(case.kind == 'train_anchor' for case in groups[key]), key))
    content_keys = [primary] + others[:1]
    styles = sorted({case.style for case in groups[primary]})[:2]
    references = {style: sorted({case.reference for case in groups[primary] if case.style == style})[:2]
                  for style in styles}
    selected: list[TestCase] = []
    targets = {(case.content, case.source, case.style): case for case in cases}
    for content, source in content_keys:
        for style in styles:
            template = targets.get((content, source, style), TestCase(content, source, style, '', kind='crossed_unpaired'))
            selected.extend(replace(template, reference=reference) for reference in references[style])
    return selected


def add_identity_controls(cases: Sequence[TestCase]) -> list[TestCase]:
    """Add A=B controls to detect unnecessary changes under a self-reference."""
    result = list(cases)
    for content, source in sorted({(case.content, case.source) for case in cases}):
        template = next(case for case in cases if (case.content, case.source) == (content, source))
        result.append(replace(template, style='__identity__', reference=source, target=source,
                              kind='self_reference_control', control='identity', reference_content=content))
    return result


def plan_coverage(cases: Sequence[TestCase]) -> dict[str, Any]:
    """Report which controlled comparisons are supported by the supplied images."""
    ordinary = [case for case in cases if case.control == 'style']
    by_content, by_style, by_reference = defaultdict(set), defaultdict(set), defaultdict(set)
    for case in ordinary:
        by_content[(case.content, case.source)].add(case.style)
        by_style[(case.content, case.source, case.style)].add(case.reference)
        by_reference[(case.style, case.reference)].add((case.content, case.source))
    return {
        'cases_per_seed': len(cases), 'style_cases': len(ordinary),
        'identity_controls': len(cases) - len(ordinary),
        'paired_target_cases': sum(case.target is not None for case in ordinary),
        'contents_with_multiple_styles': sum(len(values) > 1 for values in by_content.values()),
        'same_style_reference_groups': sum(len(values) > 1 for values in by_style.values()),
        'references_shared_across_contents': sum(len(values) > 1 for values in by_reference.values()),
        'missing_same_style_test': not any(len(values) > 1 for values in by_style.values()),
        'missing_content_swap_test': not any(len(values) > 1 for values in by_reference.values()),
        'style_labels': 'Provided dataset/manifest labels; ces filenames are separate styles, not inferred labels',
    }


def write_plan(path: Path, cases: Sequence[TestCase], settings: dict[str, Any]) -> dict[str, Any]:
    """Write the exact shared-seed sampling budget before initializing any model."""
    plan = {'cases': [asdict(case) for case in cases], 'coverage': plan_coverage(cases), **settings}
    plan['output_count'] = len(cases) * len(settings['seeds']) * len(settings['checkpoints'])
    path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
    return plan


def luminance_tensor(image: Image.Image) -> Tensor:
    """Prepare a coarse grayscale layout proxy, not a semantic content embedding."""
    image = image.convert('L').resize((128, 128), Image.Resampling.LANCZOS)
    image = image.filter(ImageFilter.GaussianBlur(radius=1.5))
    pixels = np.array(image, dtype=np.float32, copy=True) / 255.0
    return torch.from_numpy(pixels)[None, None]


@lru_cache(maxsize=1)
def metric_backend() -> Callable:
    """Resolve the official metric before GPU inference; never download dependencies."""
    from torchmetrics.functional.image import structural_similarity_index_measure
    return structural_similarity_index_measure


def layout_similarity(first: Image.Image, second: Image.Image) -> float:
    """Measure coarse grayscale SSIM using the official TorchMetrics implementation."""
    return float(metric_backend()(luminance_tensor(first), luminance_tensor(second), data_range=1.0))


def structure_metrics(images: tuple[Image.Image, Image.Image, Image.Image]) -> dict[str, float]:
    """Compare layout against B/A; the scores do not certify semantic preservation."""
    output, content, reference = images
    content_score = layout_similarity(output, content)
    reference_score = layout_similarity(output, reference)
    return {'content_layout_ssim_proxy': content_score,
            'reference_layout_ssim_proxy': reference_score,
            'content_over_reference_layout_advantage': content_score - reference_score}


def style_distance(first: Tensor, second: Tensor, distance: Callable) -> float:
    """Run the existing latent style diagnostic safely on CPU BF16 outputs."""
    return float(distance(first.float(), second.float()).mean())


def measure_sample(sample: dict, assets: Any, distance: Callable) -> dict[str, Any]:
    """Separate optional paired-target scores from unpaired external diagnostics."""
    case, output = sample['case'], sample['latent']
    metrics = structure_metrics((sample['image'], assets.image(case.source), assets.image(case.reference)))
    metrics['reference_style_distance_proxy'] = style_distance(output, assets.latent(case.reference), distance)
    metrics['content_style_distance_proxy'] = style_distance(output, assets.latent(case.source), distance)
    metrics['paired_target_style_distance'] = None
    metrics['paired_target_latent_mse'] = None
    if case.target is not None:
        target = assets.latent(case.target)
        metrics['paired_target_style_distance'] = style_distance(output, target, distance)
        metrics['paired_target_latent_mse'] = float((output.float() - target.float()).square().mean())
    return metrics


def matched_style_metrics(sample: dict, cases: Sequence[TestCase], assets: Any,
                          distance: Callable) -> dict[str, Any]:
    """Check matched versus alternate style groups and paired C when available."""
    case = sample['case']
    if case.control != 'style':
        return {}
    references = defaultdict(set)
    targets = {}
    for candidate in cases:
        if candidate.control == 'style':
            references[candidate.style].add(candidate.reference)
            if candidate.content == case.content and candidate.source == case.source and candidate.target:
                targets[candidate.style] = candidate.target
    scores = {style: mean(style_distance(sample['latent'], assets.latent(path), distance) for path in paths)
              for style, paths in references.items()}
    alternatives = [value for style, value in scores.items() if style != case.style]
    target_alternatives = [style_distance(sample['latent'], assets.latent(path), distance)
                           for style, path in targets.items() if style != case.style]
    correct_target = style_distance(sample['latent'], assets.latent(case.target), distance) if case.target else None
    return {'style_group_distances_proxy': scores, 'nearest_style_group_proxy': min(scores, key=scores.get),
            'matched_style_group_advantage_proxy': min(alternatives) - scores[case.style] if alternatives else None,
            'matched_target_advantage_proxy': min(target_alternatives) - correct_target
                if target_alternatives and correct_target is not None else None}


def compare_content_swaps(samples: Sequence[dict], assets: Any) -> list[dict[str, Any]]:
    """Compare B swaps while holding the actual A, checkpoint and noise fixed."""
    groups = defaultdict(list)
    for sample in samples:
        case = sample['case']
        if case.control == 'style':
            groups[(sample['checkpoint'], sample['seed'], case.style, case.reference)].append(sample)
    records = []
    for group in groups.values():
        for first, second in combinations(group, 2):
            if first['case'].source == second['case'].source:
                continue
            first_advantage = first['metrics']['content_layout_ssim_proxy'] - layout_similarity(
                first['image'], assets.image(second['case'].source))
            second_advantage = second['metrics']['content_layout_ssim_proxy'] - layout_similarity(
                second['image'], assets.image(first['case'].source))
            records.append({'seed': first['seed'], 'style': first['case'].style,
                'reference': first['case'].reference, 'content_a': first['case'].content,
                'content_b': second['case'].content,
                'matched_content_layout_advantage_mean': (first_advantage + second_advantage) / 2})
    return records


def compare_outputs(first: dict, second: dict, distance: Callable) -> dict[str, Any]:
    """Measure reference response on a genuinely shared content/checkpoint/seed."""
    first_case, second_case = first['case'], second['case']
    matched = (first_case.content, first_case.source, first['seed'], first['checkpoint'])
    other = (second_case.content, second_case.source, second['seed'], second['checkpoint'])
    if matched != other:
        raise ValueError('Response comparison requires identical content, checkpoint and seed')
    category = 'same_style_different_reference' if first_case.style == second_case.style else 'different_style'
    return {'kind': category, 'content': first_case.content, 'seed': first['seed'],
            'style_a': first_case.style, 'style_b': second_case.style,
            'reference_a': first_case.reference, 'reference_b': second_case.reference,
            'output_style_distance_proxy': style_distance(first['latent'], second['latent'], distance),
            'output_latent_mse': float((first['latent'].float() - second['latent'].float()).square().mean())}


def summarize_samples(samples: Sequence[dict], distance: Callable) -> dict[str, Any]:
    """Contrast between-style response and within-style reference-content sensitivity."""
    groups = defaultdict(list)
    for sample in samples:
        case = sample['case']
        if case.control == 'style':
            groups[(case.content, case.source, sample['checkpoint'], sample['seed'])].append(sample)
    pairs = [compare_outputs(first, second, distance) for group in groups.values()
             for first, second in combinations(group, 2) if first['case'].reference != second['case'].reference]
    between = [entry['output_style_distance_proxy'] for entry in pairs if entry['kind'] == 'different_style']
    within = [entry['output_style_distance_proxy'] for entry in pairs if entry['kind'] == 'same_style_different_reference']
    return {'pairs': pairs, 'between_style_response_mean': mean(between) if between else None,
            'within_style_reference_sensitivity_mean': mean(within) if within else None,
            'between_over_within_ratio': mean(between) / mean(within) if between and within and mean(within) > 0 else None,
            'automatic_disentanglement_pass': None, 'requires_visual_review': True,
            'interpretation': 'Large between-style response alone can be wrong stylization; zero within-style response alone can mean ignored A.'}


def aggregate_sample_metrics(samples: Sequence[dict]) -> dict[str, float | None]:
    """Summarize proxy measurements while keeping missing targets genuinely absent."""
    ordinary = [sample for sample in samples if sample['case'].control == 'style']
    mapping = {'mean_content_layout_ssim_proxy': 'content_layout_ssim_proxy',
               'mean_content_over_reference_layout_advantage': 'content_over_reference_layout_advantage',
               'mean_matched_style_group_advantage_proxy': 'matched_style_group_advantage_proxy',
               'mean_matched_target_advantage_proxy': 'matched_target_advantage_proxy'}
    result = {}
    for output_key, metric_key in mapping.items():
        values = [sample['metrics'][metric_key] for sample in ordinary
                  if sample['metrics'].get(metric_key) is not None]
        result[output_key] = mean(values) if values else None
    return result
