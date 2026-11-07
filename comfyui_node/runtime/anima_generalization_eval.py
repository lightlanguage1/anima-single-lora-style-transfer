"""Fixed preview selection and latent diagnostics; time O(n²), space O(n²).

Diagnostics reuse generated latents and do not call a sampler or a model.
Latent distances are diagnostic proxies, not certified content/style metrics.
"""
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from itertools import combinations
from typing import Any

import torch

PreviewCase = dict[str, Any]
StyleDistance = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
PREVIEW_ROW_COUNT = 6
REFERENCE_COUNT = 2


def _group_cases(cases: Sequence[Mapping[str, Any]]) -> list[list[PreviewCase]]:
    """Group existing cases by content and source without changing holdouts."""
    grouped: dict[tuple[str, str], list[PreviewCase]] = defaultdict(list)
    for case in cases:
        grouped[(case['B'], case['source'])].append(dict(case))
    return [sorted(grouped[key], key=lambda case: (case['style'], case['kind']))
            for key in sorted(grouped)]


def _find_pairs(cases: Sequence[Mapping[str, Any]], kind: str) -> list[list[PreviewCase]]:
    """Find deterministic cross-style pairs with two independent references."""
    eligible = [case for case in cases if len(case.get('refs', ())) >= REFERENCE_COUNT]
    pairs = []
    for group in _group_cases(eligible):
        for first, second in combinations(group, 2):
            if first['style'] == second['style']:
                continue
            categories = {case['kind'].split('_', 1)[0] for case in (first, second)}
            matches = categories == ({'seen', 'unseen'} if kind == 'mixed' else {'unseen'})
            if matches:
                pairs.append([first, second])
    return pairs


def _copy_cases(cases: Sequence[Mapping[str, Any]]) -> list[PreviewCase]:
    """Copy selected metadata while retaining at most two original references."""
    return [dict(case, refs=list(case['refs'][:REFERENCE_COUNT])) for case in cases]


def select_preview_cases(plan: Mapping[str, Any]) -> list[PreviewCase]:
    """Keep two anchors and add held-out mixed/unseen pairs within six rows."""
    original = list(plan.get('preview', ()))
    anchors = [case for case in original if case['kind'] == 'train_anchor'][:2]
    available = list(plan.get('seen_eval', ())) + list(plan.get('unseen_eval', ()))
    mixed_pairs = _find_pairs(available, 'mixed')
    unseen_pairs = _find_pairs(available, 'unseen')
    if len(anchors) != 2:
        return _copy_cases(original[:PREVIEW_ROW_COUNT])
    for mixed in mixed_pairs:
        for unseen in unseen_pairs:
            if (mixed[0]['B'], mixed[0]['source']) != (unseen[0]['B'], unseen[0]['source']):
                return _copy_cases(anchors + mixed + unseen)
    return _copy_cases(original[:PREVIEW_ROW_COUNT])


def describe_preview_coverage(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Report sparse coverage explicitly without fabricating evaluation pairs."""
    mixed_pairs = _find_pairs(cases, 'mixed')
    unseen_pairs = _find_pairs(cases, 'unseen')
    return {
        'rows': len(cases),
        'images_per_seed': sum(len(case['refs']) for case in cases),
        'train_anchor_rows': sum(case['kind'] == 'train_anchor' for case in cases),
        'seen_unseen_same_content_pairs': len(mixed_pairs),
        'unseen_same_content_pairs': len(unseen_pairs),
        'paired_generalization_coverage': bool(mixed_pairs and unseen_pairs),
        'interpretation': 'Latent diagnostics require image review for content leakage.',
    }


def choose_preview_seed(args: Any, previous_arguments: Mapping[str, Any] | None) -> int:
    """Prefer an explicit override, then the previous run's actual preview seed."""
    explicit_seed = getattr(args, 'preview_seed', None)
    if explicit_seed is not None:
        return int(explicit_seed)
    previous = previous_arguments or {}
    if previous.get('preview_seed') is not None:
        return int(previous['preview_seed'])
    if previous.get('eval_seeds'):
        return int(previous['eval_seeds'][0])
    current_seeds = getattr(args, 'eval_seeds', ())
    if not current_seeds:
        raise ValueError('A preview_seed or a nonempty eval_seeds list is required')
    return int(current_seeds[0])


def _measure_distance(first: torch.Tensor, second: torch.Tensor,
                      style_distance: StyleDistance) -> float:
    """Reduce the existing per-example style distance to a scalar."""
    return float(style_distance(first.float(), second.float()).float().mean().item())


@torch.no_grad()
def measure_output(output: torch.Tensor, target: torch.Tensor, source: torch.Tensor,
                   style_distance: StyleDistance) -> dict[str, float]:
    """Measure generated output using existing tensors; time O(latent size)."""
    target_distance = _measure_distance(output, target, style_distance)
    source_distance = _measure_distance(output, source, style_distance)
    return {
        'target_latent_MSE': float((output.float() - target.float()).square().mean().item()),
        'style_target_distance': target_distance,
        'style_source_distance': source_distance,
        'style_target_advantage': source_distance - target_distance,
    }


def _pair_metadata(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
    """Describe a comparison without serializing tensors."""
    return {'B': first['B'], 'source': first['source'], 'seed': int(first['seed']),
            'style_a': first['style'], 'style_b': second['style'],
            'reference_a': first['A'], 'reference_b': second['A']}


def _measure_cross_style(first: Mapping[str, Any], second: Mapping[str, Any],
                         style_distance: StyleDistance) -> dict[str, Any]:
    """Compare correct versus alternate targets for identical content/noise."""
    first_correct = _measure_distance(first['latent'], first['target'], style_distance)
    first_alternate = _measure_distance(first['latent'], second['target'], style_distance)
    second_correct = _measure_distance(second['latent'], second['target'], style_distance)
    second_alternate = _measure_distance(second['latent'], first['target'], style_distance)
    return {
        **_pair_metadata(first, second),
        'correct_distance_a': first_correct, 'alternate_distance_a': first_alternate,
        'correct_distance_b': second_correct, 'alternate_distance_b': second_alternate,
        'correct_target_advantage_a': first_alternate - first_correct,
        'correct_target_advantage_b': second_alternate - second_correct,
        'mean_correct_target_advantage':
            (first_alternate + second_alternate - first_correct - second_correct) / 2,
    }


def _measure_same_style(first: Mapping[str, Any], second: Mapping[str, Any],
                        style_distance: StyleDistance) -> dict[str, Any]:
    """Measure reference sensitivity; neither zero nor large values prove success."""
    first_output, second_output = first['latent'], second['latent']
    return {
        **_pair_metadata(first, second),
        'output_latent_MSE': float((first_output.float() - second_output.float()).square().mean().item()),
        'output_style_distance': _measure_distance(first_output, second_output, style_distance),
        'interpretation': 'Inspect reference following and content preservation together.',
    }


@torch.no_grad()
def summarize_responses(samples: Sequence[Mapping[str, Any]],
                        style_distance: StyleDistance) -> dict[str, Any]:
    """Compare B/source/seed-matched samples containing A/style/latent/target."""
    grouped: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[(sample['B'], sample['source'], int(sample['seed']))].append(sample)
    cross_style, same_style = [], []
    for group in grouped.values():
        for first, second in combinations(group, 2):
            if first['style'] != second['style']:
                cross_style.append(_measure_cross_style(first, second, style_distance))
            elif first['A'] != second['A']:
                same_style.append(_measure_same_style(first, second, style_distance))
    return {'cross_style': cross_style, 'same_style_reference': same_style,
            'sample_count': len(samples), 'cross_style_pair_count': len(cross_style),
            'same_style_reference_pair_count': len(same_style),
            'additional_model_forwards': 0,
            'note': 'Positive target advantage is a proxy, not a generalization success claim.'}
