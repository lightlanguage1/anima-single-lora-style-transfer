"""Factor-specific behavioral disentanglement probes; no architecture changes.

Pair selection costs O(samples²). Image/latent distances are cached per pair.
These diagnostics describe interventions on images, not purity of hidden features.
"""
from __future__ import annotations

from dataclasses import replace
from html import escape
from itertools import combinations
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Sequence

from anima_disentanglement import TestCase, layout_similarity, style_distance

MIN_INPUT_DISTANCE = 1e-6
PAIR_TYPES = ('A_style_response', 'A_content_leakage', 'B_content_response', 'B_style_leakage')


def add_source_style_probes(cases: Sequence[TestCase], pool: Sequence[TestCase]) -> list[TestCase]:
    """Reuse existing paired C as same-content source variants; never fabricate C."""
    variants: dict[str, set[tuple[str, str]]] = {}
    for candidate in pool:
        if candidate.target and candidate.control == 'style':
            variants.setdefault(candidate.content, set()).add((candidate.style, candidate.target))
    result = list(cases)
    for case in cases:
        if case.control != 'style' or case.source_variant != 'original':
            continue
        for style, target in sorted(variants.get(case.content, set())):
            if target not in (case.source, case.reference):
                result.append(replace(case, source=target, source_variant='paired_C:' + style,
                                      variant_evidence='paired_target_may_have_geometry_drift', source_style=style))
    return list(dict.fromkeys(result))


def classify_pair(first: TestCase, second: TestCase) -> tuple[str, str] | None:
    """Select one-factor interventions and state reference-content confounding."""
    if first.control != 'style' or second.control != 'style':
        return None
    same_source = (first.content, first.source) == (second.content, second.source)
    same_reference = (first.style, first.reference) == (second.style, second.reference)
    if same_source and first.reference != second.reference:
        known = first.reference_content is not None and second.reference_content is not None
        same_content = known and first.reference_content == second.reference_content
        if first.style != second.style:
            return ('A_style_response', 'content_matched' if same_content else 'content_confounded')
        if not same_content:
            return ('A_content_leakage', 'content_labelled' if known else 'content_unlabelled')
    if same_reference and first.source != second.source:
        if first.content == second.content and first.source_variant != second.source_variant:
            evidence = first.variant_evidence or second.variant_evidence or 'declared_same_content'
            return ('B_style_leakage', evidence)
        if first.content != second.content and first.source_variant == second.source_variant == 'original':
            matched = first.source_style is not None and first.source_style == second.source_style
            return ('B_content_response', 'source_style_matched' if matched else 'source_style_may_differ')
    return None


class PairDistances:
    """Cache official layout SSIM and the existing latent style statistic."""

    def __init__(self, assets: Any, distance: Callable) -> None:
        self.assets = assets
        self.distance = distance
        self.cached: dict[tuple[str, str, str], float] = {}

    def input_distance(self, first: str, second: str, channel: str) -> float:
        """Measure an input change in the same units used for output changes."""
        key = (channel, *sorted((first, second)))
        if key not in self.cached:
            if channel == 'layout':
                value = layout_distance(self.assets.image(first), self.assets.image(second))
            else:
                value = style_distance(self.assets.latent(first), self.assets.latent(second), self.distance)
            self.cached[key] = value
        return self.cached[key]


def layout_distance(first: Any, second: Any) -> float:
    """Map grayscale SSIM [-1,1] to a layout dissimilarity proxy [0,1]."""
    return max(0.0, min(1.0, (1.0 - layout_similarity(first, second)) / 2.0))


def normalized_gain(output_distance: float, input_distance: float) -> float | None:
    """Return response per input change, excluding numerically degenerate inputs."""
    return output_distance / input_distance if input_distance > MIN_INPUT_DISTANCE else None


def directional_layout_gain(first: dict, second: dict, paths: tuple[str, str], assets: Any) -> float | None:
    """Detect output layout movement toward the corresponding changed input."""
    first_input, second_input = (assets.image(path) for path in paths)
    denominator = 2 * (1 - layout_similarity(first_input, second_input))
    if denominator <= MIN_INPUT_DISTANCE:
        return None
    first_margin = layout_similarity(first['image'], first_input) - layout_similarity(first['image'], second_input)
    second_margin = layout_similarity(second['image'], second_input) - layout_similarity(second['image'], first_input)
    return (first_margin + second_margin) / denominator


def measure_pair(first: dict, second: dict, classification: tuple[str, str], backend: PairDistances) -> dict:
    """Measure wanted/unwanted responses with matched checkpoint and sampling noise."""
    kind, evidence = classification
    first_case, second_case = first['case'], second['case']
    paths = ((first_case.reference, second_case.reference) if kind.startswith('A_')
             else (first_case.source, second_case.source))
    output_layout = layout_distance(first['image'], second['image'])
    output_style = style_distance(first['latent'], second['latent'], backend.distance)
    input_layout = backend.input_distance(*paths, 'layout')
    input_style = backend.input_distance(*paths, 'style')
    return {'kind': kind, 'evidence': evidence, 'checkpoint': first['checkpoint'], 'seed': first['seed'],
            'output_a': first['output'], 'output_b': second['output'],
            'content_a': first_case.content, 'content_b': second_case.content,
            'style_a': first_case.style, 'style_b': second_case.style,
            'input_a': paths[0], 'input_b': paths[1],
            'input_layout_distance_proxy': input_layout, 'input_style_distance_proxy': input_style,
            'output_layout_distance_proxy': output_layout, 'output_style_distance_proxy': output_style,
            'layout_transmission_proxy': normalized_gain(output_layout, input_layout),
            'style_transmission_proxy': normalized_gain(output_style, input_style),
            'directional_layout_transmission_proxy': directional_layout_gain(first, second, paths, backend.assets)}


def collect_pairs(samples: Sequence[dict], backend: PairDistances) -> list[dict]:
    """Exclude all cross-seed and cross-checkpoint comparisons."""
    records = []
    for first, second in combinations(samples, 2):
        if (first['checkpoint'], first['seed']) != (second['checkpoint'], second['seed']):
            continue
        classification = classify_pair(first['case'], second['case'])
        if classification:
            records.append(measure_pair(first, second, classification, backend))
    return records


def average_metric(records: Sequence[dict], metric: str) -> float | None:
    """Average defined measurements; missing observations remain unavailable."""
    values = [record[metric] for record in records if record.get(metric) is not None]
    return mean(values) if values else None


def summarize_pair_groups(records: Sequence[dict]) -> dict:
    """Keep intervention types and confounding evidence separately visible."""
    result = {}
    for kind in PAIR_TYPES:
        selected = [record for record in records if record['kind'] == kind]
        evidence = sorted({record['evidence'] for record in selected})
        result[kind] = {**summarize_measurements(selected), 'evidence': evidence,
                       'by_evidence': {label: summarize_measurements(
                           [record for record in selected if record['evidence'] == label]) for label in evidence}}
    return result


def summarize_measurements(records: Sequence[dict]) -> dict:
    """Summarize one evidence stratum without mixing it with weaker controls."""
    return {'pair_count': len(records), **{metric: average_metric(records, metric) for metric in (
        'layout_transmission_proxy', 'style_transmission_proxy',
        'directional_layout_transmission_proxy', 'output_style_distance_proxy')}}


def channel_selectivity(wanted: float | None, nuisance: float | None) -> float | None:
    """Report relative selectivity only when the wanted response is measurable."""
    if wanted is None or nuisance is None or wanted <= MIN_INPUT_DISTANCE:
        return None
    return wanted / (wanted + nuisance)


def summarize_separation(samples: Sequence[dict], assets: Any, distance: Callable) -> dict:
    """Quantify both channels while refusing a fabricated universal purity score."""
    records = collect_pairs(samples, PairDistances(assets, distance))
    groups = summarize_pair_groups(records)
    matched = [record for record in records if record['kind'] == 'A_style_response'
               and record['evidence'] == 'content_matched']
    labelled = [record for record in records if record['kind'] == 'A_content_leakage'
                and record['evidence'] == 'content_labelled']
    wanted_style = average_metric(matched, 'style_transmission_proxy')
    content_controlled = [record for record in records if record['kind'] == 'B_content_response'
                          and record['evidence'] == 'source_style_matched']
    wanted_content = average_metric(content_controlled, 'layout_transmission_proxy')
    return {'groups': groups, 'pairs': records,
            'style_selectivity_proxy': channel_selectivity(
                wanted_style, groups['B_style_leakage']['style_transmission_proxy']),
            'content_selectivity_proxy': channel_selectivity(
                wanted_content, average_metric(labelled, 'layout_transmission_proxy')),
            'style_response_with_content_control': wanted_style,
            'universal_disentanglement_purity_percent': None,
            'interpretation': 'Selectivity is wanted/(wanted+nuisance) using input-normalized distances; '
                'not a purity percentage or correctness score. Missing controls and zero wanted response return null. '
                'SSIM/latent statistics mix style and content; paired C variants can contain geometry drift.'}


def separation_coverage(cases: Sequence[TestCase]) -> dict:
    """Count possible intervention pairs before running a GPU model."""
    groups: dict[str, dict[str, int]] = {kind: {} for kind in PAIR_TYPES}
    for first, second in combinations(cases, 2):
        classification = classify_pair(first, second)
        if classification:
            kind, evidence = classification
            groups[kind][evidence] = groups[kind].get(evidence, 0) + 1
    return groups


def render_pair_overviews(directory: Path, records: Sequence[dict], samples: Sequence[dict], assets: Any) -> None:
    """Save paginated changed-input/output pairs, grouped by intervention and seed."""
    from PIL import Image, ImageDraw
    lookup = {sample['output']: sample for sample in samples}
    for kind in PAIR_TYPES:
        selected = [record for record in records if record['kind'] == kind]
        for start in range(0, len(selected), 12):
            page = selected[start:start + 12]
            canvas = Image.new('RGB', (768, len(page) * 232), 'white')
            drawing = ImageDraw.Draw(canvas)
            for position, record in enumerate(page):
                label = f'{kind} | seed {record["seed"]} | {record["evidence"]}'
                drawing.text((4, position * 232), label, fill='black')
                pictures = [assets.image(record['input_a']), assets.image(record['input_b']),
                            lookup[record['output_a']]['image'], lookup[record['output_b']]['image']]
                for column, picture in enumerate(pictures):
                    drawing.text((column * 192 + 4, position * 232 + 18),
                                 ['changed input 1', 'changed input 2', 'output 1', 'output 2'][column], fill='black')
                    canvas.paste(picture.resize((192, 192)), (column * 192, position * 232 + 40))
            canvas.save(directory / f'separation_{kind}_{start // 12:03d}.png')


def save_separation_report(directory: Path, summary: dict, samples: Sequence[dict], assets: Any) -> None:
    """Write a separate report with all raw pairs and intervention image sheets."""
    (directory / 'separation_depth.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    save_pair_table(directory / 'separation_pairs.csv', summary['pairs'])
    render_pair_overviews(directory, summary['pairs'], samples, assets)
    labels = dict(zip(PAIR_TYPES, ['A 侧：换画风的有效响应', 'A 侧：同画风换内容的泄漏',
                                  'B 侧：固定参考换内容的有效响应', 'B 侧：同内容换原画风的残留']))
    sections = []
    for kind, label in labels.items():
        pictures = ''.join(f'<a href="{path.name}"><img src="{path.name}"></a>'
                           for path in sorted(directory.glob(f'separation_{kind}_*.png')))
        sections.append(f'<h2>{label}</h2><pre>{escape(json.dumps(summary["groups"][kind], ensure_ascii=False, indent=2))}</pre>{pictures}')
    description = ('每组固定 checkpoint、噪声和提示词，只交换指定输入。输入图1/2与输出图1/2一一对应。'
                   '传递率＝输出变化÷输入变化：泄漏项越低越好，有效响应项还需检查是否朝正确目标变化。'
                   '方向性布局传递率接近1表示布局跟随被交换的输入，接近0表示未跟随，负值表示反向。'
                   '参考内容没有标签、不同内容图原画风不同、配对C存在几何变化时都会标记混杂。'
                   'by_evidence 单独列出各类对照，内容匹配的换画风结果不会与内容也改变的结果混为一组。'
                   '这测的是当前整体路径的行为解耦，不能证明内部特征正交，也不能给出通用解耦百分比。')
    scores = {key: summary[key] for key in ('style_selectivity_proxy', 'content_selectivity_proxy')}
    document = '<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;margin:30px}img{max-width:100%}pre{white-space:pre-wrap}</style>'
    document += '<h1>画风 / 内容解耦深度测试</h1><p>' + description + '</p>'
    document += '<p>相对选择性＝有效响应/(有效响应＋干扰响应)，无有效响应或缺少对照时留空；不是纯度，也不代表迁移正确。</p>'
    document += '<pre>' + escape(json.dumps(scores, ensure_ascii=False, indent=2)) + '</pre>'
    (directory / 'separation_depth.html').write_text(document + ''.join(sections), encoding='utf-8')


def save_pair_table(path: Path, records: Sequence[dict]) -> None:
    """Export every paired intervention for independent numerical analysis."""
    columns = list(records[0]) if records else ['kind', 'evidence']
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
