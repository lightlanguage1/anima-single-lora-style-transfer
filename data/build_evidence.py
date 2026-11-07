"""Export paper tables from existing logs without modifying source artifacts.

Time: O(training records + training rows + evaluation records + candidate trials).
Space: O(training records + evaluation records + candidate trials).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import statistics
from collections import Counter
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np

OUTPUT = Path(__file__).parent
PREFIX = 'study_20261106_'
OUTPUTS = ('training_steps.csv', 'training_rows.csv', 'evaluation_records.csv', 'evaluation_coverage.csv',
           'component_candidates.csv', 'component_responses.csv', 'summary.json', 'component_results.json',
           'internal_paths.json', 'feedback.json', 'uniform_control.json', 'external_paths.json',
           'training_full.jsonl', 'evaluation_full.jsonl', 'style_split.json', 'archived_status.json',
           'training_configuration.json', 'evaluation_protocol.json', 'evaluation_environment.json', 'style_vectors.npz')
SUPPLEMENTARY = ('tensor_pair_distances.csv', 'tensor_style_matrix.csv', 'weights_matrices.csv', 'weights_factors.csv',
                 'activation_conditions.csv', 'activation_layers.csv', 'vgg19_reference_distribution_pairs.csv',
                 'styleshot_auxiliary_pairs.csv', 'weights_spectra.npz', 'activation_signedlog_histograms.npz',
                 'activation_signedlog_summary.json', 'activation_distributions.npz', 'activation_summary.json')
BRANCH_JSON = ('analysis_000400.json', 'analysis_000600.json', 'arguments.json', 'data_summary.json',
               'web_rounds130_review.json', 'web_rounds130_outputs.json', 'web_rounds130_changes.json',
               'dynamic_cloud100_data_summary.json', 'dynamic30_parameter_changes.json')
BRANCH_LOGS = ('train_localconv.jsonl', 'dynamic30_train.jsonl', 'dynamic_cloud100_train.jsonl')
PILOT_LOGS = (('rank_control.jsonl', 'pilot_cloud_10788_15x2/train_control.jsonl'),
              ('rank_treatment.jsonl', 'pilot_cloud_10788_15x2/train_rank.jsonl'),
              ('texture_control.jsonl', 'pilot_paired_texture_10788_15x2/train_control.jsonl'),
              ('texture_treatment.jsonl', 'pilot_paired_texture_10788_15x2/train_condition.jsonl'))
LOCAL_JSON = (('author_reference.json', 'author_reference_test_20261102/results_no_stretch.json'),
              ('vae_roundtrip.json', 'author_reference_test_20261102/vae_roundtrip_metrics.json'),
              ('style_invariance.json', 'author_reference_test_20261102/style_operator_invariance.json'),
              ('heldout_2x2.json', 'reference_probe_20261102/heldout_2x2_flow_diagnostic.json'),
              ('cache_diagnostic.json', 'reference_probe_20261102/cache_isolation_diagnostics.json'),
              ('highpass_diagnostic.json', 'reference_probe_20261102/highpass_operator_diagnostics.json'),
              ('reference_diagnostic.json', 'reference_probe_20261102/full_reference_diagnostics.json'))


def read_json(path: Path) -> Any:
    """Read one UTF-8 JSON source."""
    return json.loads(path.read_text(encoding='utf-8'))


def read_json_lines(path: Path) -> list[dict[str, Any]]:
    """Read nonempty lines from one JSONL source."""
    with path.open('r', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def clean_path(value: Any) -> Any:
    """Remove private absolute directories while retaining recorded basenames."""
    if isinstance(value, list):
        return [clean_path(entry) for entry in value]
    if isinstance(value, dict):
        return {name: clean_path(entry) for name, entry in value.items()}
    if isinstance(value, str) and re.match(r'^[A-Za-z]:[\\/]', value):
        return PureWindowsPath(value).name
    if isinstance(value, str) and value.startswith('/home/waas/'):
        return PurePosixPath(value).name
    if isinstance(value, str) and '/home/waas/' in value:
        return re.sub(r'/home/waas/[^\s"\\]+', lambda match: PurePosixPath(match.group(0)).name, value)
    return value


def write_json(name: str, payload: Any) -> None:
    """Write one new portable data file exclusively."""
    with (OUTPUT / (PREFIX + name)).open('x', encoding='utf-8') as stream:
        json.dump(clean_path(payload), stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def write_json_lines(name: str, records: list[dict[str, Any]]) -> None:
    """Write one sanitized JSONL dataset without replacing an existing export."""
    with (OUTPUT / (PREFIX + name)).open('x', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(clean_path(record), ensure_ascii=False) + '\n')


def write_csv(name: str, rows: list[dict[str, Any]], fields: list[str]) -> None:
    """Write one new UTF-8 CSV with an explicit schema."""
    with (OUTPUT / (PREFIX + name)).open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def validate_destinations() -> None:
    """Refuse to replace any previously generated paper table."""
    branch_names = tuple('branch_' + name for name in BRANCH_JSON + BRANCH_LOGS)
    pilot_names = tuple('pilot_' + name for name, _ in PILOT_LOGS)
    local_names = tuple('diagnostic_' + name for name, _ in LOCAL_JSON)
    existing = [name for name in OUTPUTS + SUPPLEMENTARY + branch_names + pilot_names + local_names
                if (OUTPUT / (PREFIX + name)).exists()]
    if existing:
        raise FileExistsError('Paper data files already exist: ' + ', '.join(existing))


def extract_training_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten the recorded per-update training examples, preserving sample order."""
    flattened = []
    for record in records:
        for example in record.get('rows', []):
            flattened.append({'step': record['step'], 'local_update': record.get('local_update'), **example})
    return flattened


def summarize_training(records: list[dict[str, Any]], samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize observed training exposures without extrapolating to other runs."""
    audited = [record for record in records if record.get('style_grad_audited')]
    return {'updates': len(records), 'first_step': records[0]['step'], 'last_step': records[-1]['step'],
            'sample_presentations': len(samples), 'distinct_pair_indices': len({row['pair_index'] for row in samples}),
            'distinct_content_ids': len({row['B'] for row in samples}), 'distinct_A_paths': len({row['A'] for row in samples}),
            'distinct_style_labels': len({row['style'] for row in samples}),
            'A_equal_C_rows': sum(row['A'] == row['style_target'] for row in samples),
            'audited_gradient_steps': len(audited),
            'audited_ratio_median': statistics.median(row['style_grad_ratio'] for row in audited),
            'ratio_estimated_flags_in_audits': dict(Counter(str(row['style_grad_ratio_estimated']) for row in audited)),
            'lr_used_min': min(row['lr_used'] for row in records), 'lr_used_max': max(row['lr_used'] for row in records),
            'style_weight_min': min(row['style_weight_effective'] for row in records),
            'style_weight_max': max(row['style_weight_effective'] for row in records)}


def summarize_split(split: dict[str, list[str]]) -> dict[str, Any]:
    """Report exact label counts and intersections, not visual leakage."""
    groups = {name: set(split[name]) for name in ('train', 'val', 'test')}
    return {'label_counts': {name: len(values) for name, values in groups.items()},
            'train_val_overlap': len(groups['train'] & groups['val']),
            'train_test_overlap': len(groups['train'] & groups['test']),
            'val_test_overlap': len(groups['val'] & groups['test']),
            'scope': 'Exact labels only; near duplicates and base-model pretraining exposure were not assessed.'}


def flatten_candidates(experiment: dict[str, Any]) -> list[dict[str, Any]]:
    """Export every candidate, decision and measured proxy in the two paths."""
    rows = []
    for arm in experiment['arms']:
        for trial in arm['trials']:
            for index, candidate in enumerate(trial['candidates']):
                rows.append({'schedule': arm['schedule'], 'step_index': trial['step_index'], 't': trial['t'],
                    'active_components': '+'.join(trial['active_components']), 'candidate_index': index,
                    'chosen': index == trial['winner'], 'selected_strength': candidate['selected'],
                    'remainder_strength': candidate['remainder'], 'color': candidate['parts'][0],
                    'direction': candidate['parts'][1], 'texture': candidate['parts'][2],
                    'content_proxy': candidate['content_proxy'], 'content_limit': trial['content_limit'],
                    'relative_score': candidate['relative_score'], 'allowed': candidate['allowed']})
    return rows


def flatten_responses(experiment: dict[str, Any]) -> list[dict[str, Any]]:
    """Export three finite-difference proxy changes for each response column."""
    rows = []
    for response in experiment['local_proxy_responses']:
        for name, values in zip(response['component_order'], response['relative_proxy_changes']):
            rows.append({'schedule': response['schedule'], 'step_index': response['step_index'], 'proxy': name,
                'selected_relative_change': values[0], 'remainder_relative_change': values[1],
                'strength_increment': response['strength_increment'], 'column_cosine': response['column_cosine'],
                'singular_value_1': response['singular_values'][0], 'singular_value_2': response['singular_values'][1]})
    return rows


def flatten_evaluations(records: list[dict[str, Any]], archive: Path) -> list[dict[str, Any]]:
    """Export all evaluation records and report whether each saved PNG exists."""
    rows = []
    for record in records:
        row = {name: record.get(name) for name in ('case_id', 'group', 'content', 'style', 'mode', 'seed', 'seconds')}
        row.update({name: clean_path(record.get(name)) for name in ('A', 'B', 'C', 'output')})
        row['output_exists'] = (archive / record['output']).is_file()
        rows.append(row)
    return rows


def summarize_coverage(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Count the actually recorded group/mode combinations."""
    counts = Counter((row['group'], row['mode']) for row in records)
    return [{'group': group, 'mode': mode, 'records': count} for (group, mode), count in sorted(counts.items())]


def write_training_tables(records: list[dict[str, Any]], samples: list[dict[str, Any]]) -> None:
    """Write the complete available training step and exposure tables."""
    steps = ['step', 'local_update', 'epoch', 'cursor', 'loss', 'base_loss', 'style_response_loss',
             'weighted_style_loss', 'style_weight_effective', 'style_grad_ratio', 'style_grad_audited',
             'style_grad_ratio_estimated', 'grad_cosine', 'lr_used', 'seconds', 'logical_rows']
    rows = ['step', 'local_update', 'pair_index', 'B', 'style', 'A', 'style_target', 'ordinal', 't', 'loss']
    write_csv('training_steps.csv', records, steps)
    write_csv('training_rows.csv', samples, rows)


def write_evaluation_tables(records: list[dict[str, Any]], archive: Path) -> None:
    """Write the complete 600-record archive and its group-level counts."""
    fields = ['case_id', 'group', 'content', 'style', 'mode', 'seed', 'A', 'B', 'C', 'output', 'output_exists', 'seconds']
    write_csv('evaluation_records.csv', flatten_evaluations(records, archive), fields)
    write_csv('evaluation_coverage.csv', summarize_coverage(records), ['group', 'mode', 'records'])


def write_component_tables(experiment: dict[str, Any]) -> None:
    """Write every component candidate and finite-response measurement."""
    fields = ['schedule', 'step_index', 't', 'active_components', 'candidate_index', 'chosen', 'selected_strength',
              'remainder_strength', 'color', 'direction', 'texture', 'content_proxy', 'content_limit', 'relative_score', 'allowed']
    response_fields = ['schedule', 'step_index', 'proxy', 'selected_relative_change', 'remainder_relative_change',
                       'strength_increment', 'column_cosine', 'singular_value_1', 'singular_value_2']
    write_csv('component_candidates.csv', flatten_candidates(experiment), fields)
    write_csv('component_responses.csv', flatten_responses(experiment), response_fields)


def write_supplementary(analysis: Path) -> None:
    """Include existing full measurement tables and arrays without absolute paths."""
    for name in SUPPLEMENTARY:
        source = analysis / name
        if source.suffix == '.csv':
            with source.open('r', encoding='utf-8', newline='') as stream:
                reader = csv.DictReader(stream)
                fields = list(reader.fieldnames or [])
                rows = [clean_path(row) for row in reader]
            write_csv(name, rows, fields)
        else:
            with source.open('rb') as origin, (OUTPUT / (PREFIX + name)).open('xb') as destination:
                shutil.copyfileobj(origin, destination)


def write_branch_and_pilot_data(archive: Path, branch: Path) -> None:
    """Export the existing branch, pilot and operator records without private paths."""
    for name in BRANCH_JSON:
        write_json('branch_' + name, read_json(branch / name))
    for name in BRANCH_LOGS:
        write_json_lines('branch_' + name, read_json_lines(branch / name))
    for name, relative in PILOT_LOGS:
        write_json_lines('pilot_' + name, read_json_lines(archive / 'reference_probe_20261102' / relative))
    for name, relative in LOCAL_JSON:
        write_json('diagnostic_' + name, read_json(archive / relative))


def vector_key(index: int, value: str) -> str:
    """Replace an absolute vector source path with a unique row ID and basename."""
    name = value.replace('\\', '/').rsplit('/', 1)[-1]
    return f'{index:03d}:{value.split(":", 1)[0]}:{name}'


def write_style_vectors(archive: Path) -> None:
    """Retain all vector rows and PCA arrays with unique path-free row identifiers."""
    with np.load(archive / 'style_vectors_10788.npz', allow_pickle=False) as source:
        keys = [vector_key(index, value) for index, value in enumerate(source['keys'].tolist())]
        arrays = {name: source[name] for name in source.files if name != 'keys'}
        arrays['keys'] = np.asarray(keys)
        with (OUTPUT / (PREFIX + 'style_vectors.npz')).open('xb') as stream:
            np.savez_compressed(stream, **arrays)


def write_experiment_json(archive: Path) -> dict[str, Any]:
    """Copy the observed experiment metadata while removing absolute paths."""
    probe = archive / 'reference_probe_20261102' / 'internal_lora_path_probe_20261104'
    sources = {'component_results.json': probe / 'a_parts_results.json', 'internal_paths.json': probe / 'results.json',
               'feedback.json': probe / 'feedback_results.json', 'uniform_control.json': probe / 'uniform_control_results.json',
               'external_paths.json': archive / 'reference_probe_20261102' / 'stage_path_probe_20261104' / 'results.json'}
    result = {}
    for output_name, source in sources.items():
        payload = read_json(source)
        write_json(output_name, payload)
        result[output_name] = payload
    return result['component_results.json']


def main() -> None:
    """Build evidence tables from explicitly supplied existing experiment paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--training-log', type=Path, required=True)
    parser.add_argument('--style-split', type=Path, required=True)
    parser.add_argument('--branch-dir', type=Path, required=True)
    arguments = parser.parse_args()
    validate_destinations()
    training = read_json_lines(arguments.training_log)
    samples = extract_training_rows(training)
    evaluations = read_json_lines(arguments.archive / 'records.jsonl')
    split = read_json(arguments.style_split)
    write_json_lines('training_full.jsonl', training)
    write_json_lines('evaluation_full.jsonl', evaluations)
    write_json('style_split.json', split)
    write_json('archived_status.json', read_json(arguments.archive / 'status.json'))
    write_json('training_configuration.json', read_json(arguments.training_log.with_name('launch_parameters.json')))
    write_json('evaluation_protocol.json', read_json(arguments.archive / 'protocol.json'))
    write_json('evaluation_environment.json', read_json(arguments.archive / 'environment.json'))
    write_training_tables(training, samples)
    write_evaluation_tables(evaluations, arguments.archive)
    experiment = write_experiment_json(arguments.archive)
    write_component_tables(experiment)
    write_supplementary(arguments.archive / 'lora10788_tensor_analysis')
    write_branch_and_pilot_data(arguments.archive, arguments.branch_dir)
    write_style_vectors(arguments.archive)
    summary = {'training': summarize_training(training, samples), 'split': summarize_split(split),
               'evaluation_records': len(evaluations), 'evaluation_files_present': sum(row['output_exists'] for row in flatten_evaluations(evaluations, arguments.archive)),
               'component_trials': len(flatten_candidates(experiment)), 'component_response_rows': len(flatten_responses(experiment)),
               'limitations': ['Archived outputs are 600 of 1232 planned; historical inputs used a different preprocessing protocol.',
                               'Image-level and feature-proxy metrics are not verified style success rates.',
                               'Original weights, training dataset and image licenses are not included in this table export.']}
    write_json('summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
