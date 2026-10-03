"""Compare the labeled synthetic sample with the archived original output."""
import argparse
from contextlib import chdir
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from alas_but_one.cli import load_config, run_repo
from alas_but_one.training.predictor import MLPredictor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'tests' / 'data'


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def summarize(records, labels):
    unknown = {r['word'] for r in records} - labels.keys()
    if unknown:
        raise ValueError(f'Unlabeled emitted candidates: {sorted(unknown)}')
    ranked = sorted(records, key=lambda record: (-record['confidence'], record['word']))
    true_words = {word for word, label in labels.items() if label['label'] == 'true_positive'}
    retained = true_words & {r['word'] for r in records}
    positives = sum(r['word'] in true_words for r in records)
    top = ranked[:10]
    top_positives = sum(r['word'] in true_words for r in top)
    return {
        'actionable_candidates': len(records),
        'true_positives': positives,
        'false_positives': len(records) - positives,
        'precision': positives / len(records) if records else None,
        'top_10': {'candidates': len(top), 'true_positives': top_positives,
                   'precision': top_positives / len(top) if top else None},
        'seeded_typos': {'retained': len(retained), 'total': len(true_words),
                         'retention': len(retained) / len(true_words) if true_words else None},
    }


def compare(before, after, labels):
    baseline = summarize(before, labels)
    current = summarize(after, labels)
    for record in after:
        lines = [loc['line'] for loc in record['locations']]
        if lines != labels[record['word']]['source_lines']:
            raise ValueError(f'Incorrect source locations for {record["word"]}: {lines}')
    expected = {word for word, label in labels.items() if label['label'] == 'true_positive'}
    emitted = {r['word'] for r in after}
    forbidden = {word for word, label in labels.items() if label['category'] in ('approved_term', 'non_prose')}
    if expected - emitted:
        raise ValueError(f'Lost seeded typos: {sorted(expected - emitted)}')
    if forbidden & emitted:
        raise ValueError(f'Retained approved/non-prose noise: {sorted(forbidden & emitted)}')
    if current['false_positives'] >= baseline['false_positives']:
        raise ValueError('False positives did not decrease')
    return {'baseline': baseline, 'after': current, 'acceptance_passed': True,
            'scope': 'synthetic regression sample; see separate Railway real-corpus evaluation'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='Write revised candidate JSONL here.')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    manifest = json.loads((DATA / 'baseline_manifest.json').read_text())
    for filename, field in [('evaluation.rst','sample_sha256'), ('baseline_source.zip','source_archive_sha256'), ('baseline.jsonl','baseline_sha256')]:
        if hashlib.sha256((DATA / filename).read_bytes()).hexdigest() != manifest[field]:
            raise ValueError(f'Baseline artifact changed: {filename}')
    labels = {r['word']: r for r in read_jsonl(DATA / 'labels.jsonl')}
    config = load_config(str(ROOT / 'config.json'))
    with tempfile.TemporaryDirectory() as directory, chdir(directory):
        Path('evaluation.rst').write_text((DATA / 'evaluation.rst').read_text())
        settings = {**config['settings'],
                    'maxOccurrences': manifest['maxOccurrences'], 'ai': {'enabled': False}, 'output_dir': directory}
        repo = {'name': 'offline-evaluation', 'path': directory, 'source_dir': ''}
        # Replace only the external MongoDB read. Real collection, masking,
        # scoring, review filtering and formatter run through the CLI pipeline.
        with patch('alas_but_one.matchers.ignore_list_matcher.load_words', return_value=set(manifest['approved_terms'])):
            path = run_repo('evaluation', repo, settings, config.get('modules', {}), 'jsonl', False,
                            MLPredictor(str(Path(directory) / 'absent.json')), False)
        after = read_jsonl(path)
        for record in after:
            for location in record['locations']:
                location['file'] = Path(location['file']).name
    metrics = compare(read_jsonl(DATA / 'baseline.jsonl'), after, labels)
    output.write_text(''.join(json.dumps(record) + '\n' for record in after))
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
