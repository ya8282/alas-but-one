"""Audit captured Railway exports and the explicitly labeled review sample."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from evaluate_candidates import read_jsonl, summarize

DATA = Path(__file__).resolve().parent / 'data' / 'railway'


def evaluate_review(before, after, labels, limit=40):
    ranked_before = sorted(before, key=lambda record: (-record['confidence'], record['word']))
    ranked_after = sorted(after, key=lambda record: (-record['confidence'], record['word']))
    unknown = {r['word'] for r in ranked_before[:limit] + ranked_after[:limit]} - labels.keys()
    if unknown:
        raise ValueError(f'Unlabeled review candidates: {sorted(unknown)}')
    expected = {word for word, label in labels.items() if label['label'] == 'true_positive'}
    emitted = {r['word'] for r in after}
    if expected - emitted:
        raise ValueError(f'Lost known prose typos: {sorted(expected - emitted)}')
    for record in after:
        if record['word'] not in labels:
            continue
        contexts = labels[record['word']]['source_contexts']
        expected_locations = [{'file': c['file'], 'line': c['line']} for c in contexts if c['stage'] == 'after']
        if record['locations'] != expected_locations:
            raise ValueError(f'Incorrect source location for {record["word"]}')
    review = {}
    top_10 = {}
    for stage, records in [('baseline', ranked_before), ('after', ranked_after)]:
        subset = [r for r in records if r['word'] in labels]
        review[stage] = summarize(subset, labels)
        review[stage].pop('seeded_typos')
        top_10[stage] = summarize(records[:10], labels)['top_10']
    return {
        'corpus': {'before_candidates': len(before), 'after_candidates': len(after),
                   'candidate_reduction': len(before) - len(after)},
        'review_sample': review,
        'full_queue_top_10': top_10,
        'known_prose_typos': {'retained': len(expected & emitted), 'total': len(expected),
                              'retention': len(expected & emitted) / len(expected) if expected else None},
        'full_corpus_precision': None,
        'limitation': 'Purposive labeled sample of leading candidates plus known typos; full-corpus precision is not measured.',
    }


def build_comparison(manifest):
    """Build the comparison from the captured artifacts; needs no external corpus."""
    before, after = read_jsonl(DATA / 'baseline.jsonl'), read_jsonl(DATA / 'after.jsonl')
    labels = {r['word']: r for r in read_jsonl(DATA / 'labels.jsonl')}
    metrics = evaluate_review(before, after, labels)
    metrics['corpus']['documents'] = len(manifest['files'])
    metrics['review_sample']['labeled_words'] = len(labels)
    if metrics['review_sample']['after']['false_positives'] >= metrics['review_sample']['baseline']['false_positives']:
        raise ValueError('Labeled false positives did not decrease')
    metrics['acceptance_passed'] = True
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', help='Corpus root; defaults to the recorded local path.')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    manifest = json.loads((DATA / 'manifest.json').read_text())
    corpus = Path(args.corpus or manifest['root']).resolve()
    expected_files = {entry['file'] for entry in manifest['files']}
    if {str(path.relative_to(corpus)) for path in corpus.rglob('*.md')} != expected_files:
        raise ValueError('Corpus file set differs from the captured snapshot')
    content = {}
    for entry in manifest['files']:
        path = corpus / entry['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError(f'Corpus content changed: {entry["file"]}')
        content[entry['file']] = path.read_text().splitlines()
    for filename in ['baseline.jsonl', 'after.jsonl', 'labels.jsonl']:
        if hashlib.sha256((DATA / filename).read_bytes()).hexdigest() != manifest[filename + '_sha256']:
            raise ValueError(f'Captured artifact changed: {filename}')
    before, after = read_jsonl(DATA / 'baseline.jsonl'), read_jsonl(DATA / 'after.jsonl')
    word_regex = re.compile(r"\b(?![_\-0-9])[A-Za-z0-9']+\b")
    for records, offset in [(before, 1), (after, 0)]:
        for record in records:
            for location in record['locations']:
                line = location['line'] + offset
                source_lines = content[location['file']]
                if not 1 <= line <= len(source_lines) or record['word'] not in word_regex.findall(source_lines[line - 1].lower()):
                    raise ValueError(f'Invalid source location for {record["word"]}')
    metrics = build_comparison(manifest)
    Path(args.output).write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
