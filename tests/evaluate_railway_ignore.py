"""Compare Railway review queues before/after the possessive fix and the committed ignore list (c8z)."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from evaluate_candidates import read_jsonl, summarize

TESTS = Path(__file__).resolve().parent
DATA = TESTS / 'data' / 'railway'


def verify_corpus(corpus, manifest):
    for entry in manifest['files']:
        if hashlib.sha256((corpus / entry['file']).read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError(f'Corpus content changed: {entry["file"]}')
    if len(list(corpus.rglob('*.md'))) != len(manifest['files']):
        raise ValueError('Corpus file set differs from the captured snapshot')


def scan(corpus, output, ignore_file=None):
    command = [sys.executable, str(TESTS / 'scan_real_corpus.py'), '--corpus', str(corpus), '--output', str(output)]
    if ignore_file:
        command += ['--ignore-file', str(ignore_file)]
    subprocess.run(command, check=True, capture_output=True)
    return read_jsonl(output)


def stage_metrics(records, labels):
    ranked = sorted(records, key=lambda r: (-r['confidence'], r['word']))
    sample = summarize([r for r in ranked if r['word'] in labels], labels)
    full_top = summarize(ranked[:10], labels)['top_10']
    full_top['words'] = [r['word'] for r in ranked[:10]]
    emitted = {r['word'] for r in records}
    typos = {w for w, label in labels.items() if label['label'] == 'true_positive'}
    return {'candidates': len(records),
            'sample_precision': sample['precision'], 'sample_candidates': sample['actionable_candidates'],
            'full_top_10': full_top,
            'known_prose_typos': {'retained': len(typos & emitted), 'total': len(typos)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', help='Corpus root; defaults to the manifest path.')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    manifest = json.loads((DATA / 'manifest.json').read_text())
    corpus = Path(args.corpus or manifest['root']).resolve()
    verify_corpus(corpus, manifest)
    labels = {r['word']: r for r in read_jsonl(DATA / 'labels.jsonl') + read_jsonl(DATA / 'labels_c8z.jsonl')}
    ignore = DATA / 'ignore.json'
    ignored = set(json.loads(ignore.read_text())['Railway'])
    typos = {t['word'] for t in json.loads((DATA / 'typos.json').read_text())}
    if ignored & (typos | {w for w, l in labels.items() if l['label'] == 'true_positive'}):
        raise ValueError('Ignore list contains a known prose typo')
    with tempfile.TemporaryDirectory() as directory:
        fixed = scan(corpus, Path(directory) / 'fixed.jsonl')
        final = scan(corpus, Path(directory) / 'final.jsonl', ignore)
        final_path = DATA / 'after_c8z.jsonl'
        # suggestion is dropped: spellchecker tie-breaks make it nondeterministic between runs.
        final_path.write_text(''.join(json.dumps({**r, 'suggestion': None}) + '\n' for r in sorted(final, key=lambda r: r['word'])))
    stages = {
        'baseline': stage_metrics(read_jsonl(DATA / 'baseline.jsonl'), labels),
        'after_previous': stage_metrics(read_jsonl(DATA / 'after.jsonl'), labels),
        'possessive_fix': stage_metrics(fixed, labels),
        'possessive_fix_plus_ignore_list': stage_metrics(final, labels),
    }
    result = {
        'stages': stages,
        'ignored_words': len(ignored),
        'limitation': ('Labeled sample is purposive and the ignore list was built from it, so sample precision gain from '
                       'ignoring is by construction. Full top-10 words newly surfaced were labeled in this pass '
                       '(labels_c8z.jsonl). Candidates tie at confidence 0.85 and rank alphabetically, so full-top-10 '
                       'precision reflects ranking ties; full-corpus precision is not measured.'),
        'acceptance_passed': stages['possessive_fix_plus_ignore_list']['known_prose_typos']['retained']
                             == stages['possessive_fix_plus_ignore_list']['known_prose_typos']['total'],
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
