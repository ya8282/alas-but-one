"""Draw a reproducible random sample of Railway candidates and report labeled precision.

Population: records the current code flags as misspelled (maxOccurrences=1, committed ignore list).
Step 1 (no labels file yet): writes random_sample.jsonl. Step 2: label every word in
random_labels.jsonl, rerun to write random_precision.json. Slow: scans the full corpus.
"""
import hashlib
import json
import math
from pathlib import Path
import random
import tempfile

from evaluate_candidates import read_jsonl
from evaluate_railway_ignore import DATA, scan, verify_corpus

SEED = 20261002
DRAWN_AT_COMMIT = '734bae0'
N = 150
Z = 1.959964


def wilson(k, n, z=Z):
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, centre - half), min(1.0, centre + half)]


def wilson_fpc(k, n, population):
    """Wilson interval with the effective sample size inflated for sampling without replacement."""
    if n >= population:
        return [k / n, k / n]
    n_eff = n * (population - 1) / (population - n)
    return wilson(k * n_eff / n, n_eff)


def candidate_hash(words):
    return hashlib.sha256('\n'.join(words).encode()).hexdigest()


def source_line(corpus, location):
    return (corpus / location['file']).read_text().splitlines()[location['line'] - 1].strip()


def draw(candidates):
    ordered = sorted(candidates, key=lambda r: r['word'])
    return random.Random(SEED).sample(ordered, min(N, len(ordered))), ordered


def report(sample, labels, population):
    missing = sorted({r['word'] for r in sample} - labels.keys())
    if missing:
        raise ValueError(f'Sampled words lack labels: {missing}')
    counts = {'true_positive': 0, 'false_positive': 0, 'uncertain': 0}
    for r in sample:
        counts[labels[r['word']]['label']] += 1
    n, tp, fp, unc = len(sample), counts['true_positive'], counts['false_positive'], counts['uncertain']
    decided = tp + fp
    return {
        'sample_size': n, 'population': population, 'counts': counts,
        'precision_decided': {'value': tp / decided if decided else None, 'tp': tp, 'decided': decided,
                              'wilson95': wilson(tp, decided), 'wilson95_fpc': wilson_fpc(tp, decided, population)},
        'precision_bounds_over_sample': {
            'uncertain_all_false_positive': tp / n, 'uncertain_all_true_positive': (tp + unc) / n,
            'wilson95_low': wilson(tp, n)[0], 'wilson95_high': wilson(tp + unc, n)[1]},
        'uncertain_fraction': unc / n,
    }


def main():
    manifest = json.loads((DATA / 'manifest.json').read_text())
    corpus = Path(manifest['root'])
    verify_corpus(corpus, manifest)
    with tempfile.TemporaryDirectory() as directory:
        records = scan(corpus, Path(directory) / 'scan.jsonl', DATA / 'ignore.json')
    candidates = [r for r in records if r['misspelled']]
    sample, ordered = draw(candidates)
    digest = candidate_hash([r['word'] for r in ordered])
    sample_path, labels_path = DATA / 'random_sample.jsonl', DATA / 'random_labels.jsonl'
    if sample_path.exists():
        stored = json.loads(sample_path.read_text().splitlines()[0])
        if stored['candidate_sha256'] != digest:
            raise ValueError(f'The candidate list changed since the sample was drawn at commit {DRAWN_AT_COMMIT}; '
                             f'check out that commit to reproduce.')
    header = {'candidate_sha256': digest, 'candidate_count': len(ordered), 'seed': SEED, 'n': len(sample),
              'population': 'misspelled=true records, maxOccurrences=1, committed ignore.json, sorted by word'}
    rows = [{'word': r['word'], 'confidence': r['confidence'], 'file': r['locations'][0]['file'],
             'line': r['locations'][0]['line'],
             'context': r.get('context') or source_line(corpus, r['locations'][0])} for r in sorted(sample, key=lambda r: r['word'])]
    sample_path.write_text(''.join(json.dumps(x) + '\n' for x in [header] + rows))
    if not labels_path.exists():
        print(f'Wrote {len(rows)} sampled words; add {labels_path.name} and rerun.')
        return
    result = report(sample, {r['word']: r for r in read_jsonl(labels_path)}, len(ordered))
    result.update({'drawn_at_commit': DRAWN_AT_COMMIT, 'seed': SEED, 'candidate_sha256': digest, 'candidate_count': len(ordered), 'limitations': [
        'Single corpus (Railway docs) and a single labeler; no inter-rater agreement measured.',
        'Population is the misspelled-flagged candidate list of the current code at maxOccurrences=1 with the committed ignore list; the ' + str(len(records) - len(candidates)) + ' in-dictionary records (misspelled=false, confidence <= 0.05) are still exported but rank below every flagged word and are excluded.',
        'The committed ignore list was built from earlier purposive labels, so precision is for the post-ignore queue only.',
        'Word-level, not occurrence-level: each word is judged from its first location and quoted context.',
        'Uncertain labels are kept out of the decided precision; the bounds show the range if they were all FP or all TP.',
        'Sample is a large share of a small population, so a finite-population-corrected Wilson interval (effective n = n(N-1)/(N-n)) is reported alongside plain Wilson, which is conservative here.',
        'Only sampled words are labeled; the remaining candidates were not reviewed and no full-corpus labeling is claimed.']})
    (DATA / 'random_precision.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
