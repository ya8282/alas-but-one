import hashlib
import json
import unittest
from pathlib import Path

from evaluate_real_corpus import build_comparison

DATA = Path(__file__).parent / 'data' / 'railway'


class RailwayManifestTest(unittest.TestCase):
    def test_captured_artifacts_match_manifest(self):
        manifest = json.loads((DATA / 'manifest.json').read_text())
        for name in ['baseline.jsonl', 'after.jsonl', 'labels.jsonl']:
            digest = hashlib.sha256((DATA / name).read_bytes()).hexdigest()
            self.assertEqual(digest, manifest[name + '_sha256'], f'{name}: update manifest.json')

    def test_comparison_json_is_current(self):
        manifest = json.loads((DATA / 'manifest.json').read_text())
        expected = (DATA / 'comparison.json').read_text()
        self.assertEqual(json.dumps(build_comparison(manifest), indent=2) + '\n', expected,
                         'comparison.json is stale: rerun tests/evaluate_real_corpus.py --output')


if __name__ == '__main__':
    unittest.main()
