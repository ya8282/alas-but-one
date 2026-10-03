import hashlib
import json
import unittest
from pathlib import Path

DATA = Path(__file__).parent / 'data' / 'railway'


class RailwayManifestTest(unittest.TestCase):
    def test_captured_artifacts_match_manifest(self):
        manifest = json.loads((DATA / 'manifest.json').read_text())
        for name in ['baseline.jsonl', 'after.jsonl', 'labels.jsonl']:
            digest = hashlib.sha256((DATA / name).read_bytes()).hexdigest()
            self.assertEqual(digest, manifest[name + '_sha256'], f'{name}: update manifest.json')


if __name__ == '__main__':
    unittest.main()
