"""Run archived pre-fix modules, with no MongoDB, AI or trained model."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile

DATA = Path(__file__).resolve().parent / 'data'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    with tempfile.TemporaryDirectory() as directory:
        with zipfile.ZipFile(DATA / 'baseline_source.zip') as archive:
            archive.extractall(directory)
        sys.path.insert(0, directory)
        from tokenizer.tokenize_rst import TokenizerTask
        from matchers.max_occurrence_matcher import MaxOccurrenceMatcherTask
        from matchers.spell_checker import SpellCheckerTask
        from formatters.jsonl_formatter import JsonlFormatterTask
        settings = {'maxOccurrences': 3}
        repo = {'name': 'offline-evaluation'}
        tokens = TokenizerTask(settings, repo).run({'evaluation.rst': (DATA / 'evaluation.rst').read_text()})
        tokens = MaxOccurrenceMatcherTask(settings, repo).run(tokens)
        tokens = SpellCheckerTask(settings, repo).run(tokens)
        tokens['mongodb'].ignore = 'Y'
        previous = os.getcwd()
        try:
            os.chdir(directory)
            path = JsonlFormatterTask(settings, repo).run(tokens)
            output.write_text(Path(path).read_text())
        finally:
            os.chdir(previous)
    print(json.dumps({'baseline_candidates': len(tokens), 'output': str(output)}))


if __name__ == '__main__':
    main()
