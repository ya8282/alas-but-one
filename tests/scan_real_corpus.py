"""Offline scan of a local corpus, optionally using archived pre-fix stages."""
import argparse
from contextlib import chdir
from pathlib import Path
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--ignore-file', help='File-backed ignore list (settings.ignore_list.file).')
    args = parser.parse_args()
    corpus = Path(args.corpus).resolve()
    output = Path(args.output).resolve()
    if not corpus.is_dir():
        parser.error('corpus must be an existing directory')
    sys.path.insert(0, str(ROOT))
    with tempfile.TemporaryDirectory() as directory:
        if args.baseline:
            with zipfile.ZipFile(ROOT / 'tests/data/baseline_source.zip') as archive:
                archive.extractall(directory)
            sys.path.insert(0, directory)
        from collectors.filter_files import CollectorTask
        from collectors.read_content import ReaderTask
        from tokenizer.tokenize_rst import TokenizerTask
        from matchers.max_occurrence_matcher import MaxOccurrenceMatcherTask
        from matchers.spell_checker import SpellCheckerTask
        from formatters.jsonl_formatter import JsonlFormatterTask
        settings = {'maxOccurrences': 1}
        if args.ignore_file:
            settings['ignore_list'] = {'file': str(Path(args.ignore_file).resolve())}
        repo = {'name': 'Railway'}
        # The original collector excluded .md. Use the current collector for
        # both runs to compare identical documents, then archived original
        # tokenization/scoring for the baseline. No source files are rewritten.
        paths = sorted(CollectorTask(settings, repo).run(str(corpus)))
        content = ReaderTask(settings, repo).run(paths)
        content = {str(Path(path).relative_to(corpus)): text for path, text in content.items()}
        if not content:
            parser.error('corpus contains no supported documentation files')
        tokens = TokenizerTask(settings, repo).run(content)
        tokens = MaxOccurrenceMatcherTask(settings, repo).run(tokens)
        tokens = SpellCheckerTask(settings, repo).run(tokens)
        if args.ignore_file:
            from matchers.ignore_list_matcher import IgnoreListTask
            tokens = IgnoreListTask(settings, repo).run(tokens)
            tokens = {w: t for w, t in tokens.items() if t.ignore != 'Y'}
        with chdir(directory):
            path = JsonlFormatterTask(settings, repo).run(tokens)
            output.write_text(Path(path).read_text())
    print(f'{len(content)} documents; {len(tokens)} candidates; {output}')


if __name__ == '__main__':
    main()
