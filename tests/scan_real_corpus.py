"""Offline scan of a local corpus, optionally using archived pre-fix stages."""
import argparse
import importlib
from contextlib import chdir
from pathlib import Path
import sys
import time
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--repo-name', default='Railway', help='Repository name recorded on each candidate.')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--ignore-file', help='File-backed ignore list (settings.ignore_list.file).')
    args = parser.parse_args()
    corpus = Path(args.corpus).resolve()
    output = Path(args.output).resolve()
    if not corpus.is_dir():
        parser.error('corpus must be an existing directory')
    with tempfile.TemporaryDirectory() as directory:
        if args.baseline:
            with zipfile.ZipFile(ROOT / 'tests/data/baseline_source.zip') as archive:
                archive.extractall(directory)
            sys.path.insert(0, directory)
            # Stages absent from the archive (collectors, ai, ignore_list_store) resolve flat to current code.
            sys.path.append(str(ROOT / 'src' / 'alas_but_one'))
        # The archive holds the old flat layout; current code is the installed package.
        prefix = '' if args.baseline else 'alas_but_one.'
        CollectorTask = importlib.import_module(prefix + 'collectors.filter_files').CollectorTask
        ReaderTask = importlib.import_module(prefix + 'collectors.read_content').ReaderTask
        TokenizerTask = importlib.import_module(prefix + 'tokenizer.tokenize_rst').TokenizerTask
        MaxOccurrenceMatcherTask = importlib.import_module(prefix + 'matchers.max_occurrence_matcher').MaxOccurrenceMatcherTask
        SpellCheckerTask = importlib.import_module(prefix + 'matchers.spell_checker').SpellCheckerTask
        JsonlFormatterTask = importlib.import_module(prefix + 'formatters.jsonl_formatter').JsonlFormatterTask
        settings = {'maxOccurrences': 1}
        if args.ignore_file:
            settings['ignore_list'] = {'file': str(Path(args.ignore_file).resolve())}
        repo = {'name': args.repo_name}
        # The original collector excluded .md. Use the current collector for
        # both runs to compare identical documents, then archived original
        # tokenization/scoring for the baseline. No source files are rewritten.
        times = {}

        def timed(name, task, data):
            start = time.perf_counter()
            result = task.run(data)
            times[name] = time.perf_counter() - start
            return result

        paths = sorted(timed('collector', CollectorTask(settings, repo), str(corpus)))
        content = timed('reader', ReaderTask(settings, repo), paths)
        content = {str(Path(path).relative_to(corpus)): text for path, text in content.items()}
        if not content:
            parser.error('corpus contains no supported documentation files')
        tokens = timed('tokenizer', TokenizerTask(settings, repo), content)
        tokens = timed('max_occurrence_matcher', MaxOccurrenceMatcherTask(settings, repo), tokens)
        tokens = timed('spell_checker', SpellCheckerTask(settings, repo), tokens)
        if args.ignore_file:
            IgnoreListTask = importlib.import_module(prefix + 'matchers.ignore_list_matcher').IgnoreListTask
            tokens = IgnoreListTask(settings, repo).run(tokens)
            tokens = {w: t for w, t in tokens.items() if t.ignore != 'Y'}
        with chdir(directory):
            path = JsonlFormatterTask(settings, repo).run(tokens)
            output.write_text(Path(path).read_text())
    for stage, seconds in times.items():
        print(f'    {stage}: {seconds:.2f}s')
    print(f'{len(content)} documents; {len(tokens)} candidates; {output}')


if __name__ == '__main__':
    main()
