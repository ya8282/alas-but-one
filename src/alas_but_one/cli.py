"""
alas-but-one — atomic typo candidate finder for documentation repositories.

Usage:
  alas                               # run all repos, JSONL output
  alas --format csv                  # CSV output
  alas --repo "Golang Driver Docs"   # single repo by display name
  alas --ai                          # AI review of borderline tokens
  alas --parallel                    # process repos concurrently
  alas --fail-above 0.8              # CI gate: exit 1 if any non-ignored candidate has confidence >= 0.8
  alas --fail-above 0.8 --quiet      # ...and print only those: repo<TAB>word<TAB>confidence<TAB>file:line
  alas --verbose                     # per-stage token counts
  alas --log run.log                 # debug log: stage timings, raw AI responses
  alas --train labels.jsonl          # train classifier from labeled JSONL
  alas --config ~/abo.json           # config elsewhere (or set ABO_CONFIG)
  alas --init                        # write an example config.json here and exit
"""
import argparse
import contextlib
import io
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict, Optional

from alas_but_one.config import DEFAULT_CONFIG, ConfigError, check_repo_dir, init_config, load_config, resolve_config_path, resolve_repo_dir
from alas_but_one.ignore_list_store import IgnoreListError
from alas_but_one.pipeline import Pipeline
from alas_but_one.ai.hooks import HookRegistry
from alas_but_one.ai.reviewer import AIReviewer
from alas_but_one.training.predictor import MLPredictor
from alas_but_one.tasks.factory import TaskFactory


def setup_logging(path: str) -> logging.Handler:
    """Sends DEBUG logs to `path`; caller removes the returned handler."""
    handler = logging.FileHandler(path, encoding='utf-8')
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s'))
    root = logging.getLogger()
    root.addHandler(handler)
    handler._prev_level = root.level
    root.setLevel(logging.DEBUG)
    for noisy in ('anthropic', 'httpx', 'httpcore'):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return handler


def teardown_logging(handler: logging.Handler) -> None:
    root = logging.getLogger()
    root.removeHandler(handler)
    root.setLevel(handler._prev_level)
    handler.close()


def run_repo(
    repo_name: str,
    repo_config: Dict,
    settings_config: Dict,
    modules_config: Dict,
    output_format: str,
    ai_enabled: bool,
    predictor: MLPredictor,
    verbose: bool,
    include_ignored: bool = False,
    config_dir: str = '',
    config_path: str = '',
    hits: Optional[list] = None,
    fail_above: Optional[float] = None,
) -> str:
    hooks = HookRegistry()

    if verbose:
        @hooks.post_stage('tokenizer')
        def _log_tokenizer(stage, data):
            print(f"    tokenizer: {len(data)} unique tokens")
            return data

        @hooks.post_stage('max_occurrence_matcher')
        def _log_matcher(stage, data):
            print(f"    max_occurrence_matcher: {len(data)} candidates")
            return data

        @hooks.post_stage('spell_checker')
        def _log_spell(stage, data):
            misspelled = sum(1 for t in data.values() if t.misspelled)
            print(f"    spell_checker: {misspelled}/{len(data)} flagged misspelled")
            return data

    # Merge --ai flag into settings without mutating the original
    effective_settings = dict(settings_config)
    if ai_enabled:
        ai_cfg = dict(effective_settings.get('ai', {}))
        ai_cfg['enabled'] = True
        effective_settings = {**settings_config, 'ai': ai_cfg}

    directory = resolve_repo_dir(repo_config, effective_settings, config_dir)
    check_repo_dir(repo_config, directory, config_path)

    # Main pipeline (everything except formatting)
    pipeline = Pipeline(effective_settings, repo_config, modules_config, hooks=hooks)
    pipeline.add_task('collector')
    pipeline.add_task('reader')
    pipeline.add_task('tokenizer')
    pipeline.add_task('max_occurrence_matcher')
    pipeline.add_task('spell_checker')
    pipeline.add_task('ignore_list_matcher')

    token_dict = pipeline.run(directory)
    content_map = pipeline.stage_results.get('reader', {})
    if verbose:
        for stage, seconds in pipeline.stage_times.items():
            print(f"    {stage}: {seconds:.2f}s")

    # ML confidence override (if a trained model exists)
    if predictor.available:
        token_dict = predictor.apply(token_dict)
        if verbose:
            print(f"    ml_predictor: confidence scores updated from {predictor.model_path}")

    # AI review of borderline tokens (confidence in review zone)
    reviewer = AIReviewer(effective_settings)
    token_dict = reviewer.run(token_dict, content_map)

    if hits is not None and fail_above is not None:
        hits.extend(
            (repo_config['name'], token)
            for token in token_dict.values()
            if token.ignore != 'Y' and token.confidence >= fail_above
        )

    # Format and write output
    if not include_ignored:
        token_dict = {word: token for word, token in token_dict.items() if token.ignore != 'Y'}
    formatter_key = 'jsonl_formatter' if output_format == 'jsonl' else 'formatter'
    formatter = TaskFactory.create_task(
        formatter_key, effective_settings, repo_config, modules_config
    )
    return formatter.run(token_dict, content_map)


def _hit_line(repo: str, token) -> str:
    loc = token.locations[0] if token.locations else None
    where = f"{loc.filename}:{loc.line}" if loc else '-'
    return f"{repo}\t{token.text}\t{token.confidence:.2f}\t{where}"


def cmd_run(args, config: Dict) -> int:
    """Scans repos; returns the exit code (1 when --fail-above is met or any repo failed)."""
    if not args.quiet:
        return _scan(args, config)[0]
    with contextlib.redirect_stdout(io.StringIO()):
        code, hits = _scan(args, config)
    for repo, token in sorted(hits, key=lambda h: (h[0], -h[1].confidence, h[1].text)):
        print(_hit_line(repo, token))
    return code


def _scan(args, config: Dict):
    settings = config['settings']
    modules = config.get('modules', {})

    model_path = settings.get('training', {}).get('model_path', 'models/classifier.pkl')
    predictor = MLPredictor(model_path)
    if predictor.available:
        print(f"ML model loaded from {model_path}")

    repos = {
        k: v for k, v in config['repositories'].items()
        if args.repo is None or v['name'] == args.repo
    }

    if not repos:
        sys.exit(f"No repository found matching '{args.repo}'")

    hits: list = []
    failed: list = []
    fail_above = args.fail_above

    def process(repo_name, repo_config):
        name = repo_config['name']
        logging.getLogger(__name__).debug("%s : start", name)
        print(f"Processing: {name}")
        try:
            output = run_repo(
                repo_name, repo_config, settings, modules,
                output_format=args.format,
                ai_enabled=args.ai,
                predictor=predictor,
                verbose=args.verbose,
                include_ignored=args.include_ignored,
                config_dir=config.get('config_dir', ''),
                config_path=config.get('config_path', ''),
                **({'hits': hits, 'fail_above': fail_above} if fail_above is not None else {}),
            )
        except IgnoreListError:
            raise  # the shared ignore list is unusable, so no repo can be scanned
        except Exception as e:
            logging.getLogger(__name__).debug("Repo %s failed", name, exc_info=True)
            failed.append(name)
            # ConfigError text already names the repo, directory and fix.
            print(e if isinstance(e, ConfigError) else f"{name}: FAILED ({e})", file=sys.stderr)
            return
        print(f"  {name} -> {output}")

    if len(repos) == 1 or not args.parallel:
        for repo_name, repo_config in repos.items():
            process(repo_name, repo_config)
    else:
        with ThreadPoolExecutor(thread_name_prefix='repo') as executor:
            for future in [executor.submit(process, k, v) for k, v in repos.items()]:
                future.result()

    # A repo that failed to scan cannot be certified clean.
    return (1 if failed or hits else 0), hits


def cmd_train(args, config: Dict) -> None:
    from alas_but_one.training.trainer import train
    settings = config['settings']
    model_path = settings.get('training', {}).get('model_path', 'models/classifier.pkl')
    min_samples = settings.get('training', {}).get('min_training_samples', 20)
    train(args.train, model_path, min_samples)


def _unit_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number")
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"{text} is outside 0 to 1")
    return value


def _version() -> str:
    try:
        return version('alas-but-one')
    except PackageNotFoundError:
        return 'unknown'


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Find atomic typo candidates in documentation repositories.'
    )
    parser.add_argument(
        '--version', action='version', version=f'%(prog)s {_version()}'
    )
    parser.add_argument(
        '--config', metavar='PATH',
        help='Config file (default: $ABO_CONFIG, ./config.json, then $XDG_CONFIG_HOME/alas-but-one/config.json).'
    )
    parser.add_argument(
        '--init', action='store_true',
        help='Write an example config to --config PATH (default ./config.json) and exit; never overwrites.'
    )
    parser.add_argument(
        '--train', metavar='JSONL',
        help='Train ML classifier from a labeled JSONL file and exit.'
    )
    parser.add_argument(
        '--repo', metavar='NAME',
        help='Run only on the repository with this display name.'
    )
    parser.add_argument(
        '--format', choices=['jsonl', 'csv'], default='jsonl',
        help='Output format (default: jsonl).'
    )
    parser.add_argument(
        '--ai', action='store_true',
        help='Enable AI reviewer for borderline-confidence tokens.'
    )
    parser.add_argument(
        '--parallel', action='store_true',
        help='Process multiple repos concurrently.'
    )
    parser.add_argument(
        '--include-ignored', action='store_true',
        help='Include approved words for auditing or reversing ignore decisions.'
    )
    parser.add_argument(
        '--fail-above', metavar='X', type=_unit_float,
        help='Exit 1 if any non-ignored candidate has confidence >= X (0 to 1). Output files are still written.'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='With --fail-above: print only the qualifying candidates '
             '(repo, word, confidence, file:line; tab-separated), one per line.'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Print per-stage token counts.'
    )
    parser.add_argument(
        '--log', metavar='FILE',
        help='Write a DEBUG log (stage timings, raw AI responses) to FILE; overrides settings.log_file.'
    )

    args = parser.parse_args()
    if args.quiet and args.fail_above is None:
        parser.error('--quiet requires --fail-above')
    if args.init:
        try:
            print(f'Wrote example config to {init_config(args.config or DEFAULT_CONFIG)}')
        except ConfigError as error:
            sys.exit(str(error))
        return
    try:
        config = load_config(resolve_config_path(args.config))
    except ConfigError as error:
        sys.exit(str(error))

    log_file = args.log or config['settings'].get('log_file')
    handler = setup_logging(log_file) if log_file else None
    exit_code = 0
    try:
        if args.train:
            cmd_train(args, config)
        else:
            exit_code = cmd_run(args, config)
    except (ConfigError, IgnoreListError) as error:
        sys.exit(str(error))
    finally:
        if handler:
            teardown_logging(handler)
    if exit_code:
        sys.exit(exit_code)


if __name__ == '__main__':
    main()
