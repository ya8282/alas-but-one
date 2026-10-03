"""Default pipeline stage map: stage name -> (module path, class name)."""
from typing import Any, Dict

DEFAULT_STAGES: Dict[str, Dict[str, str]] = {
    'collector': {'path': 'alas_but_one.collectors.filter_files', 'className': 'CollectorTask'},
    'reader': {'path': 'alas_but_one.collectors.read_content', 'className': 'ReaderTask'},
    'tokenizer': {'path': 'alas_but_one.tokenizer.tokenize_rst', 'className': 'TokenizerTask'},
    'max_occurrence_matcher': {'path': 'alas_but_one.matchers.max_occurrence_matcher', 'className': 'MaxOccurrenceMatcherTask'},
    'spell_checker': {'path': 'alas_but_one.matchers.spell_checker', 'className': 'SpellCheckerTask'},
    'ignore_list_matcher': {'path': 'alas_but_one.matchers.ignore_list_matcher', 'className': 'IgnoreListTask'},
    'formatter': {'path': 'alas_but_one.formatters.csv_formatter', 'className': 'CsvFormatterTask'},
    'jsonl_formatter': {'path': 'alas_but_one.formatters.jsonl_formatter', 'className': 'JsonlFormatterTask'},
}


def merged_stages(overrides: Dict[str, Any] = None) -> Dict[str, Dict[str, str]]:
    """Defaults with the optional config `modules` block merged on top."""
    return {**DEFAULT_STAGES, **(overrides or {})}
