"""
Persist ignore-list changes from a reviewed CSV or JSONL output file to MongoDB or the
settings.ignore_list.file JSON file.

Usage:
  alas-save-ignore output.jsonl
  alas-save-ignore output.csv

For JSONL: set "ignore": true/false on each record.
For CSV:   set the 'ignore' column to 'Y'/'N'.

Uses ABO_MONGO_URI when set, otherwise settings.MONGODB_URI from the config file
(found as for alas: $ABO_CONFIG, ./config.json, then $XDG_CONFIG_HOME/alas-but-one/config.json).
"""
import collections
import csv
import json
import os
import re
import sys
from alas_but_one.config import ConfigError, load_config, resolve_config_path
from alas_but_one.ignore_list_store import apply_decisions


def _clean_word(word, path: str, number: int) -> str:
    if not isinstance(word, str):
        raise ValueError(f"Input file {os.path.abspath(path)} line {number}: word must be text.")
    return word.lower()


def _load_csv(path: str) -> dict:
    """Returns {repo_name: {word: ignore_bool}}"""
    update_dict = collections.defaultdict(dict)
    with open(path, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ignore_val = row.get('ignore', '')
            word = _clean_word(row.get('word'), path, reader.line_num)
            repo = row.get('repo', '')
            if not word or not repo:
                continue
            if re.search(r'[yY]', ignore_val):
                update_dict[repo][word] = True
            elif re.search(r'[nN]', ignore_val):
                update_dict[repo][word] = False
    return update_dict


def _load_jsonl(path: str) -> dict:
    """Returns {repo_name: {word: ignore_bool}}"""
    update_dict = collections.defaultdict(dict)
    with open(path, encoding='utf-8-sig') as f:
        for number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                raise ValueError(f"Input file {os.path.abspath(path)} line {number} is not valid JSON.") from None
            if not isinstance(record, dict):
                raise ValueError(f"Input file {os.path.abspath(path)} line {number} is not a JSON object.")
            word = _clean_word(record.get('word'), path, number)
            repo = record.get('repo', '')
            ignore = record.get('ignore')
            if not word or not repo or ignore is None:
                continue
            update_dict[repo][word] = bool(ignore)
    return update_dict


def _apply_updates(update_dict: dict, settings: dict) -> None:
    if not update_dict:
        print("No updates to apply.")
        return
    for repo_name, decisions in update_dict.items():
        apply_decisions(repo_name, decisions, settings)
    print(f"Applied decisions for {len(update_dict)} repositories.")


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: alas-save-ignore <output.jsonl|output.csv>")

    input_file = sys.argv[1]

    try:
        if input_file.endswith('.jsonl'):
            update_dict = _load_jsonl(input_file)
        elif input_file.endswith('.csv'):
            update_dict = _load_csv(input_file)
        else:
            sys.exit("Input file must be .jsonl or .csv")
    except FileNotFoundError:
        sys.exit(f"Input file {os.path.abspath(input_file)} not found.")
    except OSError as error:
        sys.exit(f"Input file {os.path.abspath(input_file)} cannot be read: {error.strerror}.")
    except UnicodeDecodeError:
        sys.exit(f"Input file {os.path.abspath(input_file)} is not UTF-8 text.")
    except ValueError as error:  # bad JSONL line
        sys.exit(str(error))

    try:
        settings = load_config(resolve_config_path())['settings']
        _apply_updates(update_dict, settings)
    except (ValueError, ConfigError) as error:
        sys.exit(str(error))


if __name__ == '__main__':
    main()
