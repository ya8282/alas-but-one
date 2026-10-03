"""
Persist ignore-list changes from a reviewed CSV or JSONL output file to MongoDB or the
settings.ignore_list.file JSON file.

Usage:
  python save_ignore_list.py output.jsonl
  python save_ignore_list.py output.csv

For JSONL: set "ignore": true/false on each record.
For CSV:   set the 'ignore' column to 'Y'/'N'.

Uses ABO_MONGO_URI when set, otherwise settings.MONGODB_URI from the config file
($ABO_CONFIG or ./config.json).
"""
import collections
import csv
import json
import re
import sys
from config import ConfigError, default_config_path, load_config
from ignore_list_store import apply_decisions


def _load_csv(path: str) -> dict:
    """Returns {repo_name: {word: ignore_bool}}"""
    update_dict = collections.defaultdict(dict)
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ignore_val = row.get('ignore', '')
            word = row.get('word', '').lower()
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
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            word = record.get('word', '').lower()
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
        sys.exit("Usage: python save_ignore_list.py <output.jsonl|output.csv>")

    input_file = sys.argv[1]

    if input_file.endswith('.jsonl'):
        update_dict = _load_jsonl(input_file)
    elif input_file.endswith('.csv'):
        update_dict = _load_csv(input_file)
    else:
        sys.exit("Input file must be .jsonl or .csv")

    try:
        settings = load_config(default_config_path())['settings']
        _apply_updates(update_dict, settings)
    except (ValueError, ConfigError) as error:
        sys.exit(str(error))


if __name__ == '__main__':
    main()
