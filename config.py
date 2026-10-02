"""Config loading and repository path resolution shared by the CLI scripts."""
import json
import os
from typing import Any, Dict

DEFAULT_CONFIG = 'config.json'


def default_config_path() -> str:
    return os.environ.get('ABO_CONFIG', DEFAULT_CONFIG)


def load_config(path: str = DEFAULT_CONFIG) -> Dict[str, Any]:
    """Loads JSON config and records its directory so relative repo paths resolve against it."""
    with open(path) as f:
        config = json.load(f)
    config['config_dir'] = os.path.dirname(os.path.abspath(path))
    ignore_list = config.get('settings', {}).get('ignore_list')
    if isinstance(ignore_list, dict) and isinstance(ignore_list.get('file'), str):
        file = os.path.expanduser(ignore_list['file'])
        ignore_list['file'] = os.path.join(config['config_dir'], file)  # absolute file wins in join
    log_file = config.get('settings', {}).get('log_file')
    if isinstance(log_file, str):
        config['settings']['log_file'] = os.path.join(config['config_dir'], os.path.expanduser(log_file))
    return config


def resolve_repo_dir(repo_config: Dict[str, Any], settings: Dict[str, Any], config_dir: str) -> str:
    """
    Directory to scan for one repository.

    Preferred: repo_config['path'] (absolute, ~-expanded, or relative to the
    config file) joined with optional repo_config['source_dir'].
    Legacy: settings['repo_base_full_path'] + relative_path + source_dir.
    """
    source_dir = repo_config.get('source_dir', '')
    if 'path' in repo_config:
        base = os.path.expanduser(repo_config['path'])
        if not os.path.isabs(base):
            base = os.path.join(config_dir, base)
    elif 'repo_base_full_path' in settings and 'relative_path' in repo_config:
        base = os.path.join(settings['repo_base_full_path'], repo_config['relative_path'])
    else:
        raise ValueError(
            f"Repository {repo_config.get('name', '?')!r} needs a 'path' "
            "(or legacy repo_base_full_path + relative_path)"
        )
    return os.path.normpath(os.path.join(base, source_dir))
