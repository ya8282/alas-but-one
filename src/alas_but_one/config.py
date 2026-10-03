"""Config loading and repository path resolution shared by the CLI scripts."""
import json
import os
from importlib import resources
from typing import Any, Dict, Optional

DEFAULT_CONFIG = 'config.json'
DEFAULT_MODEL_PATH = 'models/classifier.json'


class ConfigError(Exception):
    """A user-fixable config problem; the message is shown without a traceback."""


def config_search_paths() -> list:
    """Implicit config locations in priority order (ABO_CONFIG, if set, is handled by resolve_config_path)."""
    xdg = os.environ.get('XDG_CONFIG_HOME') or os.path.join(os.path.expanduser('~'), '.config')
    return [os.path.abspath(DEFAULT_CONFIG), os.path.join(xdg, 'alas-but-one', 'config.json')]


def resolve_config_path(explicit: Optional[str] = None) -> str:
    """
    Config file to load: --config, then ABO_CONFIG, then ./config.json, then
    $XDG_CONFIG_HOME/alas-but-one/config.json. An explicit or ABO_CONFIG path is
    returned as given (a missing file is then reported by load_config).
    """
    if explicit:
        return explicit
    if os.environ.get('ABO_CONFIG'):
        return os.environ['ABO_CONFIG']
    candidates = config_search_paths()
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise ConfigError(
        'No config file found. Looked for --config (alas only), $ABO_CONFIG, '
        + ', '.join(candidates)
        + '. Run `alas --init` to create one.'
    )


def init_config(path: str) -> str:
    """Writes the packaged example config to `path`; refuses to overwrite. Returns the absolute path."""
    target = os.path.abspath(path)
    text = resources.files('alas_but_one').joinpath('config.example.json').read_text(encoding='utf-8')
    try:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, 'x', encoding='utf-8') as f:
            f.write(text)
    except FileExistsError:
        raise ConfigError(f'{target} already exists; not overwriting it. Edit it, or pass --config PATH for another location.') from None
    except OSError as error:
        raise ConfigError(f'Cannot write {target}: {error.strerror}.') from None
    return target


def config_relative(value: str, config_dir: str) -> str:
    """~-expands `value`; a relative result is joined onto the config file's directory."""
    return os.path.normpath(os.path.join(config_dir, os.path.expanduser(value)))  # absolute value wins in join


def load_config(path: str = DEFAULT_CONFIG) -> Dict[str, Any]:
    """Loads JSON config and records its directory so relative repo paths resolve against it."""
    try:
        with open(path) as f:
            config = json.load(f)
    except FileNotFoundError:
        raise ConfigError(
            f"Config file {os.path.abspath(path)} not found. Create it with `alas --init`, "
            "or point ABO_CONFIG or --config PATH at an existing one."
        ) from None
    except OSError as error:
        raise ConfigError(f"Config file {os.path.abspath(path)} cannot be read: {error.strerror}.") from None
    except json.JSONDecodeError as error:
        raise ConfigError(
            f"Config file {os.path.abspath(path)} is not valid JSON: "
            f"line {error.lineno} column {error.colno}: {error.msg}."
        ) from None
    if not isinstance(config, dict):
        raise ConfigError(f"Config file {os.path.abspath(path)} must contain a JSON object at the top level. Edit the config file.")
    config['config_path'] = os.path.abspath(path)
    config['config_dir'] = os.path.dirname(os.path.abspath(path))
    cfg_path = config['config_path']

    def require(container: Dict[str, Any], key: str, label: str, default: Any = None) -> Any:
        value = container.get(key, default)
        if value is not None and not isinstance(value, str):
            raise ConfigError(f"{label} in {cfg_path} must be a string path. Edit that key in the config file.")
        return value

    config_dir = config['config_dir']
    settings = config.setdefault('settings', {})
    if not isinstance(settings, dict):
        raise ConfigError(f"settings in {cfg_path} must be an object. Edit that key in the config file.")
    ignore_list = settings.get('ignore_list')
    if ignore_list is not None and not isinstance(ignore_list, dict):
        raise ConfigError(f"settings.ignore_list in {cfg_path} must be an object. Edit that key in the config file.")
    if ignore_list:
        file = require(ignore_list, 'file', 'settings.ignore_list.file')
        if file == '':
            raise ConfigError(f"settings.ignore_list.file in {cfg_path} is empty. Set a file path, or remove the key.")
        if file is not None:
            ignore_list['file'] = config_relative(file, config_dir)
    log_file = require(settings, 'log_file', 'settings.log_file')
    if log_file:  # '' means no log file, as in cli
        settings['log_file'] = config_relative(log_file, config_dir)
    training = settings.setdefault('training', {})
    if not isinstance(training, dict):
        raise ConfigError(f"settings.training in {cfg_path} must be an object. Edit that key in the config file.")
    training['model_path'] = config_relative(
        require(training, 'model_path', 'settings.training.model_path', DEFAULT_MODEL_PATH) or DEFAULT_MODEL_PATH, config_dir)
    settings['output_dir'] = config_relative(require(settings, 'output_dir', 'settings.output_dir', '.') or '.', config_dir)
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
        base = config_relative(repo_config['path'], config_dir)
    elif 'repo_base_full_path' in settings and 'relative_path' in repo_config:
        base = os.path.join(settings['repo_base_full_path'], repo_config['relative_path'])
    else:
        raise ValueError(
            f"Repository {repo_config.get('name', '?')!r} needs a 'path' "
            "(or legacy repo_base_full_path + relative_path)"
        )
    return os.path.normpath(os.path.join(base, source_dir))


def check_repo_dir(repo_config: Dict[str, Any], directory: str, config_path: str) -> None:
    if os.path.isdir(directory):
        return
    key = '"path"' if 'path' in repo_config else '"relative_path" and settings "repo_base_full_path"'
    raise ConfigError(
        f"Repository {repo_config.get('name', '?')}: directory {directory} does not exist. "
        f"Check {key} in {config_path or 'the config file'}."
    )
