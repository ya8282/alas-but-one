"""Ignore list operations (MongoDB or a JSON file) shared by scanning and reviewed-output saving."""
import json
import os
import re
import tempfile
from urllib.parse import unquote, urlsplit

try:  # optional: only the MongoDB backend needs pymongo
    from pymongo import MongoClient, UpdateOne
except ImportError:
    MongoClient = UpdateOne = None


class IgnoreListError(ValueError):
    """A MongoDB failure whose message has been scrubbed of the connection URI and credentials."""


def redact(text: str, uri: str) -> str:
    """Removes the URI and its user:password from `text`."""
    secrets = [uri]
    try:
        parts = urlsplit(uri)
        secrets += [parts.username, parts.password]
        secrets += [unquote(s) for s in secrets if s]
    except ValueError:
        pass
    for secret in sorted({s for s in secrets if s}, key=len, reverse=True):
        text = text.replace(secret, '***')
    return re.sub(r'(//)[^/@\s]*@', r'\1***@', text)


def _mongo(uri: str, operation):
    """Runs operation(client), closing the client; re-raises Mongo failures without credentials."""
    client = None
    try:
        try:
            client = MongoClient(uri)
        except ValueError as error:  # pymongo raises plain ValueError for malformed URIs
            raise IgnoreListError(f'MongoDB URI is invalid: {redact(str(error), uri)}') from None
        return operation(client)
    except Exception as error:
        if type(error).__module__.split('.')[0] != 'pymongo':
            raise
        raise IgnoreListError(f'MongoDB error ({type(error).__name__}): {redact(str(error), uri)}') from None
    finally:
        if client is not None:
            client.close()


def _ignore_file(settings: dict):
    config = settings.get('ignore_list')
    if isinstance(config, dict) and 'file' in config:
        path = config['file']
        if not isinstance(path, str) or not path.strip():
            raise ValueError('ignore_list.file must be a nonempty string')
        return path
    return None


def _require_pymongo() -> None:
    if MongoClient is None:
        raise IgnoreListError('pymongo is required for the MongoDB ignore list; install it or set ignore_list.file')


def _read_file(path: str) -> dict:
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    if not isinstance(data, dict):
        raise IgnoreListError(f'{path}: ignore file must be a JSON object mapping repo names to word lists, not {type(data).__name__}')
    return data


def _repo_words(data: dict, repo: str, path: str) -> set[str]:
    words = data.get(repo, [])
    if not isinstance(words, list):
        raise IgnoreListError(f'{path}: value for repo {repo!r} must be a list of words, not {type(words).__name__}')
    return set(words)


def _write_file(path: str, data: dict) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix='.ignore-', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=False)
            f.write('\n')
        os.chmod(tmp, 0o644)  # mkstemp is 0600; the list is meant to be shared
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def resolve_ignore_list_settings(settings: dict) -> tuple[str, str, str]:
    uri = os.environ.get('ABO_MONGO_URI', settings.get('MONGODB_URI'))
    config = settings.get('ignore_list', {})
    if not isinstance(config, dict):
        raise ValueError('ignore_list must contain database and collection settings')
    values = (
        ('ABO_MONGO_URI' if 'ABO_MONGO_URI' in os.environ else 'MONGODB_URI', uri),
        ('ignore_list.database', config.get('database')),
        ('ignore_list.collection', config.get('collection')),
    )
    for name, value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'{name} must be a nonempty string')
    return tuple(value for _, value in values)


def load_words(repo: str, settings: dict) -> set[str]:
    path = _ignore_file(settings)
    if path:
        return _repo_words(_read_file(path), repo, path)
    _require_pymongo()
    uri, database, collection = resolve_ignore_list_settings(settings)
    result = _mongo(uri, lambda client: client[database][collection].find_one({'repo_name': repo}))
    return set(result.get('words', [])) if result else set()


def apply_decisions(repo: str, decisions: dict[str, bool], settings: dict) -> None:
    path = _ignore_file(settings)
    if path:
        if decisions:
            data = _read_file(path)
            words = _repo_words(data, repo, path)
            words |= {w for w, ignore in decisions.items() if ignore}
            words -= {w for w, ignore in decisions.items() if not ignore}
            data[repo] = sorted(words)
            _write_file(path, data)
        return
    _require_pymongo()
    uri, database, collection = resolve_ignore_list_settings(settings)
    add_words = [word for word, ignore in decisions.items() if ignore]
    remove_words = [word for word, ignore in decisions.items() if not ignore]
    operations = []
    if add_words:
        operations.append(UpdateOne(
            {'repo_name': repo},
            {'$addToSet': {'words': {'$each': add_words}}},
            upsert=True,
        ))
    if remove_words:
        operations.append(UpdateOne(
            {'repo_name': repo}, {'$pullAll': {'words': remove_words}},
        ))
    if not operations:
        return
    _mongo(uri, lambda client: client[database][collection].bulk_write(operations))
