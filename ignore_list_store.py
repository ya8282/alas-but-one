"""Ignore list operations (MongoDB or a JSON file) shared by scanning and reviewed-output saving."""
import json
import os
import tempfile

try:  # optional: only the MongoDB backend needs pymongo
    from pymongo import MongoClient, UpdateOne
except ImportError:
    MongoClient = UpdateOne = None


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
        raise RuntimeError('pymongo is required for the MongoDB ignore list; install it or set ignore_list.file')


def _read_file(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


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
        return set(_read_file(path).get(repo, []))
    _require_pymongo()
    uri, database, collection = resolve_ignore_list_settings(settings)
    client = MongoClient(uri)
    try:
        result = client[database][collection].find_one({'repo_name': repo})
        return set(result.get('words', [])) if result else set()
    finally:
        client.close()


def apply_decisions(repo: str, decisions: dict[str, bool], settings: dict) -> None:
    path = _ignore_file(settings)
    if path:
        if decisions:
            data = _read_file(path)
            words = set(data.get(repo, []))
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
    client = MongoClient(uri)
    try:
        client[database][collection].bulk_write(operations)
    finally:
        client.close()
