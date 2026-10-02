"""Concrete MongoDB operations shared by scanning and reviewed-output saving."""
import os
from pymongo import MongoClient, UpdateOne


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
    uri, database, collection = resolve_ignore_list_settings(settings)
    client = MongoClient(uri)
    try:
        result = client[database][collection].find_one({'repo_name': repo})
        return set(result.get('words', [])) if result else set()
    finally:
        client.close()


def apply_decisions(repo: str, decisions: dict[str, bool], settings: dict) -> None:
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
