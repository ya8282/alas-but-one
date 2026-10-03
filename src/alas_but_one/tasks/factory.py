from typing import Dict, Any
import importlib
from alas_but_one.registry import merged_stages
from alas_but_one.tasks.base_task import BaseTask

PACKAGE = 'alas_but_one'


# Top-level packages of the old flat layout; legacy configs name modules relative to it.
LEGACY = {'ai', 'collectors', 'formatters', 'matchers', 'models', 'tasks', 'tokenizer', 'training'}


def _import(path: str):
    if path.split('.')[0] in LEGACY:
        path = f'{PACKAGE}.{path}'
    return importlib.import_module(path)


class TaskFactory:

    @staticmethod
    def create_task(name: str, settings_config: Dict[str, Any], repo_config: Dict[str, Any], modules_config: Dict[str, Any] = None) -> BaseTask:
        task_config = merged_stages(modules_config)[name]
        task_class = getattr(_import(task_config['path']), task_config['className'])

        return task_class(settings_config, repo_config)
