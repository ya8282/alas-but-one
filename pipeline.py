import time
from typing import Any, Dict, List, Optional

from tasks.factory import TaskFactory
from ai.hooks import HookRegistry


class Pipeline:
    """
    Linear task pipeline with optional pre/post hooks per stage.

    Stage outputs are stored in self.stage_results so downstream code
    (e.g. the AI reviewer) can access intermediate data like the content_map
    produced by the reader stage.
    """

    def __init__(
        self,
        settings_config: Dict[str, Any],
        repo_config: Dict[str, Any],
        modules_config: Dict[str, Any],
        hooks: Optional[HookRegistry] = None,
    ):
        self.settings_config = settings_config
        self.repo_config = repo_config
        self.modules_config = modules_config
        self.hooks = hooks
        self._task_names: List[str] = []
        self._tasks = []
        self.stage_results: Dict[str, Any] = {}
        self.stage_times: Dict[str, float] = {}

    def add_task(self, task_name: str) -> None:
        task = TaskFactory.create_task(
            task_name, self.settings_config, self.repo_config, self.modules_config
        )
        self._task_names.append(task_name)
        self._tasks.append(task)

    def run(self, initial_input: Any) -> Any:
        result = initial_input

        for name, task in zip(self._task_names, self._tasks):
            if self.hooks:
                result = self.hooks.fire_pre_stage(name, result)

            start = time.perf_counter()
            result = task.run(result)
            self.stage_times[name] = time.perf_counter() - start
            self.stage_results[name] = result

            if self.hooks:
                result = self.hooks.fire_post_stage(name, result)

        return result
