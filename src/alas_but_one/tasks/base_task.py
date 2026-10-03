import os
from abc import ABC
from typing import Any, Dict

def output_path(settings_config: Dict[str, Any], repo_config: Dict[str, Any], extension: str) -> str:
    """<settings.output_dir>/<repository key>.<extension>; the key is filename-safe, the display name is not."""
    directory = settings_config.get('output_dir', '.')
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, f"{repo_config.get('key', repo_config['name'])}.{extension}")


# Override this abstract class to create a pipeline task
class BaseTask(ABC):
    def __init__(self, settings_config: Dict[str, Any], repo_config: Dict[str, Any]):
        self.settings_config = settings_config
        self.repo_config = repo_config 

    def run(self, input_data: Any) -> Any:
        pass

    def validate_input(self, input_data: Any) -> Any:
        return input_data

    def validate_output(self, output_data: Any) -> Any:
        return output_data
