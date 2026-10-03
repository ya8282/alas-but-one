from abc import ABC
from typing import Any, Dict

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
