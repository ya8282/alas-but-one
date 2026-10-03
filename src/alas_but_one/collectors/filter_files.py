import os, re
from typing import List, Dict, Any
from alas_but_one.tasks.base_task import BaseTask

class CollectorTask(BaseTask):
    def __init__(self, settings_config: Dict[str, Any], repo_config: Dict[str, Any]):
        super().__init__(settings_config, repo_config)
        self.filename_re = r'\.(txt|rst|md)$'

    def run(self, directory: str) -> List[str]:
        input_data = self.validate_input(directory)
        paths = []

        for root, dirs, files in os.walk(input_data):
            for filename in files:
                if re.search(self.filename_re, filename):
                    paths.append(os.path.join(root, filename))

        return self.validate_output(paths)

    def validate_input(self, input_data: str) -> str:
        if not os.path.isdir(input_data):
            raise ValueError(f"Invalid directory: {input_data}")
        #TODO
        return input_data

    def validate_output(self, output_data: List[str]) -> List[str]:
        if not isinstance(output_data, list):
            raise TypeError("Output must be a list of strings")
        #TODO
        return output_data
