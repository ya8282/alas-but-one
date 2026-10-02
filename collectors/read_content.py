import logging
from typing import Dict, List
from tasks.base_task import BaseTask

logger = logging.getLogger(__name__)


class ReaderTask(BaseTask):
    def run(self, paths: List[str]) -> Dict[str, str]:
        validated_paths = self.validate_input(paths)
        data = {}
        for path in validated_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data[path] = f.read()
            except UnicodeDecodeError:
                logger.warning("Invalid UTF-8 in %s; replacing undecodable bytes", path)
                with open(path, 'r', encoding='utf-8', errors='replace') as f:
                    data[path] = f.read()
        return self.validate_output(data)

    def validate_input(self, input_data: List[str]) -> List[str]:
        if not isinstance(input_data, list):
            raise ValueError("Input must be a list of file paths")
        return input_data

    def validate_output(self, output_data: Dict[str, str]) -> Dict[str, str]:
        if not isinstance(output_data, dict):
            raise TypeError("Output must be a dictionary of file paths and contents")
        return output_data
