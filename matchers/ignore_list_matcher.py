from typing import Dict
from tasks.base_task import BaseTask
from models.token import Token
from ignore_list_store import load_words

class IgnoreListTask(BaseTask):
    def run(self, token_dict: Dict[str, Token]) -> Dict[str, Token]:
        validated_tokens = self.validate_input(token_dict)
        
        for word in load_words(self.repo_config['name'], self.settings_config):
            if word in validated_tokens:
                validated_tokens[word].ignore = 'Y'

        return self.validate_output(validated_tokens)

    def validate_input(self, input_data: Dict[str, Token]) -> Dict[str, Token]:
        if not isinstance(input_data, dict):
            raise ValueError("Input must be a dictionary of tokens")
        return input_data

    def validate_output(self, output_data: Dict[str, Token]) -> Dict[str, Token]:
        if not isinstance(output_data, dict):
            raise TypeError("Output must be a dictionary of tokens")
        return output_data
