from typing import Dict
from alas_but_one.tasks.base_task import BaseTask
from alas_but_one.models.token import Token

class MaxOccurrenceMatcherTask(BaseTask):
    def run(self, token_dict: Dict[str, Token]) -> Dict[str, Token]:
        validated_tokens = self.validate_input(token_dict)
        filtered_token_dict = {}

        for word, token in validated_tokens.items():
            if len(token.locations) <= self.settings_config['maxOccurrences']:
                filtered_token_dict[word] = token

        return self.validate_output(filtered_token_dict)

    def validate_input(self, input_data: Dict[str, Token]) -> Dict[str, Token]:
        if not isinstance(input_data, dict):
            raise ValueError("Input must be a dictionary of tokens")
        return input_data

    def validate_output(self, output_data: Dict[str, Token]) -> Dict[str, Token]:
        if not isinstance(output_data, dict):
            raise TypeError("Output must be a dictionary of tokens")
        return output_data
