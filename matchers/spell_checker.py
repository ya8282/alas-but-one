from typing import Dict
from tasks.base_task import BaseTask
from models.token import Token
from spellchecker import SpellChecker
from matchers.confidence_scorer import compute_confidence


class SpellCheckerTask(BaseTask):
    def __init__(self, settings_config, repo_config):
        super().__init__(settings_config, repo_config)
        self.checker = SpellChecker()

    def run(self, token_dict: Dict[str, Token]) -> Dict[str, Token]:
        validated_tokens = self.validate_input(token_dict)
        misspelled_set = self.checker.unknown(validated_tokens.keys())

        for word, token in validated_tokens.items():
            token.misspelled = word in misspelled_set
            token.confidence, token.suggestion = compute_confidence(
                word, token.misspelled, self.checker, token.uppercase_ratio
            )

        return self.validate_output(validated_tokens)

    def validate_input(self, input_data: Dict[str, Token]) -> Dict[str, Token]:
        if not isinstance(input_data, dict):
            raise ValueError("Input must be a dictionary of tokens")
        return input_data

    def validate_output(self, output_data: Dict[str, Token]) -> Dict[str, Token]:
        if not isinstance(output_data, dict):
            raise TypeError("Output must be a dictionary of tokens")
        return output_data
