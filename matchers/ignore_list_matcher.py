from typing import Dict
from tasks.base_task import BaseTask
from models.token import Token
from ignore_list_store import load_words
from matchers.spell_checker import MIN_COMPOUND_PART
from spellchecker import SpellChecker

class IgnoreListTask(BaseTask):
    def run(self, token_dict: Dict[str, Token]) -> Dict[str, Token]:
        validated_tokens = self.validate_input(token_dict)
        
        words = set(load_words(self.repo_config['name'], self.settings_config))
        checker = SpellChecker()
        for word, token in validated_tokens.items():
            parts = word.split('-')
            # A compound is covered when every long component left unlisted is a dictionary word.
            if word in words or (len(parts) > 1 and words.intersection(parts) and not checker.unknown(
                    [p for p in parts if p not in words and len(p) >= MIN_COMPOUND_PART])):
                token.ignore = 'Y'

        return self.validate_output(validated_tokens)

    def validate_input(self, input_data: Dict[str, Token]) -> Dict[str, Token]:
        if not isinstance(input_data, dict):
            raise ValueError("Input must be a dictionary of tokens")
        return input_data

    def validate_output(self, output_data: Dict[str, Token]) -> Dict[str, Token]:
        if not isinstance(output_data, dict):
            raise TypeError("Output must be a dictionary of tokens")
        return output_data
