from typing import Dict
from tasks.base_task import BaseTask
from models.token import Token
from spellchecker import SpellChecker
from matchers.confidence_scorer import compute_confidence


# Components shorter than this (pre, non, co, ons) are not judged inside a compound.
MIN_COMPOUND_PART = 4


class SpellCheckerTask(BaseTask):
    def __init__(self, settings_config, repo_config):
        super().__init__(settings_config, repo_config)
        self.checker = SpellChecker()

    def run(self, token_dict: Dict[str, Token]) -> Dict[str, Token]:
        validated_tokens = self.validate_input(token_dict)
        # A compound is misspelled only through a long enough unknown component,
        # and not when its hyphenless spelling is a word (pre-emptively).
        parts = {w: w.split('-') for w in validated_tokens if '-' in w}
        unknown = self.checker.unknown(
            [w for w in validated_tokens if w not in parts]
            + [p for ps in parts.values() for p in ps if len(p) >= MIN_COMPOUND_PART]
            + [''.join(ps) for ps in parts.values()])

        limit = self.settings_config.get('maxOccurrences', float('inf'))

        for word, token in validated_tokens.items():
            counts = token.part_occurrences or {}
            flawed = word if word in unknown and counts.get(word, 0) <= limit else None
            if word in parts:
                flawed = None
                if ''.join(parts[word]) in unknown:
                    flawed = next((p for p in parts[word] if len(p) >= MIN_COMPOUND_PART and p in unknown
                                   and counts.get(p, 0) <= limit), None)
            token.misspelled = flawed is not None
            token.confidence, suggestion = compute_confidence(
                flawed or word, token.misspelled, self.checker, token.uppercase_ratio)
            token.suggestion = ('-'.join(suggestion if p == flawed else p for p in parts[word])
                                if suggestion and word in parts else suggestion)

        return self.validate_output(validated_tokens)

    def validate_input(self, input_data: Dict[str, Token]) -> Dict[str, Token]:
        if not isinstance(input_data, dict):
            raise ValueError("Input must be a dictionary of tokens")
        return input_data

    def validate_output(self, output_data: Dict[str, Token]) -> Dict[str, Token]:
        if not isinstance(output_data, dict):
            raise TypeError("Output must be a dictionary of tokens")
        return output_data
