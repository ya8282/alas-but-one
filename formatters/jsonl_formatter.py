import json
from typing import Dict

from tasks.base_task import BaseTask
from models.token import Token


class JsonlFormatterTask(BaseTask):
    """
    Writes one JSON record per token, sorted by confidence descending.
    Output file: <repo_name>.jsonl

    Each record includes all fields needed for review and ML training:
    word, repo, locations, num_occurrences, misspelled, confidence,
    suggestion, ignore, label, ai_reviewed, ai_comment.
    """

    def run(self, tokens: Dict[str, Token]) -> str:
        validated = self.validate_input(tokens)
        repo_name = self.repo_config['name']
        output_file = f"{repo_name}.jsonl"

        sorted_tokens = sorted(
            validated.values(), key=lambda t: (-t.confidence, t.text)
        )

        with open(output_file, 'w', encoding='utf-8') as f:
            for token in sorted_tokens:
                record = {
                    'word': token.text,
                    'repo': token.repo,
                    'locations': [
                        {'file': loc.filename, 'line': loc.line}
                        for loc in token.locations
                    ],
                    'num_occurrences': len(token.locations),
                    'uppercase_occurrences': token.uppercase_occurrences,
                    'misspelled': token.misspelled,
                    'confidence': token.confidence,
                    'suggestion': token.suggestion,
                    'ignore': token.ignore == 'Y',
                    'label': token.label,
                    'ai_reviewed': token.ai_reviewed,
                    'ai_comment': token.ai_comment,
                }
                f.write(json.dumps(record) + '\n')

        return self.validate_output(output_file)

    def validate_input(self, input_data: Dict[str, Token]) -> Dict[str, Token]:
        if not isinstance(input_data, dict):
            raise ValueError("Input must be a dictionary of tokens")
        return input_data

    def validate_output(self, output_data: str) -> str:
        if not isinstance(output_data, str):
            raise TypeError("Output must be a string (file path)")
        return output_data
