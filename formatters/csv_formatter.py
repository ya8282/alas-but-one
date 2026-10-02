import csv
from typing import Dict

from tasks.base_task import BaseTask
from models.token import Token
from ai.reviewer import get_source_line


class CsvFormatterTask(BaseTask):
    def __init__(self, settings_config, repo_config):
        super().__init__(settings_config, repo_config)
        self.field_names = [
            'word', 'repo', 'locations', 'context', 'num_occurrences', 'uppercase_occurrences',
            'misspelled', 'confidence', 'suggestion', 'ignore', 'label',
        ]

    def run(self, tokens: Dict[str, Token], content_map=None) -> str:
        validated_tokens = self.validate_input(tokens)
        output_file = f"{self.repo_config['name']}.csv"

        sorted_tokens = sorted(
            validated_tokens.values(), key=lambda t: (-t.confidence, t.text)
        )

        with open(output_file, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.field_names)
            writer.writeheader()
            for token in sorted_tokens:
                writer.writerow({
                    'word': token.text,
                    'repo': token.repo,
                    'locations': self._format_locations(token.locations),
                    'context': get_source_line(token, content_map) or '',
                    'num_occurrences': len(token.locations),
                    'uppercase_occurrences': token.uppercase_occurrences,
                    'misspelled': token.misspelled,
                    'confidence': token.confidence,
                    'suggestion': token.suggestion or '',
                    'ignore': token.ignore,
                    'label': token.label or '',
                })

        return self.validate_output(output_file)

    def _format_locations(self, token_locations):
        if len(token_locations) == 1:
            return f"{token_locations[0].filename}:{token_locations[0].line}"
        return "\n".join(
            f"{loc.filename}:{loc.line}" for loc in token_locations
        )

    def validate_input(self, input_data: Dict[str, Token]) -> Dict[str, Token]:
        if not isinstance(input_data, dict):
            raise ValueError("Input must be a dictionary of tokens")
        return input_data

    def validate_output(self, output_data: str) -> str:
        if not isinstance(output_data, str):
            raise TypeError("Output must be a string (file path)")
        return output_data
