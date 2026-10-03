import re
from typing import Dict
from tasks.base_task import BaseTask
from models.token import Token
from models.token_location import TokenLocation


def _blank(text: str) -> str:
    return re.sub(r'[^\r\n]', ' ', text)


def mask_rst(content: str) -> str:
    """Mask recognized non-prose spans while retaining every source offset."""
    # ponytail: custom RST/Sphinx syntax may need more rules or a parser when real failures justify it.
    masked = []
    block_indent = None
    options_indent = None
    option_indent = None
    directive_re = re.compile(r'\.\.\s+(?:\|[^|]+\|\s+)?([\w-]+)::')
    for line in content.splitlines(keepends=True):
        stripped = line.lstrip(' \t')
        prefix_length = len(line) - len(stripped)
        indent = len(line[:prefix_length].expandtabs())
        if block_indent is not None:
            if not stripped.strip() or indent > block_indent:
                masked.append(_blank(line))
                continue
            block_indent = None
        if options_indent is not None:
            if not stripped.strip():
                masked.append(line)
                continue
            if indent > options_indent and (
                re.match(r':[\w-]+:', stripped)
                or (option_indent is not None and indent > option_indent)
            ):
                if re.match(r':[\w-]+:', stripped):
                    option_indent = indent
                masked.append(_blank(line))
                continue
            options_indent = option_indent = None
        directive = directive_re.match(stripped)
        if directive:
            if directive.group(1) in {'code', 'code-block', 'sourcecode'}:
                block_indent = indent
                masked.append(_blank(line))
            else:
                options_indent = indent
                header_end = prefix_length + directive.end()
                masked.append(_blank(line[:header_end]) + line[header_end:])
            continue
        if re.match(r'\.\.\s+_[^\n]*:', stripped):
            block_indent = indent
            masked.append(_blank(line))
            continue
        # Footnotes, citations and substitution markup are explicit markup, not comments.
        if re.match(r'\.\.(?:\s|$)', stripped) and not re.match(r'\.\.\s+[\[|_]', stripped):
            block_indent = indent
            masked.append(_blank(line))
            continue
        masked.append(line)
        if stripped.rstrip().endswith('::'):
            bullet = re.match(r'(?:[-+*]|\d+[.)])\s+', stripped)
            block_indent = len(line[:prefix_length + bullet.end()].expandtabs()) if bullet else indent
    result = ''.join(masked)
    result = re.sub(r'``(?:(?!\n[ \t\r]*\n).)*?``',
                    lambda match: _blank(match.group()), result, flags=re.DOTALL)
    # Preserve visible explicit-link labels, including links to named targets.
    result = re.sub(r'(`[^`\n]*?)(<[^>\n]+>)(`_?)',
                    lambda match: match.group(1) + _blank(match.group(2)) + match.group(3), result)
    return re.sub(r'(?:https?|ftp)://[^\s<>`]+', lambda match: _blank(match.group()), result)


def mask_markdown(content: str) -> str:
    """Mask common Markdown/MDX non-prose without changing source offsets."""
    # ponytail: uncommon MDX expressions/custom syntax need a parser if real failures justify it.
    lines = content.splitlines(keepends=True)
    first = 0
    while first < len(lines) and (not lines[first].strip() or lines[first].startswith('# Source: ')):
        first += 1
    if first < len(lines) and lines[first].strip() == '---':
        end = next((i for i in range(first + 1, len(lines)) if lines[i].strip() in ('---', '...')), None)
        if end is not None:
            lines[first:end + 1] = [_blank(line) for line in lines[first:end + 1]]
    masked = []
    fence = None
    list_stack = []
    indented_code = False
    previous_blank = True
    for line in lines:
        quote = re.match(r'^(?: {0,3}> ?)+', line)
        quote_depth = quote[0].count('>') if quote else 0
        body = line[quote.end():] if quote else line
        expanded = body.expandtabs()
        stripped = expanded.lstrip(' ')
        indent = len(expanded) - len(stripped)
        if fence:
            char, length, depth, column = fence
            if quote_depth < depth or (column and stripped.strip() and indent < column):
                fence = None
            else:
                masked.append(_blank(line))
                close = re.match(r'(`{3,}|~{3,})(.*)', stripped)
                if close and quote_depth == depth and close[1][0] == char and len(close[1]) >= length and not close[2].strip():
                    fence = None
                previous_blank = not stripped.strip()
                continue
        bullet = re.match(r'(?:[-+*]|\d+[.)])\s+', stripped)
        if bullet:
            while list_stack and indent <= list_stack[-1][0]:
                list_stack.pop()
            list_stack.append((indent, indent + bullet.end()))
        elif stripped.strip():
            while list_stack and indent < list_stack[-1][1]:
                list_stack.pop()
        column = list_stack[-1][1] if list_stack else 0
        fence_text = stripped[bullet.end():] if bullet else stripped
        marker = re.match(r'(`{3,}|~{3,})(.*)', fence_text)
        valid_fence = marker and (bullet or indent < column + 4) and not (marker[1][0] == '`' and '`' in marker[2])
        if valid_fence:
            fence = (marker[1][0], len(marker[1]), quote_depth, column)
            masked.append(_blank(line))
        elif indent >= column + 4 and (previous_blank or indented_code):
            indented_code = True
            masked.append(_blank(line))
        elif indented_code and not stripped.strip():
            masked.append(_blank(line))
        else:
            indented_code = False
            masked.append(line)
        previous_blank = not stripped.strip()
    result = ''.join(masked)
    result = re.sub(r'<!--.*?-->', lambda match: _blank(match.group()), result, flags=re.DOTALL)
    result = re.sub(r'<(pre|code|codetab|script|style)\b[^>]*>.*?</\1\s*>',
                    lambda match: _blank(match.group()), result, flags=re.DOTALL | re.IGNORECASE)
    protected = []

    def mask_tag(match):
        text = match.group()
        if re.search(r'\n[ \t\r]*\n', text):
            return text
        chars = list(_blank(text))
        for attribute in re.finditer(r'\b(?:alt|title)\s*=\s*(["\'])(.*?)\1', text, re.DOTALL):
            start, end = attribute.span(2)
            chars[start:end] = text[start:end]
            protected.append((match.start() + start, text[start:end]))
        return ''.join(chars)

    result = re.sub(r'</?[A-Za-z][\w:-]*(?:"[^"]*"|\'[^\']*\'|[^<>"\'])*>', mask_tag, result)
    # An odd number of preceding backslashes escapes an opening delimiter.
    def mask_inline(match):
        start = match.start()
        prefix = result[:start]
        slashes = len(prefix) - len(prefix.rstrip('\\'))
        return match.group() if slashes % 2 else _blank(match.group())

    result = re.sub(r'(?<!`)(`+)(?!`)(?:(?!\n[ \t\r]*\n).)*?(?<!`)\1(?!`)',
                    mask_inline, result, flags=re.DOTALL)
    result = re.sub(r'(?<=\])\((?:[^()\n]|\([^()\n]*\))*\)',
                    lambda match: _blank(match.group()), result)
    result = re.sub(r'(\[[^\]\n]+\])(\[[^\]\n]*\])',
                    lambda match: match[1] + _blank(match[2]), result)
    result = re.sub(r'^ {0,3}\[[^\]\n]+\]:[^\n]*',
                    lambda match: _blank(match.group()), result, flags=re.MULTILINE)
    result = re.sub(r'(?:https?|ftp)://[^\s<>`]+', lambda match: _blank(match.group()), result)
    chars = list(result)
    for start, text in protected:
        chars[start:start + len(text)] = text
    return ''.join(chars)


class TokenizerTask(BaseTask):
    def __init__(self, settings_config, repo_config):
        super().__init__(settings_config, repo_config)
        self.word_regex = re.compile(r"\b(?![_\-0-9])[A-Za-z0-9']+(?:-[A-Za-z0-9']+)*\b")
        if 'text_format' in repo_config and repo_config['text_format'] not in ('rst', 'plain', 'markdown'):
            raise ValueError('text_format must be rst, markdown or plain')

    def run(self, content_dict: Dict[str, str]) -> Dict[str, Token]:
        validated_content = self.validate_input(content_dict)
        token_dict = {}

        for key, content in validated_content.items():
            default_format = 'rst' if key.endswith('.rst') else 'markdown' if key.endswith('.md') else 'plain'
            text_format = self.repo_config.get('text_format', default_format)
            if text_format == 'rst':
                content = mask_rst(content)
            elif text_format == 'markdown':
                content = mask_markdown(content)
            lines = content.split('\n')
            for i, line in enumerate(lines, start=1):
                for surface in re.findall(self.word_regex, line):
                    if surface.lower().endswith("'s") and len(surface) > 3:
                        surface = surface[:-2]  # possessive: score the base word
                    word = surface.lower()
                    if word in token_dict:
                        token = token_dict[word]
                        token.locations.append(TokenLocation(key, i))
                    else:
                        token_dict[word] = Token(word, self.repo_config['name'], [TokenLocation(key, i)])
                    if surface.isupper() and len(surface) > 1:
                        token_dict[word].uppercase_occurrences += 1

        # Frequent components stay trusted (maxOccurrences) once compounds are kept whole.
        counts = {}
        for word, token in token_dict.items():
            for part in word.split('-'):
                counts[part] = counts.get(part, 0) + len(token.locations)
        for word, token in token_dict.items():
            token.part_occurrences = {part: counts[part] for part in word.split('-')}

        return self.validate_output(token_dict)

    def validate_input(self, input_data: Dict[str, str]) -> Dict[str, str]:
        if not isinstance(input_data, dict):
            raise ValueError("Input must be a dictionary of file paths and contents")
        return input_data

    def validate_output(self, output_data: Dict[str, Token]) -> Dict[str, Token]:
        if not isinstance(output_data, dict):
            raise TypeError("Output must be a dictionary of tokens")
        return output_data
