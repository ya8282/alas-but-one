import json
import logging
from typing import Dict, List, Optional

from models.token import Token

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a technical documentation quality reviewer. "
    "Your task is to classify rare words as typos or legitimate technical terms. "
    "Respond ONLY with a JSON array — no prose, no markdown."
)

_USER_TEMPLATE = """\
These words each appear at most {max_occ} time(s) in documentation. \
Some may be typos; others may be legitimate technical terms, acronyms, or jargon.

For each word, judge whether it is a typo that should be corrected.

Words (with surrounding context if provided):
{words_json}

Respond with a JSON array, one object per word in the same order:
[{{"word": "...", "is_typo": true/false, "confidence": 0.0-1.0, \
"suggestion": "corrected word or null", "comment": "one-line reasoning"}}]"""


def get_source_line(token: Token, content_map: Optional[Dict[str, str]]) -> Optional[str]:
    """Stripped source line of the token's first location (1-based), or None."""
    if not token.locations or not content_map:
        return None
    loc = token.locations[0]
    lines = (content_map.get(loc.filename) or '').split('\n')
    if not 1 <= loc.line <= len(lines):
        return None
    return lines[loc.line - 1].strip() or None


class AIReviewer:
    """
    Sends borderline-confidence tokens to Claude for classification.

    Tokens with confidence in [review_confidence_min, review_confidence_max]
    are considered borderline and sent to the model in batches.
    """

    def __init__(self, settings_config: dict):
        ai_cfg = settings_config.get('ai', {})
        self.enabled = ai_cfg.get('enabled', False)
        self.model = ai_cfg.get('model', 'claude-haiku-4-5-20251001')
        self.min_conf = ai_cfg.get('review_confidence_min', 0.3)
        self.max_conf = ai_cfg.get('review_confidence_max', 0.7)
        self.batch_size = ai_cfg.get('batch_size', 20)
        self.send_context = ai_cfg.get('send_context', True)
        self.max_occ = settings_config.get('maxOccurrences', 1)
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic()
            except ImportError:
                raise RuntimeError(
                    "anthropic package not installed. Run: pip install anthropic"
                )
        return self._client

    def _is_reviewable(self, token: Token) -> bool:
        return (
            not token.ai_reviewed
            and token.ignore != 'Y'
            and self.min_conf <= token.confidence <= self.max_conf
        )

    def _get_context(self, token: Token, content_map: Dict[str, str]) -> Optional[str]:
        if not token.locations or not content_map:
            return None
        loc = token.locations[0]
        content = content_map.get(loc.filename)
        if not content:
            return None
        lines = content.split('\n')
        i = loc.line - 1
        snippet = lines[max(0, i - 1): min(len(lines), i + 2)]
        return " | ".join(ln.strip() for ln in snippet if ln.strip()) or None

    def _review_batch(self, tokens: List[Token], content_map: Dict[str, str]) -> None:
        client = self._get_client()

        if self.send_context:
            payload = [
                {"word": t.text, "context": self._get_context(t, content_map)}
                for t in tokens
            ]
        else:
            payload = [{"word": t.text} for t in tokens]

        user_msg = _USER_TEMPLATE.format(
            max_occ=self.max_occ,
            words_json=json.dumps(payload, indent=2),
        )

        response = client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )

        raw = response.content[0].text.strip()
        logger.debug("AI batch of %d words, raw response: %s", len(tokens), raw)
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        results = json.loads(raw.strip())
        result_map = {r["word"]: r for r in results}

        for t in tokens:
            r = result_map.get(t.text)
            if not r:
                continue
            t.ai_reviewed = True
            t.confidence = round(float(r.get("confidence", t.confidence)), 4)
            suggestion = r.get("suggestion")
            if suggestion and suggestion != t.text:
                t.suggestion = suggestion
            t.ai_comment = r.get("comment")

    def run(
        self, token_dict: Dict[str, Token], content_map: Dict[str, str]
    ) -> Dict[str, Token]:
        if not self.enabled:
            return token_dict

        candidates = [t for t in token_dict.values() if self._is_reviewable(t)]
        if not candidates:
            return token_dict

        print(f"  [AI reviewer] reviewing {len(candidates)} borderline tokens via {self.model}")
        for i in range(0, len(candidates), self.batch_size):
            batch = candidates[i: i + self.batch_size]
            try:
                self._review_batch(batch, content_map)
            except Exception as e:
                logger.debug("AI batch %d failed: %r", i // self.batch_size + 1, e)
                print(f"  [AI reviewer] batch {i // self.batch_size + 1} failed: {e}")

        reviewed = sum(1 for t in candidates if t.ai_reviewed)
        print(f"  [AI reviewer] {reviewed}/{len(candidates)} tokens reviewed")
        return token_dict
