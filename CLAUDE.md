# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Purpose

`alas-but-one` finds **atomic typo candidates** in documentation: words
that appear at most N times (default: 1) across a corpus of `.rst`, `.txt` and `.md` files.
Rare words are surface candidates; human or AI review confirms whether they are
real typos or legitimate technical terms.

## Commands

```bash
pipx install alas-but-one        # or: pip install 'alas-but-one[ai,ml]' in a venv
pip install -e '.[ai,ml]'        # from a clone, for development
```

| Command | Purpose |
|---|---|
| `alas --init` | write an example config.json (to `--config PATH`, default `./config.json`); never overwrites |
| `alas /path/to/docs` | scan one dir, no config or MongoDB needed (config settings used if found; not with `--repo`/`--train`/`--init`) |
| `alas` | run all repos, JSONL output (default) |
| `alas --format csv` | CSV output |
| `alas --output-dir out` | outputs to ./out (default: settings.output_dir, config-relative) |
| `alas --repo "Golang Driver Docs"` | single repo |
| `alas --ai` | AI review of borderline tokens |
| `alas --parallel` | parallel repo processing |
| `alas --include-ignored` | audit or reverse prior ignore decisions |
| `alas --fail-above 0.8` | CI: exit 1 if a non-ignored candidate has confidence >= 0.8 |
| `alas --fail-above 0.8 --quiet` | CI: print only those (repo, word, confidence, file:line, tab-separated) |
| `alas --verbose` | per-stage token counts |
| `alas --log run.log` | DEBUG log: stage timings, raw AI responses |
| `alas --train labels.jsonl` | train ML model from labeled data |
| `alas --config PATH` | config file elsewhere (or set `ABO_CONFIG`) |
| `alas-save-ignore FILE` | save reviewed JSONL or CSV ignore decisions |

Config search order: `--config` > `ABO_CONFIG` > `./config.json` > `$XDG_CONFIG_HOME/alas-but-one/config.json`. `config_dir` is a reserved config key set by the loader to the config file's directory; never set it in a config.

## Architecture

Code lives in `src/alas_but_one/` (dev install above); module paths below are relative to it. Tests: `python -m unittest discover -s tests`.

### Pipeline stages (in order)

```
collector → reader → tokenizer → max_occurrence_matcher
         → spell_checker → ignore_list_matcher
         → [ml_predictor] → [ai_reviewer] → formatter
```

| Stage | Module | Input → Output |
|---|---|---|
| collector | `collectors/filter_files.py` | directory → `List[path]` |
| reader | `collectors/read_content.py` | `List[path]` → `Dict[path, content]` |
| tokenizer | `tokenizer/tokenize_rst.py` | `Dict[path, content]` → `Dict[word, Token]` |
| max_occurrence_matcher | `matchers/max_occurrence_matcher.py` | filters to ≤ maxOccurrences |
| spell_checker | `matchers/spell_checker.py` | sets `token.misspelled` + `token.confidence` |
| ignore_list_matcher | `matchers/ignore_list_matcher.py` | sets `token.ignore` from ignore list store |
| ml_predictor | `training/predictor.py` | overrides `token.confidence` if trained model exists |
| ai_reviewer | `ai/reviewer.py` | sends borderline tokens to Claude; updates confidence + suggestion |
| jsonl_formatter | `formatters/jsonl_formatter.py` | writes `<key>.jsonl` (in `settings.output_dir`) sorted by confidence desc |
| csv_formatter | `formatters/csv_formatter.py` | writes `<key>.csv` (backward compat) |

### Key types

**`Token`** (`models/token.py`):
- `text`: the word
- `repo`: repo display name
- `locations`: `List[TokenLocation]` (file + line number)
- `misspelled`: bool from pyspellchecker
- `confidence`: float 0.0–1.0 (likelihood of being a real typo)
- `suggestion`: best spelling correction or None
- `ignore`: 'Y'/'N' from ignore list
- `label`: 'true_positive' | 'false_positive' | None (set by human reviewer)
- `ai_reviewed`: bool
- `ai_comment`: string from AI reviewer

### Confidence scoring (`matchers/confidence_scorer.py`)

Scores are 0.0–1.0 (higher = more likely a real typo):

| Signal | Effect |
|---|---|
| Misspelled, edit distance 1 | +0.85 base |
| Misspelled, edit distance 2 | +0.60 base |
| Misspelled, no correction found | +0.40 base |
| Correctly spelled | 0.05 base |
| Word ≤ 2 chars | × 0.4 |
| Contains digits | × 0.5 |
| ALL CAPS | × 0.5 |
| Length > 20 chars | × 0.6 |

### ML training loop

1. Run the tool to generate `<key>.jsonl`
2. Open the file and set `"label"` field: `"true_positive"` or `"false_positive"`
3. Run `alas --train <key>.jsonl` to fit a logistic regression classifier
4. Subsequent runs use `models/classifier.json` (config-relative `training.model_path`) to override heuristic confidence scores
5. Re-label and re-train as the model improves

Features used: `is_misspelled`, `spell_confidence`, `edit_distance_norm`,
`word_length_norm`, `has_digits`, `is_all_upper`, `is_short`

### AI reviewer (`ai/reviewer.py`)

Only runs when `--ai` flag is passed (or `"ai": {"enabled": true}` in config).
Targets tokens with `confidence` in `[review_confidence_min, review_confidence_max]`
(defaults: 0.3–0.7). Sends batches to Claude (`claude-haiku-4-5-20251001` by default)
and updates `token.confidence`, `token.suggestion`, `token.ai_comment`.

Requires `ANTHROPIC_API_KEY` environment variable.

## Adding a new matcher or formatter

1. Create a new file in `src/alas_but_one/matchers/` or `formatters/`
2. Subclass `BaseTask` and implement `run()`
3. Register it in `src/alas_but_one/registry.py` (config `"modules"` is an optional override)
4. Add it to the pipeline in `src/alas_but_one/cli.py`

## Environment variables

| Variable | Required for |
|---|---|
| `ABO_CONFIG` | config file location (overrides `./config.json` and `$XDG_CONFIG_HOME/alas-but-one/config.json`; `--config` overrides it; `alas --init` writes the example) |
| `ABO_MONGO_URI` | ignore list (overrides config MONGODB_URI); put credentials here, never in config.json |
| `ANTHROPIC_API_KEY` | `--ai` flag |

## JSONL output format

```jsonc
{
  "word": "retreive",
  "repo": "Golang Driver Docs",
  "locations": [{"file": "/path/to/file.rst", "line": 42}],
  "num_occurrences": 1,
  "misspelled": true,
  "confidence": 0.85,       // 0.0–1.0, higher = more likely a real typo
  "suggestion": "retrieve",
  "ignore": false,
  "label": null,            // set to "true_positive"/"false_positive" for training
  "ai_reviewed": false,
  "ai_comment": null
}
```

Records are sorted by `confidence` descending, highest-priority candidates first.
