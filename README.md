# Alas, But One

Finds atomic typo candidates in documentation repositories by surfacing words that appear at most N times across a corpus of `.rst`, `.txt` and `.md` files. Rare words are scored by likelihood of being a real typo and written to JSONL for review, AI classification, or ML training.

## Installation

Recommended, an isolated command-line install:

```bash
pipx install alas-but-one
```

Or install with the optional extras (add `--force` if `alas-but-one` is already installed):

```bash
pipx install 'alas-but-one[ai,ml]'
```

Or use a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install 'alas-but-one[ai,ml]'
```

To work on the code, install from a clone:

```bash
git clone https://github.com/ccho-mongodb/alas-but-one.git
cd alas-but-one
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[ai,ml]'
```

The install provides the `alas` and `alas-save-ignore` commands. The `ai` extra (`anthropic`) is only needed for `--ai` and the `ml` extra (`scikit-learn`, `numpy`) for `--train`. The core requirements are `pyspellchecker` and `pymongo`. Prefer pipx or a clean virtual environment: the unrelated package named `spellchecker` breaks imports of `pyspellchecker` in shared environments.

## Setup

Run `alas --init` to write an example `config.json` in the current directory (or at `--config PATH`; it never overwrites a file), then edit its settings and repositories. The example uses a JSON ignore list file, so no MongoDB is needed to start; switch `ignore_list` to `database` and `collection` to use MongoDB:

```json
{
  "settings": {
    "maxOccurrences": 1,
    "MONGODB_URI": "mongodb://localhost:27017",
    "ignore_list": {"database": "alas_data", "collection": "ignore_words"}
  },
  "repositories": {
    "my-repo": {
      "name": "My Docs",
      "path": "/absolute/path/to/my-docs-repo",
      "source_dir": "source"
    }
  }
}
```

**Credentials:** keep them out of `config.json`. Leave `MONGODB_URI` credential-free (as above) and put the real connection string, with `user:password`, in the `ABO_MONGO_URI` environment variable (for example `export ABO_MONGO_URI='mongodb+srv://user:password@cluster.example.net'`; percent-encode special characters in the user and password with `urllib.parse.quote_plus`). If you must keep a secret in a config file, keep it out of version control: add `config.json` (or your `ABO_CONFIG` path) to the `.gitignore` of the repository that holds it. The `config.json` committed here is the localhost example and holds no secrets.

Each repository names the directory to walk with `path`: absolute, `~`-prefixed, or relative to the config file's directory. `source_dir` is optional and is joined onto `path`. The older `settings.repo_base_full_path` + `relative_path` + `source_dir` form still works when `path` is absent.

`config_dir` is a reserved config key: the loader sets it to the directory of the loaded config file and uses it to resolve relative paths, so do not set it yourself.

Both `alas` and `alas-save-ignore` look for the config in this order: `--config PATH` (`alas` only), `ABO_CONFIG`, `./config.json`, then `$XDG_CONFIG_HOME/alas-but-one/config.json` (default `~/.config/alas-but-one/config.json`).

`--log FILE` (or `settings.log_file`; `--log` wins, relative `log_file` resolves against the config file's directory) writes a DEBUG log of each stage's item count and elapsed time and each AI batch's raw response or failure. Stdout is unchanged; with neither set, no log file is created.

Add as many repositories as needed. Run all of them at once or target one with `--repo`.

Without a repository `text_format` setting, `.rst` files use RST masking, `.md` files use Markdown masking, and `.txt` files remain plain text. Set `"text_format": "rst"`, `"text_format": "markdown"` or `"text_format": "plain"` to apply that choice to all supported extensions. Other values fail validation. The bundled Golang configuration explicitly selects RST.

RST masking removes code/sourcecode directive bodies and options, literal blocks after `::`, double-backtick inline literals, explicit targets, comments and URL destinations. Headings, lists, emphasized text, admonition bodies and visible link labels remain checkable. It replaces excluded characters with spaces, preserving source lines.

Markdown masking removes fenced/indented code, inline backtick code, initial YAML frontmatter (including exports prefixed with a source header), HTML comments, link destinations/reference targets and HTML/MDX tags. It retains headings, lists, blockquote/callout prose, link/image labels and quoted `alt`/`title` values. Known HTML code blocks and Railway `CodeTab` bodies are excluded. Source characters are replaced with spaces, preserving line numbers; no parser dependency is added.

## Usage

```bash
alas                               # run all repos, JSONL output (default)
alas --format csv                  # CSV output instead
alas --output-dir out              # write outputs to ./out
alas --repo "My Docs"              # single repo by display name
alas --ai                          # AI review of borderline tokens
alas --parallel                    # process repos concurrently
alas --include-ignored             # audit or reverse prior ignore decisions
alas --fail-above 0.8              # CI gate: exit 1 if any non-ignored candidate scores >= 0.8
alas --fail-above 0.8 --quiet      # ...and print only those candidates
alas --verbose                     # per-stage token counts
alas --log run.log                 # debug log file (or settings.log_file)
alas --train labels.jsonl          # train ML classifier from labeled output
alas --config ~/abo.json           # config file elsewhere (or set ABO_CONFIG)
```

### CI mode

`--fail-above X` (0 to 1) scans every repo and writes the output files as usual, then exits 1 if any candidate has `confidence >= X`, else 0. Candidates on the ignore list or approved terms never count, including with `--include-ignored`. With or without `--fail-above`, a repo that fails to scan is reported on stderr, the remaining repos are still scanned, and the run exits 1.

`--quiet` (requires `--fail-above`) suppresses progress output and prints only the qualifying candidates to stdout, one per line, tab-separated: `repo`, `word`, `confidence`, `file:line` (first location). Repo scan failures go to stderr; AI reviewer messages, including batch failures, are suppressed, so use --log to see them.

```bash
alas --fail-above 0.8 --quiet || echo "typo candidates found"
```

## Output

Each run produces `<repository key>.jsonl` (the key under `repositories`, not the display name; `.csv` with `--format csv`) in `settings.output_dir` (default: the config file's directory; relative values resolve against the config file, like `path`). `--output-dir DIR` overrides it and is relative to the current directory, one record per candidate, sorted by `confidence` descending:

```jsonc
{
  "word": "retreive",
  "repo": "My Docs",
  "locations": [{"file": "/path/to/file.rst", "line": 42}],
  "context": "retreive the value",  // stripped source line of the first location
  "num_occurrences": 1,
  "uppercase_occurrences": 0,
  "misspelled": true,
  "confidence": 0.60,       // 0.0–1.0, higher = more likely a real typo
  "suggestion": "retrieve",
  "ignore": false,
  "label": null,            // set manually for ML training
  "ai_reviewed": false,
  "ai_comment": null
}
```

Locations are **1-based source lines** in both JSONL and CSV. Regenerate older exports before using their locations for navigation; reviewed files are not silently migrated.

Words and vocabulary keys stay lowercase. `uppercase_occurrences` counts ALL-CAPS surfaces longer than one character, and `uppercase_ratio` is that count divided by the number of locations (zero for no locations). `HTTP http HTTP` is one token with three locations and two uppercase occurrences; `Http` and `I` do not count as acronyms. Both exports include the count.

Default output omits approved words only. Low-confidence candidates remain in the queue. Direct formatter calls export every supplied token.

## How Classification Works

The pipeline applies heuristic scores, an available ML model, then optional AI review:

**1. Heuristic scoring (always runs)**

The spell checker flags unknown words and computes a `confidence` score using edit distance to the nearest known word plus word-feature penalties (short words, digits, ALL CAPS, very long strings score lower). The acronym multiplier is `1 - 0.5 * uppercase_ratio`, so mixed casing interpolates between lowercase and ALL CAPS.

**AI review (optional, `--ai`)**

Tokens with confidence between 0.3 and 0.7 (the borderline cases where the heuristic is uncertain) are sent to Anthropic's API in batches. Each word is sent with its context: the source line it first appears on plus the lines directly above and below (up to three lines, joined with ` | `). Set `ai.send_context` to `false` to send bare words only, with no document text or file paths; this is recommended for confidential documentation, though suggestions may be less accurate without context. Approved terms are excluded even with `--include-ignored`. Claude updates `confidence`, `suggestion`, and `ai_comment` for each. Responses use Anthropic structured outputs (a JSON schema), so no text parsing is involved. A batch fails as a whole, leaving its tokens unchanged, if the model refuses, is truncated, or returns a word list that differs from the request (missing, extra or duplicate words). The configured `ai.model` must support structured outputs. Requires `ANTHROPIC_API_KEY` to be set.

The review thresholds and model are configurable in `config.json`:

```json
"ai": {
  "enabled": false,
  "model": "claude-haiku-4-5-20251001",
  "review_confidence_min": 0.3,
  "review_confidence_max": 0.7,
  "batch_size": 20,
  "send_context": true
}
```

**ML classifier (optional)**

After reviewing output, set `"label"` on records you want to use as training data:

- `"true_positive"`: real typo
- `"false_positive"`: legitimate term (jargon, acronym, product name, etc.)

Then train:

```bash
alas --train output.jsonl
```

This fits a logistic regression classifier on your labeled examples and saves it to `settings.training.model_path` (default `models/classifier.json`, relative to the config file). Subsequent runs automatically use it to replace heuristic scores with ML-predicted probabilities. Re-label and re-train as the model improves.

The seven-feature vector keeps the `is_all_upper` slot but now stores the same uppercase ratio used by scoring. **Retrain existing classifiers** to learn the repaired feature; models are not deleted or retrained automatically. New JSONL exports preserve the count when loaded for training. Legacy records infer casing from an uppercase `word` and its location count; already-lowercased records use zero because lost casing cannot be recovered.

## Ignore List

By default the ignore list lives in MongoDB using PyMongo and one document per repository (`repo_name`, `words`). `ignore_list_store.py` owns settings resolution, reads and add/remove updates, and closes clients on success or failure.

To avoid MongoDB entirely, set `"ignore_list": {"file": "ignore.json"}` in `settings`. The file is one JSON object `{"repo_name": ["word", ...]}`, resolved relative to the config file (absolute and `~` paths also work), created on first save, with sorted, deduplicated lists so it can be committed and reviewed in PRs. `pymongo` is not needed in this mode.

For both scanning and saving, `ABO_MONGO_URI` overrides `settings.MONGODB_URI` when explicitly set. If absent, config is used. An empty or whitespace-only override raises a configuration error; it never falls back. URI, database and collection must be nonempty strings. Credentials are not printed in configuration errors, and MongoDB connection failures are reported with the URI and `user:password` redacted.

Default JSONL/CSV output omits approved words. To audit or remove an existing approval:

```bash
alas --include-ignored             # add --format csv if needed
alas-save-ignore my-docs.jsonl  # or reviewed CSV
```

In JSONL, set `"ignore": true` to add a word and `false` to remove it. In CSV, use `Y` and `N`. Audit exports restore approved terms with their flags intact, so setting them to false/N and saving reverses the decision. Repeated additions are idempotent, and repository decisions remain isolated. MongoDB (or the file backend below) must be available for normal scans and saving; the offline checks below mock that boundary.

## Extending the Pipeline

The pipeline runs these stages in order:

```
collector → reader → tokenizer → max_occurrence_matcher
         → spell_checker → ignore_list_matcher
         → [ml_predictor] → [ai_reviewer] → formatter
```

To add a new matcher or formatter:

1. Create a file in `src/alas_but_one/matchers/` or `formatters/`, subclass `BaseTask`, implement `run()`
2. Register it in `src/alas_but_one/registry.py` (a config `"modules"` block, if present, overrides the defaults)
3. Add it to the pipeline in `src/alas_but_one/cli.py`

## Offline checks and evaluation

Run from a clone after `pip install -e .`; MongoDB, API credentials and a trained model are unnecessary:

```bash
python3 -m unittest discover -s tests -v
python3 tests/evaluate_candidates.py --output /tmp/alas-after.jsonl
alas --help
```

The evaluator runs the real pipeline against `tests/data/evaluation.rst`, replacing only the MongoDB read with an in-memory approved-term set. AI and trained-model overrides are disabled. Every emitted word needs an explicit label in `tests/data/labels.jsonl`; unlabeled candidates fail evaluation. Precision includes all emitted candidates, including correctly spelled rare words, and an empty queue reports undefined precision (`null`).

The original output, source archive and exact invocation are retained in `tests/data/baseline.jsonl`, `baseline_source.zip` and `baseline_manifest.json`. To reproduce the original baseline using your active environment:

```bash
python3 tests/capture_baseline.py --output /tmp/alas-baseline.jsonl
```

The recorded run used Python 3.12.7 and pyspellchecker 0.8.1. Its temporary dependency copy repaired conflicting installed exports without changing the library or dictionary; the manifest records that environment and artifact hashes.

| Synthetic sample metric | Before | After |
|---|---:|---:|
| Actionable candidates | 38 | 20 |
| Labeled false positives | 31 | 13 |
| Full precision | 7/38 (18.4%) | 7/20 (35.0%) |
| Top-10 precision | 6/10 (60%) | 7/10 (70%) |
| Seeded prose typo retention | 7/7 (100%) | 7/7 (100%) |

All seeded typo locations match their original source lines after masking. Approved terms and the selected code/markup noise are absent. `tests/data/comparison.json` records the metrics. These are synthetic regression results. The separate Railway validation below uses real documents; neither result establishes full-corpus accuracy.

## Railway Markdown validation

The user-supplied Railway snapshot contains 411 `.md` files (2,632,562 bytes). Both runs scan identical source files with `maxOccurrences: 1`, no approved words, no MongoDB, no AI and no trained model. The baseline uses the archived original tokenization/scoring with current collection so Markdown files can be compared at all; the original CLI did not collect `.md`.

| Real corpus result | Before | After |
|---|---:|---:|
| All emitted candidates | 2,541 | 1,819 |
| Candidates in labeled review sample | 61 | 50 |
| False positives in labeled sample | 53 | 42 |
| Precision across labeled sample | 8/61 (13.1%) | 8/50 (16.0%) |
| Full-queue top-10 precision | 0/10 | 1/10 |
| Known natural prose typos retained | 8/8 | 8/8 |

The 70-word labeled sample is the union of each run's top 40 candidates plus eight naturally occurring typos found during source review. It is a **purposive review sample**, not a random estimate of corpus-wide precision. All eight confirmed typo locations match source lines. No typos were inserted into these documents. Candidate-count reduction is not a count of eliminated false positives across the entire corpus, which was not fully labeled.

`tests/data/railway/` retains complete baseline/after exports, explicit labels with source contexts, corpus/artifact hashes, invocation details and metrics. `typos.json` lists the eight confirmed words, corrections and exact source locations. A final uncached scan through the real collector, reader, tokenizer, occurrence matcher, scorer and formatter reproduced the after export byte-for-byte. The saved-artifact audit verifies corpus identity, labels and locations:

```bash
python3 tests/evaluate_real_corpus.py --output /tmp/railway-comparison.json
```

To rescan the local corpus without live integrations (these spelling checks can take several minutes):

```bash
python3 tests/scan_real_corpus.py --corpus /path/to/railway_docs_markdown --baseline --output /tmp/railway-baseline.jsonl
python3 tests/scan_real_corpus.py --corpus /path/to/railway_docs_markdown --output /tmp/railway-after.jsonl
```

Ranking still surfaces legitimate technical terms before real typos; a top ten with only one confirmed typo makes that limitation visible. Further queue improvement and a broader labeled precision estimate belong in Beads rather than being inferred from this sample.

## Known limitations

The four original bugs (source numbering, lost casing, inconsistent MongoDB settings and raw RST tokenization) have regression checks. The RST masker is a small scanner, not a complete RST/Sphinx parser: unknown custom syntax retains prose and may need additional masking rules when real examples justify them. The Markdown/MDX masker likewise covers common constructs rather than the entire CommonMark or MDX grammar; uncommon JSX expressions and complex custom components may retain noise. Plain-text files deliberately keep their literal/code text.

Rare words are candidates, not confirmed errors. Known words and legitimate technical terms can still enter the queue; no new ranking system or confidence threshold was introduced. Optional AI calls and ML fitting were not exercised against a live service or newly trained classifier during these offline checks.
