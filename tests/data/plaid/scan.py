import sys,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import chdir
import tempfile
sys.path.insert(0, '/Volumes/Travel/Projects_2/docs-alas-but-one/alas-but-one')
from collectors.filter_files import CollectorTask
from collectors.read_content import ReaderTask
from tokenizer.tokenize_rst import TokenizerTask
from matchers.max_occurrence_matcher import MaxOccurrenceMatcherTask
from matchers.spell_checker import SpellCheckerTask
from formatters.jsonl_formatter import JsonlFormatterTask
SETTINGS={'maxOccurrences':1}
REPO={'name':'Plaid'}
def score(items):
    return SpellCheckerTask(SETTINGS,REPO).run(dict(items))
def main():
    started=time.monotonic()
    root=Path('/Volumes/Travel/Projects_2/docs-research/plaid/plaid_docs_markdown/docs')
    output=Path('/Volumes/Travel/Projects_2/docs-alas-but-one/alas-but-one/tests/data/plaid/candidates.jsonl')
    paths=sorted(CollectorTask(SETTINGS,REPO).run(str(root)))
    content=ReaderTask(SETTINGS,REPO).run(paths)
    content={str(Path(p).relative_to(root)):text for p,text in content.items()}
    tokens=MaxOccurrenceMatcherTask(SETTINGS,REPO).run(TokenizerTask(SETTINGS,REPO).run(content))
    items=list(tokens.items());scored={}
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(score,items[i::8]) for i in range(8)]
        for n,future in enumerate(as_completed(futures),1):
            scored.update(future.result());print(f'{n}/8 scoring chunks finished',flush=True)
    assert set(scored)==set(tokens)
    with tempfile.TemporaryDirectory() as directory,chdir(directory):
        path=JsonlFormatterTask(SETTINGS,REPO).run(scored)
        output.write_bytes(Path(path).read_bytes())
    print(f'{len(content)} documents; {len(scored)} candidates; {time.monotonic()-started:.1f} seconds; {output}',flush=True)
if __name__=='__main__':main()
