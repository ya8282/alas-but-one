import csv
import io
import json
import logging
import os
import sys
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import Mock, patch

# Eval scripts live beside this file; make them importable under any runner.
sys.path.append(str(Path(__file__).resolve().parent))

from alas_but_one.tokenizer.tokenize_rst import TokenizerTask
from alas_but_one.models.token import Token
from alas_but_one.models.token_location import TokenLocation
from alas_but_one.ai.reviewer import AIReviewer
from alas_but_one.formatters.jsonl_formatter import JsonlFormatterTask
from alas_but_one.formatters.csv_formatter import CsvFormatterTask
from alas_but_one.collectors.read_content import ReaderTask


class RegressionTests(unittest.TestCase):
    def test_reader_keeps_universal_newlines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'mixed.rst')
            Path(path).write_bytes(b'a\r\nb\rc\n')
            self.assertEqual(ReaderTask({}, {'name': 'test'}).run([path])[path], 'a\nb\nc\n')

    def test_reader_tolerates_non_utf8_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'latin1.rst')
            Path(path).write_bytes(b'caf\xe9 ok\n')
            with self.assertLogs('alas_but_one.collectors.read_content', 'WARNING') as logs:
                content = ReaderTask({}, {'name': 'test'}).run([path])
            self.assertIn('ok', content[path])
            self.assertIn('\ufffd', content[path])
            self.assertIn(path, logs.output[0])

    def test_source_lines_and_export_locations(self):
        content = 'retreive\n\nmispeled'
        tokens = TokenizerTask({}, {'name': 'test'}).run({'sample.rst': content})
        self.assertEqual(tokens['retreive'].locations[0].line, 1)
        self.assertEqual(tokens['mispeled'].locations[0].line, 3)
        with tempfile.TemporaryDirectory() as directory:
            previous = os.getcwd()
            try:
                os.chdir(directory)
                path = JsonlFormatterTask({}, {'name': 'test'}).run(tokens)
                rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
                self.assertEqual({r['word']:r['locations'][0]['line'] for r in rows}, {'retreive':1,'mispeled':3})
                path = CsvFormatterTask({}, {'name': 'test'}).run(tokens)
                with open(path) as file:
                    rows = list(csv.DictReader(file))
                self.assertEqual({r['word']:r['locations'] for r in rows}, {'retreive':'sample.rst:1','mispeled':'sample.rst:3'})
            finally:
                os.chdir(previous)

    def test_context_is_stripped_first_location_line(self):
        content = {'sample.rst': 'intro\n\n   retreive  here  \nmispeled\nretreive again'}
        tokens = TokenizerTask({}, {'name': 'test'}).run(content)
        expected = {'retreive': 'retreive  here', 'mispeled': 'mispeled'}
        with tempfile.TemporaryDirectory() as directory:
            previous = os.getcwd()
            try:
                os.chdir(directory)
                path = JsonlFormatterTask({}, {'name': 'test'}).run(tokens, content)
                rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
                self.assertEqual({r['word']: r['context'] for r in rows if r['word'] in expected}, expected)
                path = CsvFormatterTask({}, {'name': 'test'}).run(tokens, content)
                with open(path) as file:
                    rows = list(csv.DictReader(file))
                self.assertEqual({r['word']: r['context'] for r in rows if r['word'] in expected}, expected)
            finally:
                os.chdir(previous)

    def test_ai_context_at_first_and_last_source_lines(self):
        reviewer = AIReviewer({})
        content = {'sample.rst': 'retreive\nneighbor\nmispeled'}
        first = Token('retreive', 'test', [TokenLocation('sample.rst', 1)])
        last = Token('mispeled', 'test', [TokenLocation('sample.rst', 3)])
        self.assertEqual(reviewer._get_context(first, content), 'retreive | neighbor')
        self.assertEqual(reviewer._get_context(last, content), 'neighbor | mispeled')


    def test_casing_counts_and_scores_are_order_independent(self):
        from alas_but_one.matchers.spell_checker import SpellCheckerTask
        checker = Mock()
        checker.unknown.return_value = {'http', 'i'}
        checker.correction.side_effect = lambda word: None
        contents = {'first.txt': 'HTTP http HTTP', 'second.txt': 'Http I'}
        results = []
        for content in [contents, dict(reversed(list(contents.items())))]:
            tokens = TokenizerTask({}, {'name': 'test'}).run(content)
            self.assertEqual(tokens['http'].uppercase_occurrences, 2)
            self.assertEqual(tokens['http'].uppercase_ratio, 0.5)
            self.assertEqual(tokens['i'].uppercase_occurrences, 0)
            with patch('alas_but_one.matchers.spell_checker.SpellChecker', return_value=checker):
                SpellCheckerTask({}, {}).run(tokens)
            results.append(tokens['http'].confidence)
        self.assertEqual(results, [0.3, 0.3])
        self.assertEqual(Token('empty', 'test', []).uppercase_ratio, 0)

    def test_hyphenated_compound_is_one_token_judged_by_its_components(self):
        from alas_but_one.matchers.spell_checker import SpellCheckerTask
        content = {'sample.txt': 'Act pre-emptively.\nAdd-ons and how-tos.\n\nA well-knwon trick.'}
        tokens = TokenizerTask({}, {'name': 'test'}).run(content)
        self.assertEqual(set(tokens), {'act', 'pre-emptively', 'add-ons', 'and', 'how-tos', 'a', 'well-knwon', 'trick'})
        for fragment in ('emptively', 'ons', 'tos', 'knwon', 'well'):
            self.assertNotIn(fragment, tokens)
        tokens = SpellCheckerTask({}, {}).run(tokens)
        self.assertEqual({w for w, t in tokens.items() if t.misspelled}, {'well-knwon'})
        self.assertEqual([(loc.filename, loc.line) for loc in tokens['well-knwon'].locations], [('sample.txt', 4)])
        self.assertEqual(tokens['well-knwon'].suggestion, 'well-known')
        self.assertEqual(tokens['well-knwon'].confidence, 0.6)
        tokens = SpellCheckerTask({}, {}).run(TokenizerTask({}, {'name': 'test'}).run({'b.txt': 'Add-ons and ons.'}))
        self.assertFalse(tokens['add-ons'].misspelled)
        tokens = SpellCheckerTask({}, {}).run(TokenizerTask({}, {'name': 'test'}).run({'c.txt': 'knwon-knwonly'}))
        self.assertEqual(tokens['knwon-knwonly'].suggestion, 'known-knwonly')

    def test_ignore_list_covers_compound_of_listed_component(self):
        from alas_but_one.matchers.ignore_list_matcher import IgnoreListTask
        tokens = TokenizerTask({}, {'name': 'repo'}).run({'a.txt': 'graphile-worker graphile-wroker graphile'})
        with patch('alas_but_one.matchers.ignore_list_matcher.load_words', return_value={'graphile'}):
            result = IgnoreListTask({}, {'name': 'repo'}).run(tokens)
        self.assertEqual({w: t.ignore for w, t in result.items()},
                         {'graphile-worker': 'Y', 'graphile-wroker': 'N', 'graphile': 'Y'})

    def test_compound_with_frequent_component_stays_trusted(self):
        from alas_but_one.matchers.spell_checker import SpellCheckerTask
        settings = {'maxOccurrences': 1}
        content = {'a.txt': 'zzyx-alpha\nzzyx-beta\nzzyx\nqqwv-gamma'}
        tokens = TokenizerTask(settings, {'name': 'test'}).run(content)
        tokens = SpellCheckerTask(settings, {}).run(tokens)
        self.assertEqual({w for w, t in tokens.items() if t.misspelled}, {'qqwv-gamma'})

    def test_acronym_score_interpolates_and_standalone_casing_survives(self):
        from alas_but_one.matchers.confidence_scorer import compute_confidence
        checker = Mock()
        checker.correction.return_value = None
        for ratio, expected in [(0, 0.4), (0.5, 0.3), (1, 0.2)]:
            self.assertEqual(compute_confidence('http', True, checker, uppercase_ratio=ratio)[0], expected)
        self.assertEqual(compute_confidence('HTTP', True, checker)[0], 0.2)
        self.assertEqual(compute_confidence('Http', True, checker)[0], 0.4)
        self.assertEqual(compute_confidence('I', True, checker)[0], 0.16)

    def test_long_identifiers_skip_edit_distance_two_but_prose_typos_keep_suggestions(self):
        from spellchecker import SpellChecker
        from alas_but_one.matchers.confidence_scorer import compute_confidence, MAX_EDIT2_LENGTH
        checker = SpellChecker()
        with patch.object(SpellChecker, '_SpellChecker__edit_distance_alt', autospec=True,
                          side_effect=SpellChecker._SpellChecker__edit_distance_alt) as ed2:
            self.assertEqual(compute_confidence('avilable', True, checker), (0.85, 'available'))
            self.assertEqual(compute_confidence('avlable', True, checker), (0.6, 'available'))  # distance 2 still searched
            ed2.reset_mock()
            ident = 'watchlistscreeningindividualreviewcreaterequest'
            self.assertGreater(len(ident), MAX_EDIT2_LENGTH)
            self.assertEqual(compute_confidence(ident, True, checker), (0.24, None))
            ed2.assert_not_called()
            # distance-1 typo on a long word still gets its suggestion
            confidence, suggestion = compute_confidence('internationalizationn', True, checker)
            self.assertEqual((confidence, suggestion is not None), (0.51, True))

    def test_json_model_matches_sklearn_and_pkl_refused(self):
        try:
            import numpy as np
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            self.skipTest('scikit-learn not installed')
        from alas_but_one.training.features import extract
        from alas_but_one.training.predictor import MLPredictor
        from alas_but_one.training.trainer import load_labeled_jsonl, train
        rows = [
            {'word': w, 'misspelled': m, 'confidence': c, 'label': label,
             'locations': [{'file': 'a.rst', 'line': 1}]}
            for w, m, c, label in [('teh', True, .9, 'true_positive'), ('recieve', True, .8, 'true_positive'),
                               ('API', False, .1, 'false_positive'), ('kubectl', True, .3, 'false_positive')] * 3
        ]
        checker = Mock()
        checker.correction.return_value = None
        with tempfile.TemporaryDirectory() as directory, patch('alas_but_one.training.features._get_checker', return_value=checker):
            labels = Path(directory) / 'l.jsonl'
            labels.write_text('\n'.join(json.dumps(r) for r in rows))
            out = str(Path(directory) / 'm.json')
            with redirect_stdout(io.StringIO()):
                self.assertTrue(train(str(labels), out, min_samples=4))
            X, y = load_labeled_jsonl(str(labels))
            ref = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(np.array(X), np.array(y))
            predictor = MLPredictor(out)
            self.assertIsNone(predictor.problem)
            token = Token(text='teh', repo='r', locations=[TokenLocation(filename='a.rst', line=1)],
                          misspelled=True, confidence=.9)
            self.assertAlmostEqual(predictor.predict_confidence(token),
                                   ref.predict_proba(np.array([extract(token)]))[0][1], places=3)
            with self.assertRaisesRegex(ValueError, r'--train'):
                MLPredictor(str(Path(directory) / 'classifier.pkl'))

    def test_bad_model_warns_once_and_falls_back(self):
        from alas_but_one.training.features import FEATURE_NAMES
        from alas_but_one.training.predictor import MLPredictor
        n = len(FEATURE_NAMES)
        token = Token(text='teh', repo='r', locations=[TokenLocation(filename='a.rst', line=1)],
                      misspelled=True, confidence=.9)
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / 'classifier.json'
            model.write_text('{"intercept": 0}')
            bad = MLPredictor(str(model))
            self.assertFalse(bad.available)
            self.assertRegex(bad.problem, r'classifier\.json.*--train')
            self.assertEqual(bad.apply({'teh': token})['teh'].confidence, .9)
            ok, z, o = [0] * n, [0] * (n - 1), [1] * n
            for label, bad_model in [
                    ('short', {'intercept': 0, 'weights': z, 'mean': ok, 'scale': o}),
                    ('zero scale', {'intercept': 0, 'weights': ok, 'mean': ok, 'scale': ok}),
                    ('string weights', {'intercept': 0, 'weights': 'a' * n, 'mean': ok, 'scale': o}),
                    ('null scale', {'intercept': 0, 'weights': ok, 'mean': ok, 'scale': [None] * n}),
                    ('bool intercept', {'intercept': True, 'weights': ok, 'mean': ok, 'scale': o}),
                    ('reordered', {'intercept': 0, 'weights': ok, 'mean': ok, 'scale': o, 'features': ['x']})]:
                model.write_text(json.dumps(bad_model))
                self.assertRegex(MLPredictor(str(model)).problem or '', r'classifier\.json.*--train', label)
            model.write_text('[' * 100000)
            self.assertRegex(MLPredictor(str(model)).problem or '', r'--train')
            model.write_text(json.dumps({'intercept': 0, 'weights': [0] * n, 'mean': [0] * n, 'scale': [1] * n}))
            good = MLPredictor(str(model))
            self.assertIsNone(good.problem)
            self.assertTrue(good.available)
            self.assertEqual(good.predict_confidence(token), .5)
            model.unlink()
            (Path(directory) / 'classifier.pkl').write_bytes(b'x')
            old = MLPredictor(str(model))
            self.assertRegex(old.problem, r'classifier\.pkl.*--train')
            self.assertFalse(old.available)
            (Path(directory) / 'classifier.pkl').unlink()
            self.assertIsNone(MLPredictor(str(model)).problem)

    def test_huge_int_model_is_a_problem_not_a_crash(self):
        from alas_but_one.training.features import FEATURE_NAMES
        from alas_but_one.training.predictor import MLPredictor
        n = len(FEATURE_NAMES)
        huge = '9' * 400
        z, o = [0] * n, [1] * n
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / 'classifier.json'
            w = ', '.join([huge] + ['0'] * (n - 1))
            for label, text in [
                    ('intercept', '{"intercept": %s, "weights": %s, "mean": %s, "scale": %s}' % (huge, z, z, o)),
                    ('weights', '{"intercept": 0, "weights": [%s], "mean": %s, "scale": %s}' % (w, z, o))]:
                model.write_text(text)
                predictor = MLPredictor(str(model))
                self.assertRegex(predictor.problem or '', r'classifier\.json.*--train', label)
                self.assertFalse(predictor.available)

    def test_nonfinite_score_keeps_heuristic_confidence(self):
        from alas_but_one.training.features import FEATURE_NAMES
        from alas_but_one.training.predictor import MLPredictor
        n = len(FEATURE_NAMES)
        token = Token(text='teh', repo='r', locations=[TokenLocation(filename='a.rst', line=1)],
                      misspelled=True, confidence=.9)
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / 'classifier.json'
            model.write_text(json.dumps({'intercept': 0, 'weights': [1e308, -1e308] + [0] * (n - 2),
                                         'mean': [0] * n, 'scale': [1e-300] * n}))
            predictor = MLPredictor(str(model))
            self.assertIsNone(predictor.problem)
            self.assertEqual(predictor.apply({'teh': token})['teh'].confidence, .9)

    def test_scan_warns_once_on_bad_model_and_still_succeeds(self):
        import alas_but_one.cli as alas
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / 'classifier.json'
            model.write_text('{"intercept": 0}')
            config = {'settings': {'training': {'model_path': str(model)}}, 'modules': {},
                      'repositories': {'a': {'name': 'A'}, 'b': {'name': 'B'}}}
            out, err = io.StringIO(), io.StringIO()
            with patch('sys.argv', ['alas.py']), patch('alas_but_one.cli.resolve_config_path', return_value='config.json'), \
                    patch('alas_but_one.cli.load_config', return_value=config), patch('alas_but_one.cli.run_repo', return_value='out'), \
                    redirect_stdout(out), redirect_stderr(err):
                alas.main()
            self.assertEqual(err.getvalue().count('Warning: ML model'), 1)
            self.assertIn('classifier.json', err.getvalue())
            self.assertNotIn('ML model loaded', out.getvalue())

    def test_casing_exports_training_round_trip_and_legacy(self):
        from alas_but_one.training.features import extract
        from alas_but_one.training.trainer import load_labeled_jsonl
        from contextlib import chdir
        tokens = TokenizerTask({}, {'name': 'test'}).run({'sample.txt': 'HTTP http HTTP'})
        tokens['http'].label = 'false_positive'
        checker = Mock()
        checker.correction.return_value = None
        with tempfile.TemporaryDirectory() as directory, chdir(directory), patch('alas_but_one.training.features._get_checker', return_value=checker):
            path = JsonlFormatterTask({}, {'name': 'test'}).run(tokens)
            record = json.loads(Path(path).read_text())
            self.assertEqual(record['uppercase_occurrences'], 2)
            X, y = load_labeled_jsonl(path)
            self.assertEqual(X, [extract(tokens['http'])])
            self.assertEqual(X[0][5], 2/3)
            self.assertEqual(len(X[0]), 7)
            self.assertEqual(y, [0])
            path = CsvFormatterTask({}, {'name': 'test'}).run(tokens)
            with open(path) as file:
                self.assertEqual(next(csv.DictReader(file))['uppercase_occurrences'], '2')
            del record['uppercase_occurrences']
            for word, ratio in [('HTTP', 1), ('http', 0), ('Http', 0)]:
                record['word'] = word
                Path('legacy.jsonl').write_text(json.dumps(record)+'\n')
                self.assertEqual(load_labeled_jsonl('legacy.jsonl')[0][0][5], ratio)


    def test_mongo_settings_precedence_and_validation(self):
        from alas_but_one.ignore_list_store import resolve_ignore_list_settings, load_words, apply_decisions
        settings = {'MONGODB_URI': 'mongodb://config', 'ignore_list': {'database': 'db', 'collection': 'words'}}
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_ignore_list_settings(settings), ('mongodb://config', 'db', 'words'))
        with patch.dict(os.environ, {'ABO_MONGO_URI': 'mongodb://override'}, clear=True):
            self.assertEqual(resolve_ignore_list_settings(settings), ('mongodb://override', 'db', 'words'))
        for value in ['', '  ']:
            with patch.dict(os.environ, {'ABO_MONGO_URI': value}, clear=True), patch('alas_but_one.ignore_list_store.MongoClient') as client:
                for operation in [lambda: load_words('repo', settings), lambda: apply_decisions('repo', {'word': True}, settings)]:
                    with self.assertRaisesRegex(ValueError, 'ABO_MONGO_URI'):
                        operation()
                client.assert_not_called()
        for invalid in [{}, {'MONGODB_URI': 'mongodb://secret', 'ignore_list': {'database': '', 'collection': 'words'}}, {'MONGODB_URI': 1, 'ignore_list': {'database': 'db', 'collection': 'words'}}]:
            with patch.dict(os.environ, {}, clear=True), patch('alas_but_one.ignore_list_store.MongoClient') as client:
                with self.assertRaises(ValueError) as caught:
                    load_words('repo', invalid)
                self.assertNotIn('secret', str(caught.exception))
                client.assert_not_called()

    def test_mongo_operations_select_same_store_and_close_clients(self):
        from alas_but_one.ignore_list_store import load_words, apply_decisions
        settings = {'MONGODB_URI': 'mongodb://config', 'ignore_list': {'database': 'db', 'collection': 'words'}}
        # In-memory Mongo boundary executes the concrete UpdateOne documents.
        state = {'other': {'untouched'}}
        collection = Mock()
        collection.find_one.side_effect = lambda query: {'words': list(state.get(query['repo_name'], set()))}
        def write(operations):
            for op in operations:
                words = state.setdefault(op._filter['repo_name'], set())
                if '$addToSet' in op._doc:
                    self.assertTrue(op._upsert)
                    words.update(op._doc['$addToSet']['words']['$each'])
                if '$pullAll' in op._doc:
                    self.assertFalse(op._upsert)
                    words.difference_update(op._doc['$pullAll']['words'])
        collection.bulk_write.side_effect = write
        client = Mock()
        database = Mock()
        client.__getitem__ = Mock(return_value=database)
        database.__getitem__ = Mock(return_value=collection)
        with patch.dict(os.environ, {'ABO_MONGO_URI': 'mongodb://override'}, clear=True), patch('alas_but_one.ignore_list_store.MongoClient', return_value=client) as constructor:
            apply_decisions('repo', {'term': True, 'remove': True}, settings)
            apply_decisions('repo', {'term': True, 'remove': False}, settings)
            self.assertEqual(load_words('repo', settings), {'term'})
            self.assertEqual(load_words('other', settings), {'untouched'})
            self.assertEqual(constructor.call_count, 4)
            for call in constructor.call_args_list:
                self.assertEqual(call.args, ('mongodb://override',))
            for call in client.__getitem__.call_args_list:
                self.assertEqual(call.args, ('db',))
            for call in database.__getitem__.call_args_list:
                self.assertEqual(call.args, ('words',))
            self.assertEqual(client.close.call_count, 4)
            collection.find_one.side_effect = RuntimeError('read failed')
            with self.assertRaises(RuntimeError):
                load_words('repo', settings)
            self.assertEqual(client.close.call_count, 5)
            collection.bulk_write.side_effect = RuntimeError('write failed')
            with self.assertRaises(RuntimeError):
                apply_decisions('repo', {'term': True}, settings)
            self.assertEqual(client.close.call_count, 6)

    def test_ignore_consumers_use_shared_operations(self):
        from alas_but_one.matchers.ignore_list_matcher import IgnoreListTask
        import alas_but_one.save_ignore_list as save_ignore_list
        tokens = {'approved': Token('approved', 'repo', []), 'typo': Token('typo', 'repo', [])}
        with patch('alas_but_one.matchers.ignore_list_matcher.load_words', return_value={'approved'}) as load:
            result = IgnoreListTask({}, {'name': 'repo'}).run(tokens)
            self.assertEqual(result['approved'].ignore, 'Y')
            self.assertEqual(result['typo'].ignore, 'N')
            load.assert_called_once_with('repo', {})
        with patch('alas_but_one.save_ignore_list.apply_decisions') as apply:
            save_ignore_list._apply_updates({'repo': {'approved': False}, 'other': {'term': True}}, {})
            self.assertEqual(apply.call_count, 2)
            apply.assert_any_call('repo', {'approved': False}, {})
            apply.assert_any_call('other', {'term': True}, {})


    def test_rst_masks_non_prose_without_moving_source_lines(self):
        from alas_but_one.tokenizer.tokenize_rst import mask_rst
        cases = [
            ('``inlineNoise``\nretreive', {'inlinenoise'}),
            ('.. _targetNoise: https://example.test/urlNoise\n   continuedNoise\n\nretreive', {'targetnoise','urlnoise','continuednoise'}),
            ('.. code-block:: python\n   :linenos:\n\n   codeNoise\n     nestedNoise\n\nretreive', {'python','linenos','codenoise','nestednoise'}),
            ('.. code:: python\n\n   codeNoise\nretreive', {'codenoise','python'}),
            ('.. sourcecode:: python\n   codeNoise\nretreive', {'codenoise','python'}),
            ('Use this example::\n\n   literalNoise\n\nretreive', {'literalnoise'}),
            ('.. commentNoise\n   continuedNoise\n\nretreive', {'commentnoise','continuednoise'}),
            ('Visit https://example.test/urlNoise\nretreive', {'urlnoise','https','test'}),
        ]
        for escaped, excluded in cases:
            content = escaped.replace('\\n', '\n')
            with self.subTest(content=content):
                masked = mask_rst(content)
                self.assertEqual(len(masked), len(content))
                self.assertEqual([i for i,c in enumerate(masked) if c=='\n'], [i for i,c in enumerate(content) if c=='\n'])
                tokens = TokenizerTask({}, {'name':'test'}).run({'sample.rst':content})
                self.assertFalse(excluded & tokens.keys())
                self.assertEqual(tokens['retreive'].locations[0].line, len(content.split('\n')))

    def test_rst_preserves_admonitions_lists_headings_and_link_labels(self):
        content = ('retreive\n========\n\n.. note::\n   :class: hiddenOption\n\n'
                   '   mispeled\n\n.. warning::\n\n   - recieve\n     - seperate\n\n'
                   '`occurrance guide <https://example.test/destinationNoise>`_\n'
                   '*definately*\n\n.. custom::\n   :option: hiddenOption\n\n   lasttypoo\n')
        tokens = TokenizerTask({}, {'name':'test'}).run({'sample.rst':content})
        for word in ['retreive','mispeled','recieve','seperate','occurrance','definately','lasttypoo']:
            expected = next(i for i,line in enumerate(content.split('\n'),1) if word in line)
            self.assertEqual(tokens[word].locations[0].line, expected)
        self.assertNotIn('destinationnoise', tokens)
        self.assertNotIn('hiddenoption', tokens)
        self.assertIn('guide', tokens)

    def test_text_format_defaults_overrides_and_malformed_rst(self):
        content = '``codeNoise``\nretreive'
        for filename, override, excluded in [('sample.txt', None, False), ('sample.rst', None, True), ('sample.txt','rst',True), ('sample.rst','plain',False)]:
            repo = {'name':'test'}
            if override:
                repo['text_format'] = override
            tokens = TokenizerTask({}, repo).run({filename:content})
            self.assertEqual('codenoise' not in tokens, excluded)
            self.assertEqual(tokens['retreive'].locations[0].line, 2)
        for value in ['unsupported', '', None]:
            with self.assertRaisesRegex(ValueError, 'text_format'):
                TokenizerTask({}, {'name':'test','text_format':value}).run({'sample.rst':content})
        malformed = '``unclosed retreive\nmispeled\n`recieve <unfinished\n.. |term| replace:: seperate'
        tokens = TokenizerTask({}, {'name':'test'}).run({'sample.rst':malformed})
        self.assertTrue({'retreive','mispeled','recieve','seperate'} <= tokens.keys())

    def test_nested_code_blocks_do_not_hide_following_list_prose(self):
        content = ('- retreive\n\n  .. code-block:: python\n\n     codeNoise\n\n'
                   '  mispeled\n\n- recieve::\n\n    literalNoise\n\n  seperate\n')
        tokens = TokenizerTask({}, {'name':'test'}).run({'sample.rst':content})
        self.assertTrue({'retreive','mispeled','recieve','seperate'} <= tokens.keys())
        self.assertFalse({'codenoise','literalnoise'} & tokens.keys())


    def test_pipeline_default_and_audit_exports_preserve_scores(self):
        from contextlib import chdir
        from alas_but_one.cli import run_repo, load_config
        from alas_but_one.save_ignore_list import _load_jsonl, _load_csv
        config = load_config()
        with tempfile.TemporaryDirectory() as directory, chdir(directory):
            Path('sample.txt').write_text('MongoDB retreive ordinary')
            repo = {'name':'test','relative_path':'','source_dir':''}
            settings = {**config['settings'], 'repo_base_full_path':directory + '/', 'ai':{'enabled':False}, 'output_dir':directory}
            for output_format in ['jsonl','csv']:
                for include_ignored in [False, True]:
                    with patch('alas_but_one.matchers.ignore_list_matcher.load_words', return_value={'mongodb'}):
                        path = run_repo('test', repo, settings, config.get('modules', {}), output_format, False, Mock(available=False, problem=None), False, include_ignored=include_ignored)
                    if output_format == 'jsonl':
                        rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
                        decisions = _load_jsonl(path)
                    else:
                        with open(path) as file:
                            rows = list(csv.DictReader(file))
                        decisions = _load_csv(path)
                    by_word = {r['word']:r for r in rows}
                    self.assertEqual('mongodb' in by_word, include_ignored)
                    self.assertEqual(float(by_word['ordinary']['confidence']), 0.05)
                    self.assertIn('retreive', by_word)
                    if include_ignored:
                        self.assertTrue(decisions['test']['mongodb'])
                        by_word['mongodb']['ignore'] = False if output_format=='jsonl' else 'N'
                        if output_format=='jsonl':
                            Path(path).write_text(''.join(json.dumps(r)+'\n' for r in rows))
                        else:
                            with open(path,'w') as file:
                                writer=csv.DictWriter(file,fieldnames=list(rows[0]))
                                writer.writeheader()
                                writer.writerows(rows)
                        loader = _load_jsonl if output_format=='jsonl' else _load_csv
                        self.assertFalse(loader(path)['test']['mongodb'])

    def test_verbose_prints_per_stage_timing_only_when_verbose(self):
        import io
        from contextlib import chdir, redirect_stdout
        from alas_but_one.cli import run_repo, load_config
        config = load_config()
        with tempfile.TemporaryDirectory() as directory, chdir(directory):
            Path('sample.txt').write_text('retreive ordinary')
            repo = {'name':'test','relative_path':'','source_dir':''}
            settings = {**config['settings'], 'repo_base_full_path':directory + '/', 'ai':{'enabled':False}, 'output_dir':directory}
            outputs = {}
            for verbose in [True, False]:
                buffer = io.StringIO()
                with redirect_stdout(buffer), patch('alas_but_one.matchers.ignore_list_matcher.load_words', return_value=set()):
                    run_repo('test', repo, settings, config.get('modules', {}), 'jsonl', False, Mock(available=False, problem=None), verbose)
                outputs[verbose] = buffer.getvalue()
            for stage in ['collector','reader','tokenizer','max_occurrence_matcher','spell_checker','ignore_list_matcher']:
                self.assertRegex(outputs[True], rf'    {stage}: \d+\.\d\ds')
            self.assertNotRegex(outputs[False], r'\d\.\d\ds')

    def test_ai_payload_never_includes_approved_terms(self):
        approved = Token('mongodb', 'test', [], confidence=0.4, ignore='Y')
        typo = Token('retreive', 'test', [], confidence=0.4)
        reviewer = AIReviewer({'ai':{'enabled':True}})
        client = Mock()
        client.messages.create.return_value.content = [Mock(type='text', text='{"results":[{"word":"retreive","is_typo":true,"confidence":0.9,"suggestion":null,"comment":"c"}]}')]
        with patch.object(reviewer, '_get_client', return_value=client):
            reviewer.run({'mongodb':approved, 'retreive':typo}, {})
        user_message = client.messages.create.call_args.kwargs['messages'][0]['content']
        self.assertNotIn('mongodb', user_message)
        self.assertIn('retreive', user_message)
        self.assertFalse(approved.ai_reviewed)
        self.assertTrue(typo.ai_reviewed)

    def test_cli_forwards_ignore_flag_serial_and_parallel(self):
        import alas_but_one.cli as alas
        config = {'settings':{},'modules':{},'repositories':{'a':{'name':'A'},'b':{'name':'B'}}}
        for flags, expected in [([],False), (['--include-ignored'],True), (['--include-ignored','--parallel'],True)]:
            with patch('sys.argv', ['alas.py']+flags), patch('alas_but_one.cli.resolve_config_path', return_value='config.json'), patch('alas_but_one.cli.load_config', return_value=config), patch('alas_but_one.cli.MLPredictor', return_value=Mock(available=False, problem=None)), patch('alas_but_one.cli.run_repo', return_value='output') as run:
                alas.main()
                self.assertEqual(run.call_count, 2)
                for call in run.call_args_list:
                    self.assertEqual(call.kwargs['include_ignored'], expected)


    def test_evaluation_metrics_counts_empty_queue_and_unlabeled_failure(self):
        from evaluate_candidates import summarize
        labels = {'typo':{'label':'true_positive'}, 'prose':{'label':'false_positive'}}
        metrics = summarize([{'word':'typo','confidence':0.8}, {'word':'prose','confidence':0.1}], labels)
        self.assertEqual(metrics['actionable_candidates'], 2)
        self.assertEqual(metrics['true_positives'], 1)
        self.assertEqual(metrics['false_positives'], 1)
        self.assertEqual(metrics['precision'], 0.5)
        self.assertEqual(metrics['top_10'], {'candidates':2, 'true_positives':1, 'precision':0.5})
        self.assertEqual(metrics['seeded_typos'], {'retained':1,'total':1,'retention':1.0})
        empty = summarize([], labels)
        self.assertIsNone(empty['precision'])
        self.assertIsNone(empty['top_10']['precision'])
        self.assertEqual(empty['seeded_typos']['retained'], 0)
        with self.assertRaisesRegex(ValueError, 'Unlabeled.*unknown'):
            summarize([{'word':'unknown'}], labels)

    def test_evaluation_comparison_checks_all_required_invariants(self):
        from evaluate_candidates import compare
        labels = {'typo':{'label':'true_positive','category':'prose_typo','source_lines':[1]}, 'noise':{'label':'false_positive','category':'non_prose','source_lines':[2]}, 'approved':{'label':'false_positive','category':'approved_term','source_lines':[3]}}
        before = [{'word':word,'confidence':0.4,'locations':[{'file':'evaluation.rst','line':line}]} for word,line in [('typo',0),('noise',1),('approved',2)]]
        after = [{'word':'typo','confidence':0.4,'locations':[{'file':'evaluation.rst','line':1}]}]
        self.assertTrue(compare(before,after,labels)['acceptance_passed'])
        for invalid in [[], [{**after[0],'locations':[{'line':0}]}], after+[{'word':'noise','confidence':0.4,'locations':[{'line':2}]}], after+[{'word':'approved','confidence':0.4,'locations':[{'line':3}]}]]:
            with self.assertRaises(ValueError):
                compare(before,invalid,labels)


    def test_inline_literals_allow_single_backticks_and_stop_at_paragraphs(self):
        from alas_but_one.tokenizer.tokenize_rst import mask_rst
        cases = [
            ('Use ``foo`codeNoise`` here.\nretreive', 2),
            ('An unclosed ``literal\n\nretreive in real prose\n\nUse ``codeNoise`` here.\nmispeled', 3),
            ('Use ``foo\ncodeNoise`` here.\nretreive', 3),
        ]
        for content, line in cases:
            with self.subTest(content=content):
                tokens = TokenizerTask({}, {'name':'test'}).run({'sample.rst':content})
                self.assertEqual(tokens['retreive'].locations[0].line if 'retreive' in tokens else None, line)
                self.assertNotIn('codenoise', tokens)
                self.assertEqual(len(mask_rst(content)), len(content))
        content = 'An unclosed ``literal\r\n \t\r\nretreive\r\n\r\n``codeNoise``'
        tokens = TokenizerTask({}, {'name':'test'}).run({'sample.rst':content})
        self.assertEqual(tokens['retreive'].locations[0].line, 3)
        self.assertNotIn('codenoise', tokens)

    def test_tabs_mask_blocks_without_changing_source_offsets(self):
        from alas_but_one.tokenizer.tokenize_rst import mask_rst
        for content, line in [
            ('- Example::\n\n\tcodeNoise\n\n  retreive', 5),
            ('\t.. code-block:: python\n\n\t    codeNoise\n\n\tretreive', 5),
        ]:
            tokens = TokenizerTask({}, {'name':'test'}).run({'sample.rst':content})
            self.assertNotIn('codenoise', tokens)
            self.assertEqual(tokens['retreive'].locations[0].line, line)
            masked = mask_rst(content)
            self.assertEqual(len(masked), len(content))
            self.assertEqual([i for i,c in enumerate(masked) if c=='\n'], [i for i,c in enumerate(content) if c=='\n'])


    def test_collector_and_markdown_format_defaults(self):
        from alas_but_one.collectors.filter_files import CollectorTask
        with tempfile.TemporaryDirectory() as directory:
            for name in ['a.md','b.rst','c.txt','d.png']:
                Path(directory,name).write_text('')
            self.assertEqual({Path(p).name for p in CollectorTask({},{}).run(directory)}, {'a.md','b.rst','c.txt'})
        content = '`codeNoise` retreive'
        for filename, override, masked in [('a.md',None,True), ('a.txt','markdown',True), ('a.md','plain',False)]:
            repo={'name':'test'}
            if override:
                repo['text_format']=override
            tokens=TokenizerTask({},repo).run({filename:content})
            self.assertEqual('codenoise' not in tokens, masked)
            self.assertIn('retreive',tokens)

    def test_markdown_masks_code_metadata_and_destinations(self):
        from alas_but_one.tokenizer.tokenize_rst import mask_markdown
        cases=[
            ('---\ntitle: metadataNoise\n---\nretreive', {'metadatanoise','title'},4),
            ('# Source: https://example.test\n\n---\ntitle: metadataNoise\n---\nretreive', {'metadatanoise','title'},6),
            ('```python\ncodeNoise\n```\nretreive', {'python','codenoise'},4),
            ('~~~~\ncodeNoise\n~~~\nstillCodeNoise\n~~~~\nretreive', {'codenoise','stillcodenoise'},6),
            ('> ```python\n> codeNoise\n> ```\n> retreive', {'codenoise','python'},4),
            ('Prose\n\n    codeNoise\n\nretreive', {'codenoise'},5),
            ('Use `codeNoise` and ``foo`otherNoise``.\nretreive', {'codenoise','othernoise','foo'},2),
            ('<!-- commentNoise\ncontinuedNoise -->\nretreive', {'commentnoise','continuednoise'},3),
            ('<CodeTab label="install" lang="bash">{"codeNoise"}</CodeTab>\nretreive', {'codenoise'},2),
            ('[retreive](https://example.test/destinationNoise)\n[refNoise]: /targetNoise', {'destinationnoise','refnoise','targetnoise'},1),
            ('[retreive][refNoise]\n[refNoise]: /targetNoise', {'refnoise','targetnoise'},1),
            ('Visit https://example.test/urlNoise\nretreive', {'urlnoise','https'},2),
        ]
        for content, excluded, line in cases:
            with self.subTest(content=content):
                masked=mask_markdown(content)
                self.assertEqual(len(masked),len(content))
                self.assertEqual([i for i,c in enumerate(masked) if c=='\n'], [i for i,c in enumerate(content) if c=='\n'])
                tokens=TokenizerTask({}, {'name':'test'}).run({'a.md':content})
                self.assertFalse(excluded & tokens.keys())
                self.assertEqual(tokens['retreive'].locations[0].line,line)

    def test_markdown_preserves_visible_prose_and_malformed_markup(self):
        content=('# retreive\n\n- mispeled\n  - recieve\n\n    continuedProse\n\n'
                 '<Banner variant="info">seperate</Banner>\n'
                 '<Image src="/imageNoise.png"\n alt="occurrance" title="definately" />\n'
                 '![lasttypoo](/imageNoise.png)\n\n'
                 'An unclosed `inlineLiteral\n\nrealProse\n\nUse `codeNoise` here.\n'
                 'Unclosed <tagNoise\n\nlastProse')
        tokens=TokenizerTask({}, {'name':'test'}).run({'a.md':content})
        for word in ['retreive','mispeled','recieve','continuedprose','seperate','occurrance','definately','lasttypoo','realprose','lastprose']:
            self.assertIn(word,tokens)
            expected=next(i for i,line in enumerate(content.lower().split('\n'),1) if word in line)
            self.assertEqual(tokens[word].locations[0].line,expected)
        self.assertFalse({'banner','variant','imagenoise','codenoise'} & tokens.keys())


    def test_markdown_fences_and_list_containers_preserve_prose(self):
        cases=[
            ('- ```python\n  codeNoise\n  ```\n\nretreive',5),
            ('> ```python\n> codeNoise\n\nretreive',4),
            ('```codeNoise``` is retreive\n\nmispeled',1),
            ('- outer\n  - inner\n\n  outer continuation\n\n    retreive',6),
            ('- Example:\n\n      codeNoise\n\n  retreive',5),
        ]
        for content,line in cases:
            with self.subTest(content=content):
                tokens=TokenizerTask({}, {'name':'test'}).run({'a.md':content})
                self.assertIn('retreive',tokens)
                self.assertEqual(tokens['retreive'].locations[0].line,line)
                self.assertNotIn('codenoise',tokens)

    def test_markdown_escaped_delimiters_and_html_prose(self):
        cases=[
            (r'Use \`retreive\` here.',1),
            ('<Image alt="Use `retreive` now" />',1),
            ('Unclosed <tagNoise\n\nretreive is visible >\nmispeled',3),
        ]
        for content,line in cases:
            with self.subTest(content=content):
                tokens=TokenizerTask({}, {'name':'test'}).run({'a.md':content})
                self.assertIn('retreive',tokens)
                self.assertEqual(tokens['retreive'].locations[0].line,line)


    def test_real_corpus_scanner_reproduces_original_and_markdown_stages(self):
        import subprocess
        import sys
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            corpus=Path(directory)/'corpus'
            corpus.mkdir()
            (corpus/'sample.md').write_text('```\nbanana\n```\nretreive')
            for baseline in [False,True]:
                output=Path(directory)/'output.jsonl'
                command=[sys.executable,str(root/'tests/scan_real_corpus.py'),'--corpus',str(corpus),'--output',str(output)]
                if baseline:
                    command.append('--baseline')
                subprocess.run(command,check=True,capture_output=True,text=True)
                records={r['word']:r for r in map(json.loads,output.read_text().splitlines())}
                self.assertEqual('banana' in records,baseline)
                self.assertEqual(records['retreive']['locations'],[{'file':'sample.md','line':3 if baseline else 4}])


    def test_real_corpus_review_requires_labels_and_correct_locations(self):
        from evaluate_real_corpus import evaluate_review
        labels={'typo':{'word':'typo','label':'true_positive','category':'prose_typo','source_contexts':[{'stage':'after','file':'a.md','line':2}]},'noise':{'word':'noise','label':'false_positive','category':'code_literal','source_contexts':[]}}
        before=[{'word':'typo','confidence':0.8,'locations':[{'file':'a.md','line':1}]},{'word':'noise','confidence':0.9,'locations':[{'file':'a.md','line':0}]}]
        after=[{'word':'typo','confidence':0.8,'locations':[{'file':'a.md','line':2}]}]
        metrics=evaluate_review(before,after,labels,limit=2)
        self.assertEqual(metrics['review_sample']['after']['precision'],1)
        self.assertEqual(metrics['known_prose_typos']['retained'],1)
        self.assertEqual(metrics['corpus']['after_candidates'],1)
        with self.assertRaisesRegex(ValueError,'Unlabeled'):
            evaluate_review(before+[{'word':'unknown','confidence':1,'locations':[]}],after,labels,limit=2)
        with self.assertRaisesRegex(ValueError,'location'):
            evaluate_review(before,[{**after[0],'locations':[{'file':'a.md','line':0}]}],labels,limit=2)
        with self.assertRaisesRegex(ValueError,'Lost'):
            evaluate_review(before,[],labels,limit=2)


    def test_repo_path_resolution_absolute_tilde_relative_and_legacy(self):
        from alas_but_one.config import resolve_repo_dir
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(resolve_repo_dir({'path': directory}, {}, '/elsewhere'), directory)
            self.assertEqual(resolve_repo_dir({'path': directory, 'source_dir': 'source'}, {}, '/x'), os.path.join(directory, 'source'))
            with patch.dict(os.environ, {'HOME': directory}):
                self.assertEqual(resolve_repo_dir({'path': '~/docs'}, {}, '/x'), os.path.join(directory, 'docs'))
            self.assertEqual(resolve_repo_dir({'path': 'docs', 'source_dir': 'source/'}, {}, directory), os.path.join(directory, 'docs', 'source'))
            legacy = {'relative_path': 'docs-golang/', 'source_dir': 'source/'}
            self.assertEqual(resolve_repo_dir(legacy, {'repo_base_full_path': directory + '/'}, '/x'), os.path.join(directory, 'docs-golang', 'source'))
            with self.assertRaises(ValueError):
                resolve_repo_dir({'name': 'x'}, {}, directory)

    def test_cli_config_flag_and_env_select_config_file(self):
        import alas_but_one.cli as alas
        from alas_but_one.config import load_config
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'custom.json')
            Path(path).write_text(json.dumps({'settings': {}, 'modules': {}, 'repositories': {}}))
            config = load_config(path)
            self.assertEqual(config['config_dir'], directory)
            for argv, env in [(['alas.py', '--config', path], {}), (['alas.py'], {'ABO_CONFIG': path})]:
                with patch('sys.argv', argv), patch.dict(os.environ, env, clear=True), patch('alas_but_one.cli.load_config', return_value=config) as load, patch('alas_but_one.cli.MLPredictor', return_value=Mock(available=False, problem=None)):
                    with self.assertRaises(SystemExit):
                        alas.main()
                    load.assert_called_once_with(path)

class TaskFactoryTests(unittest.TestCase):

    def test_legacy_module_path_resolves_and_custom_import_errors_are_not_masked(self):
        from alas_but_one.tasks.factory import TaskFactory
        from alas_but_one.collectors.filter_files import CollectorTask
        legacy = {'collector': {'path': 'collectors.filter_files', 'className': 'CollectorTask'}}
        self.assertIsInstance(TaskFactory.create_task('collector', {}, {'name': 'x'}, legacy), CollectorTask)
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'mytask.py').write_text('import notinstalled_dep\n')
            sys.path.insert(0, directory)
            try:
                custom = {'collector': {'path': 'mytask', 'className': 'X'}}
                with self.assertRaisesRegex(ModuleNotFoundError, 'notinstalled_dep'):
                    TaskFactory.create_task('collector', {}, {'name': 'x'}, custom)
            finally:
                sys.path.remove(directory)
                sys.modules.pop('mytask', None)


class ConfigErrorMessageTests(unittest.TestCase):
    MISSING = 'Config file {} not found. Create it with `alas --init`, or point ABO_CONFIG or --config PATH at an existing one.'

    def _cli(self, *args, script='alas_but_one.cli', cwd=None, env_extra=None):
        import subprocess
        root = str(Path(__file__).resolve().parent.parent)
        env = {k: v for k, v in os.environ.items() if k != 'ABO_CONFIG'}
        env.update(env_extra or {})
        return subprocess.run([sys.executable, '-m', script, *args], cwd=cwd or root, env=env, capture_output=True, text=True)

    def test_invalid_directory_names_repo_directory_and_config(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = os.path.join(directory, 'c.json')
            Path(cfg).write_text(json.dumps({'settings': {}, 'modules': {}, 'repositories': {
                'r': {'name': 'My Docs', 'path': 'nope'}}}))
            result = self._cli('--config', cfg)
            expected = f'Repository My Docs: directory {os.path.join(directory, "nope")} does not exist. Check "path" in {cfg}.'
            self.assertEqual(result.returncode, 1)
            self.assertEqual(result.stderr.strip(), expected)
            self.assertNotIn('Traceback', result.stdout)

    def test_missing_config_file_names_absolute_path_and_fix(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = os.path.realpath(directory)
            result = self._cli('--config', 'absent.json', cwd=directory)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(result.stderr.strip(), self.MISSING.format(os.path.join(directory, 'absent.json')))

    def test_save_ignore_list_missing_config_exits_with_message(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = os.path.realpath(directory)
            Path(directory, 'out.jsonl').write_text('')
            cfg = os.path.join(directory, 'absent.json')
            result = self._cli('out.jsonl', script='alas_but_one.save_ignore_list', cwd=directory, env_extra={'ABO_CONFIG': cfg})
            self.assertEqual(result.returncode, 1)
            self.assertEqual(result.stderr.strip(), self.MISSING.format(cfg))

    def test_malformed_config_json_names_file_and_position(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = os.path.join(directory, 'c.json')
            Path(cfg).write_text('{\n  "settings": }')
            result = self._cli('--config', cfg)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(result.stderr.strip(), f'Config file {cfg} is not valid JSON: line 2 column 15: Expecting value.')

    def test_save_ignore_list_missing_input_exits_with_message(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = os.path.realpath(directory)
            result = self._cli('nope.jsonl', script='alas_but_one.save_ignore_list', cwd=directory)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(result.stderr.strip(), f'Input file {os.path.join(directory, "nope.jsonl")} not found.')

    def _assert_clean_failure(self, result, message):
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stderr.strip(), message)
        self.assertNotIn('Traceback', result.stderr)

    def test_save_ignore_list_bad_jsonl_line_reports_line_number(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = os.path.realpath(directory)
            Path(directory, 'out.jsonl').write_text('{"repo": "r", "word": "a", "ignore": true}\n\n{oops\n')
            result = self._cli('out.jsonl', script='alas_but_one.save_ignore_list', cwd=directory)
            self._assert_clean_failure(result, f'Input file {os.path.join(directory, "out.jsonl")} line 3 is not valid JSON.')

    def test_save_ignore_list_non_text_word_is_a_clean_error(self):
        cases = [
            ('w.jsonl', '{"repo": "r", "word": "a", "ignore": true}\n{"repo": "r", "word": 1, "ignore": true}\n', 2),
            ('m.jsonl', '{"repo": "r", "ignore": true}\n', 1),
            ('w.csv', 'repo,word,ignore\nr,a,Y\nr\n', 3),
        ]
        for name, text, line in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                Path(directory, name).write_text(text)
                result = self._cli(name, script='alas_but_one.save_ignore_list', cwd=directory)
                self._assert_clean_failure(result, f'Input file {os.path.join(os.path.realpath(directory), name)} line {line}: word must be text.')

    def test_save_ignore_list_reads_bom_input(self):
        from alas_but_one.save_ignore_list import _load_csv, _load_jsonl
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory, 'excel.csv')
            csv_path.write_bytes('\ufeffrepo,word,ignore\nr,Recieve,Y\n'.encode('utf-8'))
            jsonl_path = Path(directory, 'bom.jsonl')
            jsonl_path.write_bytes('\ufeff{"repo": "r", "word": "Recieve", "ignore": true}\n'.encode('utf-8'))
            self.assertEqual(_load_csv(str(csv_path)), {'r': {'recieve': True}})
            self.assertEqual(_load_jsonl(str(jsonl_path)), {'r': {'recieve': True}})

    def test_save_ignore_list_non_object_jsonl_line_is_a_clean_error(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = os.path.realpath(directory)
            Path(directory, 'out.jsonl').write_text('{"repo": "r", "word": "a", "ignore": true}\n[1]\n')
            result = self._cli('out.jsonl', script='alas_but_one.save_ignore_list', cwd=directory)
            self._assert_clean_failure(result, f'Input file {os.path.join(directory, "out.jsonl")} line 2 is not a JSON object.')

    def test_save_ignore_list_non_utf8_input_is_a_clean_error(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = os.path.realpath(directory)
            for name in ('bin.jsonl', 'bin.csv'):
                Path(directory, name).write_bytes(b'\xff\xfe\x00\x80\x81')
                result = self._cli(name, script='alas_but_one.save_ignore_list', cwd=directory)
                self._assert_clean_failure(result, f'Input file {os.path.join(directory, name)} is not UTF-8 text.')

    def test_save_ignore_list_input_directory_is_a_clean_error(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = os.path.realpath(directory)
            os.mkdir(os.path.join(directory, 'd.jsonl'))
            result = self._cli('d.jsonl', script='alas_but_one.save_ignore_list', cwd=directory)
            self._assert_clean_failure(result, f'Input file {os.path.join(directory, "d.jsonl")} cannot be read: Is a directory.')

    def test_config_path_directory_is_a_clean_error(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = os.path.realpath(directory)
            result = self._cli('--config', directory)
            self._assert_clean_failure(result, f'Config file {directory} cannot be read: Is a directory.')

    def test_ignore_file_malformed_json_is_a_clean_error(self):
        from alas_but_one.ignore_list_store import IgnoreListError, _read_file
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'i.json')
            Path(path).write_text('{')
            with self.assertRaisesRegex(IgnoreListError, 'ignore file is not valid JSON: line 1 column 2'):
                _read_file(path)
            with self.assertRaisesRegex(IgnoreListError, 'cannot be read: Is a directory'):
                _read_file(directory)

    def test_malformed_mongo_uri_is_a_clean_error_without_credentials(self):
        from alas_but_one.ignore_list_store import IgnoreListError, _mongo
        with self.assertRaises(IgnoreListError) as caught:
            _mongo('mongodb://SECRETUSER:SECRETPW@host:12x/db', lambda client: None)
        self.assertIn('MongoDB URI is invalid: Port contains non-digit characters', str(caught.exception))
        self.assertNotIn('SECRETPW', str(caught.exception))


class FailAboveTests(unittest.TestCase):
    def _run(self, directory, *flags, ignored=()):
        from contextlib import chdir
        import alas_but_one.cli as alas
        base = alas.load_config()
        config = {'settings': {**base['settings'], 'ai': {'enabled': False}, 'output_dir': directory}, 'modules': base.get('modules', {}), 'repositories': {
            'r': {'name': 'test', 'path': directory, 'text_format': 'plain'}}, 'config_dir': directory}
        out, err = io.StringIO(), io.StringIO()
        code = 0
        with chdir(directory), redirect_stdout(out), redirect_stderr(err), patch('sys.argv', ['alas.py', *flags]), \
                patch('alas_but_one.cli.resolve_config_path', return_value='config.json'), patch('alas_but_one.cli.load_config', return_value=config), patch('alas_but_one.cli.MLPredictor', return_value=Mock(available=False, problem=None)), \
                patch('alas_but_one.matchers.ignore_list_matcher.load_words', return_value=set(ignored)):
            try:
                alas.main()
            except SystemExit as e:
                code = e.code
        return code, out.getvalue(), err.getvalue()

    def _corpus(self, directory):
        Path(directory, 'sample.txt').write_text('retreive ordinary')

    def _score(self, directory):
        _, out, _ = self._run(directory, '--fail-above', '0', '--quiet')
        return float(out.splitlines()[0].split('\t')[2])

    def test_exit_code_boundary_is_inclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            self._corpus(directory)
            score = self._score(directory)
            self.assertGreater(score, 0.5)
            self.assertEqual(self._run(directory, '--fail-above', f'{score:.2f}')[0], 1)
            self.assertEqual(self._run(directory, '--fail-above', f'{score + 0.01:.2f}')[0], 0)
            self.assertTrue(os.path.exists(os.path.join(directory, 'r.jsonl')))

    def test_ignored_candidates_never_fail_even_with_include_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            self._corpus(directory)
            for flags in [(), ('--include-ignored',)]:
                code, out, _ = self._run(directory, '--fail-above', '0', '--quiet', *flags, ignored={'retreive'})
                self.assertEqual(out.count('retreive'), 0)
                self.assertEqual(self._run(directory, '--fail-above', '0.5', *flags, ignored={'retreive'})[0], 0)
            self.assertIn('retreive', Path(directory, 'r.jsonl').read_text())

    def test_quiet_prints_only_qualifying_lines_and_requires_fail_above(self):
        with tempfile.TemporaryDirectory() as directory:
            self._corpus(directory)
            score = self._score(directory)
            code, out, err = self._run(directory, '--fail-above', f'{score:.2f}', '--quiet')
            self.assertEqual(out, f"test\tretreive\t{score:.2f}\t{os.path.join(directory, 'sample.txt')}:1\n")
            self.assertEqual(self._run(directory, '--fail-above', '0.99', '--quiet')[1], '')
            code, out, err = self._run(directory, '--quiet')
            self.assertEqual((code, out), (2, ''))
            self.assertIn('--quiet requires --fail-above', err)
            self.assertEqual(self._run(directory, '--fail-above', '1.5')[0], 2)

    def test_parallel_aggregates_across_repos(self):
        import alas_but_one.cli as alas
        config = {'settings': {}, 'modules': {}, 'repositories': {'a': {'name': 'A'}, 'b': {'name': 'B'}}}
        def fake(name, repo_config, *a, hits=None, fail_above=None, **k):
            if repo_config['name'] == 'B':
                hits.append(('B', Token('x', 'B', [TokenLocation('f', 1)], confidence=0.9)))
            return 'out'
        for flags in [(), ('--parallel',)]:
            with patch('sys.argv', ['alas.py', '--fail-above', '0.8', *flags]), patch('alas_but_one.cli.resolve_config_path', return_value='config.json'), patch('alas_but_one.cli.load_config', return_value=config), \
                    patch('alas_but_one.cli.MLPredictor', return_value=Mock(available=False, problem=None)), patch('alas_but_one.cli.run_repo', side_effect=fake), \
                    redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    alas.main()
            self.assertEqual(raised.exception.code, 1)

    def test_parallel_repo_failure_fails_the_gate_quietly(self):
        import alas_but_one.cli as alas
        config = {'settings': {}, 'modules': {}, 'repositories': {'a': {'name': 'A'}, 'b': {'name': 'B'}}}
        def fake(name, repo_config, *a, **k):
            if repo_config['name'] == 'B':
                raise ValueError('Invalid directory: nope')
            return 'out'
        out, err = io.StringIO(), io.StringIO()
        with patch('sys.argv', ['alas.py', '--fail-above', '0.8', '--quiet', '--parallel']), patch('alas_but_one.cli.resolve_config_path', return_value='config.json'), patch('alas_but_one.cli.load_config', return_value=config), \
                patch('alas_but_one.cli.MLPredictor', return_value=Mock(available=False, problem=None)), patch('alas_but_one.cli.run_repo', side_effect=fake), \
                redirect_stdout(out), redirect_stderr(err):
            with self.assertRaises(SystemExit) as raised:
                alas.main()
        self.assertEqual((raised.exception.code, out.getvalue()), (1, ''))
        self.assertIn('B: FAILED', err.getvalue())
    def test_failed_repo_does_not_stop_others_and_exits_1(self):
        import alas_but_one.cli as alas
        config = {'settings': {}, 'modules': {}, 'repositories': {
            'a': {'name': 'A'}, 'b': {'name': 'B'}, 'c': {'name': 'C'}}}
        for flags in [(), ('--parallel',)]:
            scanned = []
            def fake(name, repo_config, *a, **k):
                scanned.append(repo_config['name'])
                if repo_config['name'] == 'A':
                    raise ValueError('Invalid directory: nope')
                return 'out'
            out, err = io.StringIO(), io.StringIO()
            with patch('sys.argv', ['alas.py', *flags]), patch('alas_but_one.cli.resolve_config_path', return_value='config.json'), patch('alas_but_one.cli.load_config', return_value=config), \
                    patch('alas_but_one.cli.MLPredictor', return_value=Mock(available=False, problem=None)), patch('alas_but_one.cli.run_repo', side_effect=fake), \
                    redirect_stdout(out), redirect_stderr(err), self.assertLogs('alas_but_one.cli', 'DEBUG'):
                with self.assertRaises(SystemExit) as raised:
                    alas.main()
            self.assertEqual((raised.exception.code, sorted(scanned)), (1, ['A', 'B', 'C']), flags)
            self.assertIn('A: FAILED (Invalid directory: nope)', err.getvalue())
            self.assertIn('B -> out', out.getvalue())
            self.assertIn('C -> out', out.getvalue())

    def test_quiet_serial_prints_hits_from_repos_that_finished_despite_failure(self):
        import alas_but_one.cli as alas
        config = {'settings': {}, 'modules': {}, 'repositories': {'a': {'name': 'A'}, 'b': {'name': 'B'}}}
        def fake(name, repo_config, *a, hits=None, **k):
            if repo_config['name'] == 'A':
                hits.append(('A', Token('x', 'A', [TokenLocation('f', 1)], confidence=0.9)))
                return 'out'
            raise ValueError('boom')
        out, err = io.StringIO(), io.StringIO()
        with patch('sys.argv', ['alas.py', '--fail-above', '0.8', '--quiet']), patch('alas_but_one.cli.resolve_config_path', return_value='config.json'), patch('alas_but_one.cli.load_config', return_value=config), \
                patch('alas_but_one.cli.MLPredictor', return_value=Mock(available=False, problem=None)), patch('alas_but_one.cli.run_repo', side_effect=fake), \
                redirect_stdout(out), redirect_stderr(err), self.assertLogs('alas_but_one.cli', 'DEBUG'):
            with self.assertRaises(SystemExit) as raised:
                alas.main()
        self.assertEqual(raised.exception.code, 1)
        self.assertEqual(out.getvalue(), 'A\tx\t0.90\tf:1\n')
        self.assertIn('B: FAILED (boom)', err.getvalue())


class LogFileTests(unittest.TestCase):
    def setUp(self):
        self._handlers = list(logging.getLogger().handlers)

    def _main(self, directory, *flags, settings=None):
        from contextlib import chdir
        import alas_but_one.cli as alas
        base = alas.load_config()
        config = {'settings': {**base['settings'], 'ai': {'enabled': False}, 'output_dir': directory, **(settings or {})}, 'modules': base.get('modules', {}), 'repositories': {
            'r': {'name': 'test', 'path': directory, 'text_format': 'plain'}}, 'config_dir': directory}
        buffer = io.StringIO()
        with chdir(directory), redirect_stdout(buffer), patch('sys.argv', ['alas.py', *flags]), \
                patch('alas_but_one.cli.resolve_config_path', return_value='config.json'), patch('alas_but_one.cli.load_config', return_value=config), patch('alas_but_one.cli.MLPredictor', return_value=Mock(available=False, problem=None)), \
                patch('alas_but_one.matchers.ignore_list_matcher.load_words', return_value=set()):
            alas.main()
        self.assertEqual(logging.getLogger().handlers, self._handlers)
        return buffer.getvalue()

    def test_log_file_has_stage_timings_and_stdout_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'sample.txt').write_text('retreive ordinary')
            log = os.path.join(directory, 'run.log')
            plain = self._main(directory)
            self.assertFalse(os.path.exists(log))
            logged = self._main(directory, '--log', log)
            self.assertEqual(plain, logged)
            text = Path(log).read_text()
            for stage in ['collector', 'reader', 'tokenizer', 'max_occurrence_matcher', 'spell_checker', 'ignore_list_matcher']:
                self.assertRegex(text, rf'stage {stage}: count=\d+ elapsed=\d+\.\d{{3}}s')

    def test_settings_log_file_is_config_relative_and_cli_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'sample.txt').write_text('retreive')
            from alas_but_one.config import load_config
            cfg = os.path.join(directory, 'c.json')
            Path(cfg).write_text(json.dumps({'settings': {'log_file': 'rel.log'}}))
            self.assertEqual(load_config(cfg)['settings']['log_file'], os.path.join(directory, 'rel.log'))
            cli = os.path.join(directory, 'cli.log')
            self._main(directory, '--log', cli, settings={'log_file': os.path.join(directory, 's.log')})
            self.assertTrue(os.path.exists(cli))
            self.assertFalse(os.path.exists(os.path.join(directory, 's.log')))

    def test_ai_batch_raw_response_and_failure_are_logged(self):
        import alas_but_one.cli as alas
        with tempfile.TemporaryDirectory() as directory:
            log = os.path.join(directory, 'ai.log')
            handler = alas.setup_logging(log)
            try:
                t = Token('teh', 'test', [TokenLocation('a.md', 1)], confidence=0.5)
                reviewer = AIReviewer({'ai': {'enabled': True}})
                client = Mock()
                client.messages.create.return_value.content = [Mock(type='text', text='RAW-NOT-JSON-123')]
                with patch.object(reviewer, '_get_client', return_value=client), redirect_stdout(io.StringIO()):
                    reviewer.run({'teh': t}, {'a.md': 'teh'})
            finally:
                alas.teardown_logging(handler)
            text = Path(log).read_text()
            self.assertIn('RAW-NOT-JSON-123', text)
            self.assertIn('AI batch 1 failed', text)

    def test_ai_raw_response_line_carries_batch_index(self):
        import alas_but_one.cli as alas
        with tempfile.TemporaryDirectory() as directory:
            log = os.path.join(directory, 'ai.log')
            handler = alas.setup_logging(log)
            try:
                tokens = {w: Token(w, 'test', [TokenLocation('a.md', 1)], confidence=0.5) for w in ('teh', 'wrld')}
                reviewer = AIReviewer({'ai': {'enabled': True, 'batch_size': 1}})
                client = Mock()
                client.messages.create.return_value.content = [Mock(type='text', text='RAW-NOT-JSON')]
                with patch.object(reviewer, '_get_client', return_value=client), redirect_stdout(io.StringIO()):
                    reviewer.run(tokens, {'a.md': 'teh wrld'})
            finally:
                alas.teardown_logging(handler)
            text = Path(log).read_text()
            for n in (1, 2):
                self.assertRegex(text, rf'AI batch {n} \(1 words\) raw response: RAW-NOT-JSON')
                self.assertIn(f'AI batch {n} failed', text)

    def test_log_lines_carry_thread_name_and_repo_start(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'sample.txt').write_text('retreive')
            log = os.path.join(directory, 'run.log')
            self._main(directory, '--log', log)
            self.assertRegex(Path(log).read_text(), r'DEBUG \[MainThread\] alas_but_one.cli: test : start')


class AISendContextTests(unittest.TestCase):
    def _run(self, send_context=None):
        t = Token('teh', 'test', [TokenLocation('a.md', 2)], confidence=0.5)
        content = {'a.md': 'first line\nSECRET teh sentence\nthird line'}
        ai_cfg = {'enabled': True}
        if send_context is not None:
            ai_cfg['send_context'] = send_context
        reviewer = AIReviewer({'ai': ai_cfg})
        client = Mock()
        client.messages.create.return_value.content = [Mock(type='text', text='{"results": []}')]
        with patch.object(reviewer, '_get_client', return_value=client):
            reviewer.run({'teh': t}, content)
        return client.messages.create.call_args.kwargs['messages'][0]['content']

    def test_context_sent_by_default(self):
        self.assertIn('SECRET teh sentence', self._run())

    def test_bare_words_when_send_context_false(self):
        msg = self._run(False)
        self.assertIn('teh', msg)
        for leaked in ('SECRET', 'first line', 'a.md', '"context"'):
            self.assertNotIn(leaked, msg)


class AIStructuredResponseTests(unittest.TestCase):
    @staticmethod
    def _item(word, confidence=0.9, suggestion='the'):
        return {'word': word, 'is_typo': True, 'confidence': confidence, 'suggestion': suggestion, 'comment': 'c'}

    def _run(self, results=None, stop_reason='end_turn'):
        tokens = {w: Token(w, 'test', [TokenLocation('a.md', 1)], confidence=0.5) for w in ('teh', 'wrld')}
        reviewer = AIReviewer({'ai': {'enabled': True}})
        client = Mock()
        response = client.messages.create.return_value
        response.stop_reason = stop_reason
        response.content = [Mock(type='text', text=json.dumps({'results': results or []}))]
        with patch.object(reviewer, '_get_client', return_value=client), redirect_stdout(io.StringIO()) as out:
            reviewer.run(tokens, {'a.md': 'teh wrld'})
        return tokens, client, out.getvalue()

    def _assert_untouched(self, tokens, out):
        for t in tokens.values():
            self.assertFalse(t.ai_reviewed)
            self.assertEqual(t.confidence, 0.5)
            self.assertIsNone(t.ai_comment)
        self.assertIn('batch 1 failed', out)

    def test_request_uses_json_schema_output(self):
        _, client, _ = self._run()
        fmt = client.messages.create.call_args.kwargs['output_config']['format']
        self.assertEqual(fmt['type'], 'json_schema')
        self.assertFalse(fmt['schema']['additionalProperties'])
        self.assertEqual(fmt['schema']['required'], ['results'])

    def test_matching_response_is_applied(self):
        tokens, _, _ = self._run([self._item('wrld', 0.8, None), self._item('teh', 0.95)])
        self.assertTrue(all(t.ai_reviewed for t in tokens.values()))
        self.assertEqual(tokens['teh'].confidence, 0.95)
        self.assertEqual(tokens['teh'].suggestion, 'the')
        self.assertEqual(tokens['wrld'].confidence, 0.8)
        self.assertIsNone(tokens['wrld'].suggestion)

    def test_missing_word_fails_whole_batch(self):
        tokens, _, out = self._run([self._item('teh')])
        self._assert_untouched(tokens, out)

    def test_extra_word_fails_whole_batch(self):
        tokens, _, out = self._run([self._item('teh'), self._item('wrld'), self._item('other')])
        self._assert_untouched(tokens, out)

    def test_duplicate_word_fails_whole_batch(self):
        tokens, _, out = self._run([self._item('teh'), self._item('teh'), self._item('wrld')])
        self._assert_untouched(tokens, out)

    def test_invalid_value_leaves_earlier_tokens_untouched(self):
        tokens, _, out = self._run([self._item('teh'), self._item('wrld', confidence='high')])
        self._assert_untouched(tokens, out)

    def test_refusal_and_truncation_fail_batch(self):
        for reason in ('refusal', 'max_tokens'):
            tokens, _, out = self._run([self._item('teh'), self._item('wrld')], stop_reason=reason)
            self._assert_untouched(tokens, out)


class FileIgnoreListTests(unittest.TestCase):
    def test_file_backend_round_trip(self):
        from alas_but_one.ignore_list_store import load_words, apply_decisions
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'sub', 'ignore.json')
            settings = {'ignore_list': {'file': path}}
            self.assertEqual(load_words('a', settings), set())
            apply_decisions('a', {'zeta': True, 'alpha': True}, settings)
            apply_decisions('a', {'alpha': True}, settings)
            apply_decisions('b', {'other': True}, settings)
            self.assertEqual(load_words('a', settings), {'alpha', 'zeta'})
            self.assertEqual(load_words('b', settings), {'other'})
            with open(path) as f:
                self.assertEqual(json.load(f)['a'], ['alpha', 'zeta'])
            apply_decisions('a', {'alpha': False}, settings)
            self.assertEqual(load_words('a', settings), {'zeta'})
            self.assertEqual(load_words('b', settings), {'other'})
            self.assertEqual(os.listdir(os.path.dirname(path)), ['ignore.json'])

    def test_malformed_ignore_file_names_the_file(self):
        from alas_but_one.ignore_list_store import IgnoreListError, load_words, apply_decisions
        for label, content in [('list top level', '["a"]'), ('string repo value', '{"a": "word"}')]:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                path = os.path.join(directory, 'ignore.json')
                Path(path).write_text(content)
                settings = {'ignore_list': {'file': path}}
                for operation in [lambda: load_words('a', settings), lambda: apply_decisions('a', {'w': True}, settings)]:
                    with self.assertRaises(IgnoreListError) as caught:
                        operation()
                    self.assertIn(path, str(caught.exception))

    def test_save_ignore_list_missing_pymongo_exits_without_traceback(self):
        import alas_but_one.save_ignore_list as save_ignore_list
        from contextlib import chdir
        with tempfile.TemporaryDirectory() as directory, chdir(directory):
            config = os.path.join(directory, 'config.json')
            Path(config).write_text(json.dumps({'settings': {'ignore_list': {'database': 'd', 'collection': 'c'}}, 'repos': []}))
            reviewed = os.path.join(directory, 'out.jsonl')
            Path(reviewed).write_text(json.dumps({'repo': 'r', 'word': 'w', 'ignore': True}) + '\n')
            with patch('sys.argv', ['alas_but_one.save_ignore_list.py', reviewed]), patch.dict(os.environ, {'ABO_CONFIG': config}, clear=True), \
                    patch('alas_but_one.ignore_list_store.MongoClient', None), self.assertRaises(SystemExit) as caught:
                save_ignore_list.main()
            self.assertIn('pymongo is required', str(caught.exception.code))

    def test_relative_file_resolves_against_config_dir(self):
        from alas_but_one.config import load_config
        from contextlib import chdir
        with tempfile.TemporaryDirectory() as directory:
            config_path = os.path.join(directory, 'config.json')
            with open(config_path, 'w') as f:
                json.dump({'settings': {'ignore_list': {'file': 'ignore.json'}}}, f)
            with chdir(tempfile.gettempdir()):
                config = load_config(config_path)
            resolved = config['settings']['ignore_list']['file']
            self.assertEqual(os.path.realpath(resolved), os.path.realpath(os.path.join(directory, 'ignore.json')))

    def test_possessive_folds_into_base_word(self):
        tokens = TokenizerTask({}, {'name': 'test'}).run({'a.txt': "the API's role and API docs; it's\nmispeled's"})
        self.assertEqual(tokens['api'].uppercase_occurrences, 2)
        self.assertEqual(len(tokens['api'].locations), 2)
        self.assertNotIn("api's", tokens)
        self.assertIn('mispeled', tokens)

    def test_railway_ignore_list_excludes_known_typos(self):
        data = Path(__file__).parent / 'data' / 'railway'
        ignored = set(json.loads((data / 'ignore.json').read_text())['Railway'])
        typos = {t['word'] for t in json.loads((data / 'typos.json').read_text())}
        self.assertFalse(ignored & typos)


class MongoCredentialLeakTests(unittest.TestCase):
    URI = 'mongodb://SECRETUSER:SECRETPW@db.example.com:27017/admin'

    def _failures(self):
        from pymongo.errors import ConfigurationError, ServerSelectionTimeoutError
        uri = self.URI
        timeout = ServerSelectionTimeoutError(f'db.example.com:27017: timed out ({uri}), user SECRETUSER password SECRETPW')
        yield 'timeout on operation', {'side_effect': None}, timeout
        yield 'constructor failure', {'side_effect': ConfigurationError(f'bad uri {uri}')}, None

    def _client(self, params, error):
        if params['side_effect'] is not None:
            return Mock(side_effect=params['side_effect'])
        client = Mock()
        client.__getitem__ = Mock(side_effect=error)
        return Mock(return_value=client)

    def _assert_clean(self, *texts):
        for text in texts:
            self.assertNotIn('SECRETUSER', text)
            self.assertNotIn('SECRETPW', text)
            self.assertNotIn(self.URI, text)

    def _config(self, directory, uri_in_config):
        Path(directory, 'docs').mkdir()
        Path(directory, 'docs', 'sample.txt').write_text('retreive')
        settings = {'maxOccurrences': 1, 'ignore_list': {'database': 'db', 'collection': 'words'}}
        if uri_in_config:
            settings['MONGODB_URI'] = self.URI
        path = os.path.join(directory, 'config.json')
        Path(path).write_text(json.dumps({'settings': settings, 'repositories': {
            'r': {'name': 'R', 'path': os.path.join(directory, 'docs')}}}))
        return path

    def test_alas_run_failures_never_print_uri_or_credentials(self):
        import alas_but_one.cli as alas
        from contextlib import chdir
        for uri_in_config in [True, False]:
            for label, params, error in self._failures():
                with self.subTest(label=label, uri_in_config=uri_in_config), tempfile.TemporaryDirectory() as directory, chdir(directory):
                    config = self._config(directory, uri_in_config)
                    log = os.path.join(directory, 'run.log')
                    env = {} if uri_in_config else {'ABO_MONGO_URI': self.URI}
                    out, err = io.StringIO(), io.StringIO()
                    with patch('sys.argv', ['alas.py', '--config', config, '--log', log]), patch.dict(os.environ, env, clear=True), \
                            patch('alas_but_one.cli.MLPredictor', return_value=Mock(available=False, problem=None)), \
                            patch('alas_but_one.ignore_list_store.MongoClient', self._client(params, error)), \
                            redirect_stdout(out), redirect_stderr(err), self.assertRaises(SystemExit) as caught:
                        alas.main()
                    self.assertNotEqual(caught.exception.code, 0)
                    self.assertIn('MongoDB error', str(caught.exception.code))
                    self._assert_clean(out.getvalue(), err.getvalue(), str(caught.exception.code), Path(log).read_text())

    def test_save_ignore_list_failures_never_print_uri_or_credentials(self):
        import alas_but_one.save_ignore_list as save_ignore_list
        from contextlib import chdir
        for label, params, error in self._failures():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory, chdir(directory):
                config = self._config(directory, False)
                reviewed = os.path.join(directory, 'out.jsonl')
                Path(reviewed).write_text(json.dumps({'repo': 'r', 'word': 'retreive', 'ignore': True}) + '\n')
                out, err = io.StringIO(), io.StringIO()
                with patch('sys.argv', ['alas_but_one.save_ignore_list.py', reviewed]), patch.dict(os.environ, {'ABO_CONFIG': config, 'ABO_MONGO_URI': self.URI}, clear=True), \
                        patch('alas_but_one.ignore_list_store.MongoClient', self._client(params, error)), \
                        redirect_stdout(out), redirect_stderr(err), self.assertRaises(SystemExit) as caught:
                    save_ignore_list.main()
                self.assertIn('MongoDB error', str(caught.exception.code))
                self._assert_clean(out.getvalue(), err.getvalue(), str(caught.exception.code))

    def test_invalid_config_errors_never_print_uri_or_credentials(self):
        from alas_but_one.ignore_list_store import resolve_ignore_list_settings
        settings = {'MONGODB_URI': self.URI, 'ignore_list': {'database': '', 'collection': 'words'}}
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError) as caught:
            resolve_ignore_list_settings(settings)
        self._assert_clean(str(caught.exception))

    def test_committed_example_config_has_no_credentials(self):
        text = Path(__file__).resolve().parent.parent.joinpath('config.json').read_text()
        self.assertNotRegex(text, r'://[^/\s"]*@')

    def test_redaction_covers_encoded_credentials_and_other_uris(self):
        from pymongo.errors import OperationFailure
        from alas_but_one.ignore_list_store import IgnoreListError, load_words
        uri = 'mongodb+srv://us%40er:p%40ss%3Aw%2Frd@cluster.example.net/?retryWrites=true'
        message = 'auth us@er p@ss:w/rd failed; seed mongodb://OTHERUSER:OTHERPW@replica.example.net:27017'
        client = Mock()
        client.__getitem__ = Mock(side_effect=OperationFailure(message))
        settings = {'ignore_list': {'database': 'db', 'collection': 'words'}}
        with patch.dict(os.environ, {'ABO_MONGO_URI': uri}, clear=True), \
                patch('alas_but_one.ignore_list_store.MongoClient', Mock(return_value=client)), \
                self.assertRaises(IgnoreListError) as caught:
            load_words('repo', settings)
        text = str(caught.exception)
        for secret in ['us@er', 'p@ss:w/rd', 'OTHERUSER', 'OTHERPW']:
            self.assertNotIn(secret, text)
        self.assertIn('replica.example.net', text)
        self.assertIsNone(caught.exception.__cause__)
        self.assertTrue(caught.exception.__suppress_context__)


if __name__ == '__main__':
    unittest.main()
