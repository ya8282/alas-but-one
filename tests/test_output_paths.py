import json
import os
import tempfile
import unittest
from contextlib import chdir, redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from alas_but_one import cli
from alas_but_one.config import load_config

LABELS = Path(__file__).parent / 'data' / 'labels.jsonl'


class OutputPathTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = os.path.realpath(tmp.name)
        self.cfgdir = os.path.join(self.root, 'cfg')
        self.cwd = os.path.join(self.root, 'cwd')
        os.makedirs(os.path.join(self.cfgdir, 'docs'))
        os.makedirs(self.cwd)
        Path(self.cfgdir, 'docs', 'a.txt').write_text('retreive ordinary')
        self.config_file = os.path.join(self.cfgdir, 'config.json')
        self.write_config({})

    def write_config(self, settings):
        config = {
            'settings': {'maxOccurrences': 5, 'ai': {'enabled': False}, 'ignore_list': {'file': 'ignore.json'}, **settings},
            'repositories': {'my-docs': {'name': 'My Docs/Odd Name', 'path': 'docs', 'text_format': 'plain'}},
        }
        Path(self.config_file).write_text(json.dumps(config))

    def main(self, *flags):
        self.err = StringIO()
        with chdir(self.cwd), redirect_stdout(StringIO()), redirect_stderr(self.err), \
                patch('sys.argv', ['alas', '--config', self.config_file, *flags]):
            try:
                cli.main()
            except SystemExit as e:
                return e.code
        return 0

    def test_load_config_resolves_model_path_and_output_dir_against_config(self):
        self.write_config({'training': {'model_path': 'm/x.json'}, 'output_dir': 'out'})
        settings = load_config(self.config_file)['settings']
        self.assertEqual(settings['training']['model_path'], os.path.join(self.cfgdir, 'm/x.json'))
        self.assertEqual(settings['output_dir'], os.path.join(self.cfgdir, 'out'))

    def test_defaults_are_config_relative_and_absolute_paths_win(self):
        settings = load_config(self.config_file)['settings']
        self.assertEqual(settings['training']['model_path'], os.path.join(self.cfgdir, 'models/classifier.json'))
        self.assertEqual(settings['output_dir'], self.cfgdir)
        absolute = os.path.join(self.root, 'abs')
        self.write_config({'output_dir': absolute})
        self.assertEqual(load_config(self.config_file)['settings']['output_dir'], absolute)

    def test_scan_writes_to_config_dir_named_by_key_not_display_name(self):
        self.assertEqual(self.main(), 0, self.err.getvalue())
        self.assertTrue(os.path.isfile(os.path.join(self.cfgdir, 'my-docs.jsonl')))
        self.assertEqual(os.listdir(self.cwd), [])

    def test_csv_named_by_key(self):
        self.main('--format', 'csv')
        self.assertTrue(os.path.isfile(os.path.join(self.cfgdir, 'my-docs.csv')))

    def test_settings_output_dir_is_config_relative(self):
        self.write_config({'output_dir': 'out'})
        self.main()
        self.assertTrue(os.path.isfile(os.path.join(self.cfgdir, 'out', 'my-docs.jsonl')))

    def test_cli_output_dir_overrides_and_is_cwd_relative(self):
        self.write_config({'output_dir': 'out'})
        self.main('--output-dir', 'here')
        self.assertTrue(os.path.isfile(os.path.join(self.cwd, 'here', 'my-docs.jsonl')))
        self.assertFalse(os.path.exists(os.path.join(self.cfgdir, 'out')))

    def test_train_writes_model_config_relative_and_scan_loads_it(self):
        self.assertEqual(self.main('--train', str(LABELS)), 0)
        self.assertTrue(os.path.isfile(os.path.join(self.cfgdir, 'models', 'classifier.json')))
        self.assertEqual(os.listdir(self.cwd), [])
