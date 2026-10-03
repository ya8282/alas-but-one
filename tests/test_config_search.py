import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from unittest.mock import patch

from alas_but_one import cli
from alas_but_one.config import ConfigError, config_search_paths, load_config, resolve_config_path


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('{}')
    return path


class SearchOrderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = os.path.realpath(self.tmp.name)
        self.cwd = os.path.join(self.root, 'cwd')
        self.xdg = os.path.join(self.root, 'xdg')
        os.makedirs(self.cwd)
        old = os.getcwd()
        os.chdir(self.cwd)
        self.addCleanup(os.chdir, old)
        env = patch.dict(os.environ, {'XDG_CONFIG_HOME': self.xdg})
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop('ABO_CONFIG', None)
        self.cwd_cfg = os.path.join(self.cwd, 'config.json')
        self.xdg_cfg = os.path.join(self.xdg, 'alas-but-one', 'config.json')

    def test_explicit_beats_everything(self):
        _touch(self.cwd_cfg)
        os.environ['ABO_CONFIG'] = 'env.json'
        self.assertEqual(resolve_config_path('flag.json'), 'flag.json')

    def test_env_beats_cwd_and_xdg(self):
        _touch(self.cwd_cfg)
        _touch(self.xdg_cfg)
        os.environ['ABO_CONFIG'] = 'env.json'
        self.assertEqual(resolve_config_path(), 'env.json')

    def test_cwd_beats_xdg(self):
        _touch(self.cwd_cfg)
        _touch(self.xdg_cfg)
        self.assertEqual(resolve_config_path(), self.cwd_cfg)

    def test_xdg_used_last(self):
        _touch(self.xdg_cfg)
        self.assertEqual(resolve_config_path(), self.xdg_cfg)

    def test_xdg_defaults_to_home_config(self):
        with patch.dict(os.environ, {'HOME': self.root}):
            del os.environ['XDG_CONFIG_HOME']
            self.assertEqual(config_search_paths()[1], os.path.join(self.root, '.config', 'alas-but-one', 'config.json'))

    def test_none_found_lists_locations_and_suggests_init(self):
        with self.assertRaises(ConfigError) as ctx:
            resolve_config_path()
        message = str(ctx.exception)
        self.assertIn(self.cwd_cfg, message)
        self.assertIn(self.xdg_cfg, message)
        self.assertIn('alas --init', message)
        self.assertNotIn('\n', message)


class InitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = os.path.realpath(self.tmp.name)

    def _main(self, *argv):
        out, err = StringIO(), StringIO()
        with patch('sys.argv', ['alas', *argv]), redirect_stdout(out), redirect_stderr(err):
            try:
                cli.main()
                code = 0
            except SystemExit as exit_:
                code = exit_.code
        return code, out.getvalue(), err.getvalue()

    def test_init_writes_loadable_credential_free_config(self):
        path = os.path.join(self.root, 'c.json')
        code, out, _ = self._main('--config', path, '--init')
        self.assertEqual(code, 0)
        self.assertIn(path, out)
        self.assertIn('ABO_MONGO_URI', out)
        config = load_config(path)
        self.assertNotIn('@', config['settings']['MONGODB_URI'])
        self.assertNotIn('@', open(path).read())

    def test_init_refuses_overwrite(self):
        path = _touch(os.path.join(self.root, 'c.json'))
        code, _, _ = self._main('--config', path, '--init')
        self.assertIn('already exists', str(code))
        self.assertEqual(open(path).read(), '{}')

    def test_init_then_run_reports_missing_directory_naming_repo(self):
        path = os.path.join(self.root, 'c.json')
        self._main('--config', path, '--init')
        with patch.dict(os.environ, {'HOME': self.root}):  # the example's ~/docs/my-project must not exist
            code, _, err = self._main('--config', path)
        self.assertEqual(code, 1)
        self.assertIn('Repository My Project Docs: directory', err)
        self.assertIn('does not exist', err)

    def test_init_defaults_to_cwd(self):
        old = os.getcwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, old)
        with patch.dict(os.environ, {'ABO_CONFIG': os.path.join(self.root, 'ignored.json')}):
            code, _, _ = self._main('--init')
        self.assertEqual(code, 0)
        self.assertTrue(os.path.isfile(os.path.join(self.root, 'config.json')))
        self.assertFalse(os.path.exists(os.path.join(self.root, 'ignored.json')))


class SettingsTypeTests(unittest.TestCase):
    def load(self, settings):
        import json
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = os.path.join(tmp.name, 'config.json')
        with open(path, 'w') as f:
            json.dump({'settings': settings}, f)
        return load_config(path)

    def test_ignore_list_file_expands_tilde(self):
        with patch.dict(os.environ, {'HOME': '/home/someone'}):
            config = self.load({'ignore_list': {'file': '~/ignore.json'}})
        self.assertEqual(config['settings']['ignore_list']['file'], '/home/someone/ignore.json')

    def test_bad_types_raise_config_error_naming_key(self):
        cases = [
            ({'training': []}, 'settings.training'),
            ({'training': {'model_path': 3}}, 'settings.training.model_path'),
            ({'output_dir': 3}, 'settings.output_dir'),
            ({'ignore_list': {'file': 3}}, 'settings.ignore_list.file'),
            ({'ignore_list': {'file': ''}}, 'settings.ignore_list.file'),
            ({'ignore_list': []}, 'settings.ignore_list'),
            ({'log_file': 3}, 'settings.log_file'),
        ]
        for settings, key in cases:
            with self.subTest(key=key):
                with self.assertRaises(ConfigError) as ctx:
                    self.load(settings)
                self.assertIn(key, str(ctx.exception))

    def test_non_dict_settings_and_top_level(self):
        for body, key in (({'settings': []}, 'settings'), ([], 'top level')):
            with self.subTest(key=key):
                import json
                tmp = tempfile.TemporaryDirectory()
                self.addCleanup(tmp.cleanup)
                path = os.path.join(tmp.name, 'config.json')
                with open(path, 'w') as f:
                    json.dump(body, f)
                with self.assertRaises(ConfigError) as ctx:
                    load_config(path)
                self.assertIn(key, str(ctx.exception))

    def test_empty_strings_fall_back(self):
        config = self.load({'training': {'model_path': ''}, 'output_dir': '', 'log_file': ''})
        self.assertTrue(config['settings']['training']['model_path'].endswith('models/classifier.json'))
        self.assertEqual(config['settings']['output_dir'], config['config_dir'])
        self.assertEqual(config['settings']['log_file'], '')


if __name__ == '__main__':
    unittest.main()
