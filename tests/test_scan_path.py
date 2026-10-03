import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from unittest.mock import patch

from alas_but_one import cli, ignore_list_store


class ScanPathTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = os.path.realpath(tmp.name)
        self.cwd = os.path.join(self.root, 'cwd')
        self.docs = os.path.join(self.root, 'my docs')
        os.makedirs(self.cwd)
        os.makedirs(self.docs)
        with open(os.path.join(self.docs, 'a.rst'), 'w') as f:
            f.write('A sentence with a zzyzxtypo word.\n')
        old = os.getcwd()
        os.chdir(self.cwd)
        self.addCleanup(os.chdir, old)
        env = patch.dict(os.environ, {'XDG_CONFIG_HOME': os.path.join(self.root, 'xdg'), 'HOME': self.root})
        env.start()
        self.addCleanup(env.stop)
        for name in ('ABO_CONFIG', 'ABO_MONGO_URI'):
            os.environ.pop(name, None)

    def main(self, *argv):
        out, err = StringIO(), StringIO()
        with patch('sys.argv', ['alas', *argv]), redirect_stdout(out), redirect_stderr(err):
            try:
                cli.main()
                code = 0
            except SystemExit as exit_:
                code = exit_.code
        return code, out.getvalue(), err.getvalue()

    def test_no_config_no_mongo_scans_and_never_touches_pymongo(self):
        boom = patch.object(ignore_list_store, 'MongoClient', side_effect=AssertionError('MongoClient used'))
        with boom:
            code, out, err = self.main(self.docs)
        self.assertEqual(code, 0, err)
        output = os.path.join(self.cwd, 'my-docs.jsonl')
        self.assertTrue(os.path.isfile(output))
        with open(output) as f:
            self.assertIn('zzyzxtypo', f.read())

    def test_found_config_settings_used_but_only_given_dir(self):
        other = os.path.join(self.root, 'other')
        os.makedirs(other)
        out_dir = os.path.join(self.root, 'out')
        config = {
            'settings': {'output_dir': out_dir},
            'repositories': {'other': {'name': 'Other', 'path': other}},
        }
        with open(os.path.join(self.cwd, 'config.json'), 'w') as f:
            json.dump(config, f)
        code, _, err = self.main(self.docs)
        self.assertEqual(code, 0, err)
        self.assertEqual(os.listdir(out_dir), ['my-docs.jsonl'])

    def test_markdown_file_is_masked_by_extension(self):
        with open(os.path.join(self.docs, 'b.md'), 'w') as f:
            f.write('Prose with `zzinline` code.\n')
        code, _, err = self.main(self.docs)
        self.assertEqual(code, 0, err)
        with open(os.path.join(self.cwd, 'my-docs.jsonl')) as f:
            words = {json.loads(line)['word'] for line in f}
        self.assertNotIn('zzinline', words)
        self.assertIn('zzyzxtypo', words)

    def test_missing_directory_names_path(self):
        code, _, _ = self.main(os.path.join(self.root, 'nope'))
        self.assertIn(os.path.join(self.root, 'nope'), str(code))
        self.assertIn('does not exist', str(code))

    def test_no_path_no_config_still_errors(self):
        code, _, _ = self.main()
        self.assertIn('alas --init', str(code))

    def test_config_without_repositories_errors_alone_but_works_with_path(self):
        cfg = os.path.join(self.root, 'norepos.json')
        with open(cfg, 'w') as f:
            json.dump({'settings': {'output_dir': self.cwd}}, f)
        code, _, _ = self.main('--config', cfg)
        self.assertIn('repositories', str(code))
        self.assertIn(cfg, str(code))
        code, _, err = self.main(self.docs, '--config', cfg)
        self.assertEqual(code, 0, err)

    def test_train_does_not_require_repositories(self):
        cfg = os.path.join(self.root, 'norepos.json')
        with open(cfg, 'w') as f:
            json.dump({'settings': {'output_dir': self.cwd}}, f)
        with patch.object(cli, 'cmd_train') as train:
            code, _, err = self.main('--train', 'labeled.jsonl', '--config', cfg)
        self.assertNotIn('repositories', str(code) + err)
        train.assert_called_once()

    def test_path_with_repo_rejected(self):
        code, _, err = self.main(self.docs, '--repo', 'x')
        self.assertEqual(code, 2)
        self.assertIn('cannot be combined', err)


if __name__ == '__main__':
    unittest.main()
