"""Checks for the shared entry points and hand-in integrity, without fitting models."""

import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile

from submission_tools.runner import ROOT, SUBMISSIONS, bundle, preflight, resolve_data, validate_csv


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / 'data'
        self.data.mkdir()
        for name in ('train.jsonl', 'test.jsonl', 'type_chart_gen2.csv'):
            (self.data / name).touch()
        self.sample = self.data / 'sample_submission.csv'
        self.write_csv(self.sample, [(7, 0), (11, 0)])

    @staticmethod
    def write_csv(path, rows):
        with path.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['battle_id', 'player_won'])
            writer.writerows(rows)

    def test_same_layout_and_common_launcher(self):
        expected = {'README.md', 'run.py', 'requirements.txt', 'source.json',
                    'reference.json', 'code', 'original', 'outputs'}
        launchers = []
        for name in SUBMISSIONS:
            directory = ROOT / 'submissions' / name
            entries = {p.name for p in directory.iterdir() if p.name != '__pycache__'}
            self.assertEqual(entries, expected)
            launchers.append((directory / 'run.py').read_bytes())
        self.assertEqual(len(set(launchers)), 1)

    def test_complete_sources_pass_preflight(self):
        for name in SUBMISSIONS:
            self.assertTrue(preflight(ROOT / 'submissions' / name, self.data)['ready_to_execute'])
        check = preflight(ROOT / 'submissions' / SUBMISSIONS[2], self.data)
        self.assertIn('72/5000', check['reproduction_note'])

    def test_restored_source_still_has_to_match_its_recorded_hash(self):
        altered = self.root / 'altered-v2'
        shutil.copytree(ROOT / 'submissions' / SUBMISSIONS[2], altered,
                        ignore=shutil.ignore_patterns('outputs', '__pycache__'))
        with (altered / 'code/cell_06.py').open('a') as stream:
            stream.write('\n# Changed after source verification\n')
        check = preflight(altered, self.data)
        self.assertFalse(check['ready_to_execute'])
        self.assertIn('cell_06.py', ' '.join(check['errors']))

    def test_explicit_input_does_not_fall_back_to_another_dataset(self):
        with self.assertRaises(FileNotFoundError):
            resolve_data(self.root / 'missing')

    def test_prediction_fingerprint_is_independent_of_row_order(self):
        first, second = self.root / 'first.csv', self.root / 'second.csv'
        self.write_csv(first, [(7, 1), (11, 0)])
        self.write_csv(second, [(11, 0), (7, 1)])
        a, b = validate_csv(first, self.sample), validate_csv(second, self.sample)
        self.assertEqual(a['predictions_sha256'], b['predictions_sha256'])
        self.assertNotEqual(a['file_sha256'], b['file_sha256'])

    def test_csv_rejects_missing_duplicate_or_non_binary_predictions(self):
        path = self.root / 'bad.csv'
        for rows in ([(7, 1)], [(7, 1), (7, 0)], [(7, 2), (11, 0)], [(7, 1), (12, 0)]):
            with self.subTest(rows=rows):
                self.write_csv(path, rows)
                with self.assertRaises(ValueError):
                    validate_csv(path, self.sample)

    def records(self):
        records = []
        for index, name in enumerate(SUBMISSIONS):
            path = self.root / f'{index}.csv'
            self.write_csv(path, [(7, 1), (11, 0)])
            records.append({'submission': name, 'git_commit': 'same-commit',
                            'inputs': {'train.jsonl': 'same-data'},
                            'output': validate_csv(path, self.sample),
                            'reference': {'predictions_match': True}, 'csv_path': str(path)})
        return records

    def test_export_contains_exactly_the_three_csvs_and_manifest(self):
        archive = bundle(self.records(), self.root / 'export')
        with zipfile.ZipFile(archive) as stream:
            self.assertEqual(set(stream.namelist()),
                             {name + '.csv' for name in SUBMISSIONS} | {'all_submissions.json'})

    def test_export_rejects_partial_mismatched_or_stale_runs(self):
        records = self.records()
        with self.assertRaises(ValueError):
            bundle(records[:2], self.root / 'export')
        for field, value in (('git_commit', 'other-commit'), ('inputs', {'train.jsonl': 'other-data'})):
            changed = json.loads(json.dumps(records))
            changed[0][field] = value
            with self.assertRaises(ValueError):
                bundle(changed, self.root / 'export')
        changed = json.loads(json.dumps(records))
        changed[0]['reference']['predictions_match'] = False
        with self.assertRaises(ValueError):
            bundle(changed, self.root / 'export')
        self.write_csv(Path(records[0]['csv_path']), [(7, 0), (11, 0)])
        with self.assertRaises(ValueError):
            bundle(records, self.root / 'export')


if __name__ == '__main__':
    unittest.main()
