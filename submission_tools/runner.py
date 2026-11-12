"""Run unchanged algorithms with common paths and check their original predictions."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SUBMISSIONS = ('01_hidden_ko_history', '02_turn31_v5', '03_turn31_v2')
INPUTS = ('train.jsonl', 'test.jsonl', 'type_chart_gen2.csv', 'sample_submission.csv')


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_data(path=None):
    """Find a complete competition mount; an explicit path never falls back silently."""
    explicit = path or os.environ.get('FDS_DATA')
    if explicit:
        candidates = [Path(explicit).expanduser().resolve()]
    else:
        candidates = [ROOT / 'data']
        mount = Path('/kaggle/input')
        if mount.exists():
            candidates.extend(p.parent for p in sorted(mount.rglob('train.jsonl')))
    for candidate in candidates:
        if all((candidate / name).is_file() for name in INPUTS):
            return candidate.resolve()
    raise FileNotFoundError('Set --data or FDS_DATA to the folder containing: ' + ', '.join(INPUTS))


def _read_csv(path, require_binary=True):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['battle_id', 'player_won']:
            raise ValueError(f'{path}: expected battle_id,player_won columns')
        result = {}
        for row in reader:
            battle_id = int(row['battle_id'])
            if battle_id in result:
                raise ValueError(f'{path}: duplicate battle_id {battle_id}')
            value = int(row['player_won']) if require_binary else row['player_won']
            if require_binary and value not in (0, 1):
                raise ValueError(f'{path}: predictions must be 0 or 1')
            result[battle_id] = value
    if not result:
        raise ValueError(f'{path}: empty CSV')
    return result


def validate_csv(path, sample):
    rows = _read_csv(path)
    expected = _read_csv(sample, require_binary=False)
    if rows.keys() != expected.keys():
        raise ValueError(f'{path}: battle IDs do not match sample_submission.csv')
    canonical = ''.join(f'{key},{rows[key]}\n' for key in sorted(rows)).encode()
    return {'rows': len(rows), 'p1_wins': sum(rows.values()),
            'file_sha256': file_hash(path),
            'predictions_sha256': hashlib.sha256(canonical).hexdigest()}


def preflight(approach, data):
    """Check paths, source integrity and recorded missing code without fitting models."""
    approach = Path(approach).resolve()
    data = resolve_data(data)
    source = json.loads((approach / 'source.json').read_text())
    errors = []
    for entry in source['files']:
        path = approach / 'code' / entry['path']
        if not path.is_file() or file_hash(path) != entry['sha256']:
            errors.append('Original code changed or missing: ' + entry['path'])
        else:
            compile(path.read_text(), str(path), 'exec')
    archive = approach / source['original']
    if not archive.is_file() or file_hash(archive) != source['original_sha256']:
        errors.append('Original source archive changed or missing')
    if source['missing']:
        errors.append('Incomplete original source: ' + '; '.join(source['missing']))
    reference = json.loads((approach / 'reference.json').read_text())
    comparison = reference.get('local_check', {})
    note = ''
    if comparison and not comparison['exact_reproduction']:
        note = (f'Previous fresh run differed in {comparison["different_predictions"]}/'
                f'{comparison["total_predictions"]} predictions; exact reproduction '
                'has not been established')
    return {'submission': approach.name, 'ready_to_execute': not errors, 'errors': errors,
            'data': str(data), 'reproduction_note': note}


def _runtime():
    versions = {'python': platform.python_version(), 'platform': platform.platform()}
    for name in ('numpy', 'pandas', 'scikit-learn', 'lightgbm', 'xgboost', 'catboost', 'scipy'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _commit():
    return subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()


class _Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, value):
        self.console.write(value)
        self.log.write(value)
        self.flush()

    def flush(self):
        self.console.flush()
        self.log.flush()


def _execute(approach, source, data, artifacts):
    if source['kind'] == 'python_project':
        env = dict(os.environ, FDS_DATA=str(data))
        command = [sys.executable, '-u', str(approach / 'code' / source['entrypoint']),
                   '--no-ablation', '--out', str(artifacts)]
        with subprocess.Popen(command, cwd=artifacts, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True) as process:
            for line in process.stdout:
                print(line, end='', flush=True)
            if process.wait():
                raise subprocess.CalledProcessError(process.returncode, command)
        return artifacts / 'submission.csv'
    namespace = {'__name__': '__main__'}
    for cell in source['files']:
        path = approach / 'code' / cell['path']
        print(f'Running source cell {cell["cell_index"]}', flush=True)
        exec(compile(path.read_text(), str(path), 'exec'), namespace)
        if cell['cell_index'] == source['config_cell']:
            # Only filesystem locations are adapted. Model code is executed verbatim.
            namespace.update(DATA_DIR=str(artifacts),
                             TYPE_CHART_PATH=str(data / 'type_chart_gen2.csv'),
                             TRAIN_JSONL_PATH=str(data / 'train.jsonl'),
                             TEST_JSONL_PATH=str(data / 'test.jsonl'),
                             EXCEL_UNPACKED_PATH=str(artifacts / 'unused.xlsx'))
    reference = json.loads((approach / 'reference.json').read_text())
    return artifacts / reference['filename']


def run(approach, data, out):
    approach, out = Path(approach).resolve(), Path(out).resolve()
    data = resolve_data(data)
    check = preflight(approach, data)
    if not check['ready_to_execute']:
        raise RuntimeError('\n'.join(check['errors']))
    source = json.loads((approach / 'source.json').read_text())
    reference = json.loads((approach / 'reference.json').read_text())
    out.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    with (out / 'run.log').open('w') as log, tempfile.TemporaryDirectory(prefix='.run-', dir=out) as tmp:
        artifacts = Path(tmp) / 'artifacts'
        artifacts.mkdir()
        try:
            os.chdir(artifacts)
            with redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(_Tee(sys.stderr, log)):
                generated = _execute(approach, source, data, artifacts)
                stats = validate_csv(generated, data / 'sample_submission.csv')
        finally:
            os.chdir(previous)
        shutil.copy2(generated, out / 'submission.csv')
        shutil.copytree(artifacts, out / 'artifacts', dirs_exist_ok=True)
    comparison = {'submission_ref': reference['submission_ref'],
                  'predictions_match': stats['predictions_sha256'] == reference['predictions_sha256'],
                  'expected_predictions_sha256': reference['predictions_sha256']}
    local_reference = approach / 'outputs/reference/submission.csv'
    if local_reference.exists():
        old, new = _read_csv(local_reference), _read_csv(out / 'submission.csv')
        comparison['different_predictions'] = sum(old[k] != new[k] for k in old)
    record = {'submission': approach.name, 'git_commit': _commit(),
              'source_manifest_sha256': file_hash(approach / 'source.json'),
              'inputs': {name: file_hash(data / name) for name in INPUTS},
              'runtime': _runtime(), 'output': stats, 'reference': comparison,
              'csv_path': str(out / 'submission.csv')}
    (out / 'run_manifest.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'submission': approach.name, **stats}, indent=2))
    return record


def bundle(records, out):
    """Export the three generated submissions after checking their run and file integrity."""
    by_name = {record['submission']: record for record in records}
    if len(records) != 3 or set(by_name) != set(SUBMISSIONS):
        raise ValueError('All three submissions must be generated in this notebook run before export')
    if len({record['git_commit'] for record in records}) != 1:
        raise ValueError('Submissions were generated from different commits')
    if len({json.dumps(record['inputs'], sort_keys=True) for record in records}) != 1:
        raise ValueError('Submissions were generated from different input files')
    for name, record in by_name.items():
        current = file_hash(record['csv_path'])
        if current != record['output']['file_sha256']:
            raise ValueError(f'{name}: CSV changed after validation')
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / 'all_submissions.json'
    manifest.write_text(json.dumps(records, indent=2) + '\n')
    path = out / 'final_submissions.zip'
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in SUBMISSIONS:
            archive.write(by_name[name]['csv_path'], name + '.csv')
        archive.write(manifest, manifest.name)
    return path


def main(approach=None):
    parser = argparse.ArgumentParser(description=__doc__)
    if approach is None:
        parser.add_argument('submission', choices=SUBMISSIONS)
    parser.add_argument('--data', help='Competition input folder (or use FDS_DATA)')
    parser.add_argument('--out', help='Output folder (default: this submission/outputs/run)')
    parser.add_argument('--check', action='store_true', help='Check inputs and original source; do not train')
    args = parser.parse_args()
    approach = Path(approach) if approach else ROOT / 'submissions' / args.submission
    data = resolve_data(args.data)
    if args.check:
        check = preflight(approach, data)
        print(json.dumps({key: value for key, value in check.items()
                          if key != 'reproduction_note'}, indent=2))
        if not check['ready_to_execute']:
            raise SystemExit(2)
    else:
        run(approach, data, args.out or approach / 'outputs/run')


if __name__ == '__main__':
    main()
