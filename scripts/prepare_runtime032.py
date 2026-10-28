"""Prepare only the pinned data-disk models and a scoped CPU-dependency venv.

Never installs into the native interpreter or imports a GPU library. Existing
files or environments that differ are preserved and reported as errors.
"""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import os
import socket
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT))
import download_auk_models_006 as transfer
import entry as original

PREFIX = Path('/root/private_data/guoxin-auk-port006/venv002')
PYTHON = PREFIX / 'bin/python'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def no_links(path):
    if any(p.is_symlink() for p in (Path(path), *Path(path).parents)):
        raise ValueError('Symlinked destination is preserved; explicit inspection required')


def fixed_rows():
    rows = transfer.validate(load(ROOT / 'model-acquisition-lock.json'))
    models = load(ROOT / 'candidate/model-lock.json')['models']
    expected = {('auk006' if m['id'] == 'tencent/AuK' else 'qwen25-omni3b006', r['path']): r
                for m in models for r in m['files'] if r.get('required_for_conservative_snapshot')}
    assert len(rows) == len(expected) == 20
    assert {(r['namespace'], r['path']) for r in rows} == set(expected)
    for row in rows:
        target = expected[(row['namespace'], row['path'])]
        assert row['size'] == target['bytes']
        if target.get('sha256'):
            assert row['sha256'] == target['sha256']
    return rows


def fingerprint(metadata):
    return {k: {'version': v['version'], 'metadata_sha256': v['metadata_sha256']} if v else None
            for k, v in metadata['packages'].items()}


def environment_matches(executable, expected):
    state = original.distributions(executable, expected)
    assert state['prefix'] != state['base_prefix'], 'A scoped venv is required'
    actual = fingerprint(state)
    changed = {name for name in expected if actual.get(name) != expected[name]}
    if changed == {'antlr4-python3-runtime'}:
        # The pinned wheel carries PEP metadata2.1; the measured sdist install
        # carries metadata2.4. Do not alter either artifact or accept arbitrary
        # metadata: require the exact observed pair and every runtime code byte.
        lock = load(ROOT / 'publisher037/antlr4-runtime-package-fingerprint.json')
        name = lock['package']
        assert expected[name] == {'version': lock['version'], 'metadata_sha256': lock['original_metadata_sha256']}
        assert actual[name] == {'version': lock['version'], 'metadata_sha256': lock['compatible_metadata_sha256']}
        package = Path(state['packages'][name]['root']) / 'antlr4'
        files = {}
        for folder, dirs, names in os.walk(package, onerror=lambda error: (_ for _ in ()).throw(error)):
            dirs[:] = [part for part in dirs if part != '__pycache__']
            for filename in names:
                path = Path(folder) / filename
                assert path.is_file() and not path.is_symlink()
                files[path.relative_to(package).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        assert files == lock['package_files'], 'ANTLR runtime source differs; preserve the environment'
        state['declared_metadata_only_compatibility037'] = {
            'package': name, 'version': lock['version'], 'code_files_identical': len(files),
            'actual_metadata_sha256': actual[name]['metadata_sha256'],
            'original_metadata_sha256': expected[name]['metadata_sha256']}
        actual[name] = expected[name]
    assert actual == expected, 'Existing venv differs from the measured package profile; preserve it'
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['plan', 'check', 'ensure'])
    ap.add_argument('--max-download-seconds', type=int, default=10800)
    args = ap.parse_args()
    assert 60 <= args.max_download_seconds <= 10800
    rows = fixed_rows()
    expected = load(ROOT / 'publisher032/expected-native-and-venv-packages.json')['packages']
    wheels = original.wheel_lock()
    assert all(expected[r['name']]['version'] == r['version'] for r in wheels['files'])
    result = {'scope': 'Pinned model data reuse/download and dedicated venv only', 'files': 20,
              'models': rows, 'model_bytes': sum(r['size'] for r in rows), 'venv': PREFIX.as_posix(),
              'python': PYTHON.as_posix(), 'expected_packages': expected,
              'GPU_imported': False, 'GPU_started': False, 'native_package_install': False,
              'model_downloads_executed': False, 'new_venv_created': False}
    if args.action == 'plan':
        print(json.dumps(dict(result, plan_only=True), indent=2))
        return 0
    assert sys.platform == 'linux', 'Actual preparation requires the SCNet Linux environment'
    for port in (30000, 30001, 30002, 30109, 30119, 30129, 30139):
        with socket.socket() as sock:
            sock.settimeout(.5)
            assert sock.connect_ex(('127.0.0.1', port)) != 0, 'Existing model service is preserved'
    no_links(transfer.BASE)
    no_links(PREFIX)
    missing = []
    from download_candidate_models_20261023 import verified_file
    for row in rows:
        target = transfer.BASE / row['namespace'] / row['path']
        no_links(target)
        if target.exists():
            assert verified_file(target, row), 'Existing checkpoint differs; refusing replacement'
        else:
            missing.append(row)
    if missing:
        assert args.action == 'ensure', 'Pinned model files are missing; run ensure or review plan'
        transfer.apply(rows, args.max_download_seconds)
        result['model_downloads_executed'] = True
    result.update(all20_model_files_verified=True, existing_model_files_reused=20 - len(missing))
    if PYTHON.exists():
        result['venv_metadata'] = environment_matches(PYTHON, expected)
    else:
        assert args.action == 'ensure', 'Fixed venv is missing'
        assert not PREFIX.exists(), 'Incomplete existing venv is preserved; inspect before continuing'
        before = original.distributions(sys.executable, original.PROTECTED)
        protected = {k: expected[k] for k in original.PROTECTED}
        assert fingerprint(before) == protected, 'Native image differs from the fixed measured environment'
        PREFIX.parent.mkdir(parents=True, exist_ok=True)
        claim = PREFIX.parent / '.venv002-preparation032.lock'
        with claim.open('x', encoding='utf-8') as stream:
            json.dump({'pid': os.getpid(), 'UTC': dt.datetime.now(dt.timezone.utc).isoformat()}, stream)
        subprocess.run([sys.executable, '-B', '-m', 'venv', '--without-pip', '--system-site-packages', str(PREFIX)],
                       check=True, timeout=120)
        env = os.environ.copy()
        for key in ('PIP_TARGET', 'PIP_PREFIX', 'PIP_USER'):
            env.pop(key, None)
        env['PIP_CONFIG_FILE'] = os.devnull
        wanted = [r['name'] + '==' + r['version'] for r in wheels['files']]
        subprocess.run([str(PYTHON), '-B', '-m', 'pip', 'install', '--require-virtualenv',
            '--disable-pip-version-check', '--no-index', '--no-deps', '--find-links', str(ROOT / 'cpu-wheels'), *wanted],
            env=env, check=True, timeout=300)
        after = original.distributions(sys.executable, original.PROTECTED)
        assert before == after, 'Protected native metadata changed; inspect the isolated preparation'
        result['venv_metadata'] = environment_matches(PYTHON, expected)
        result.update(new_venv_created=True, protected_native_metadata_unchanged=True)
        with (PREFIX / 'auk-fixed-profile032-ready.json').open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2)
        claim.unlink()
    print(json.dumps(dict(result, preparation_complete=True), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
