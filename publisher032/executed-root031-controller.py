"""Bounded, single owned AuK fixed100 root replay on the existing A instance."""
from pathlib import Path
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import uuid
import zipfile


OUT = Path('/root/private_data/guoxin-session031/auk-root-replay001')
PYTHON = '/root/private_data/guoxin-auk-port006/venv002/bin/python'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def token(pid):
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]


def ports_open():
    values = {}
    for port in (30000, 30001, 30002, 30109, 30119, 30129, 30139):
        with socket.socket() as sock:
            sock.settimeout(.5)
            values[str(port)] = sock.connect_ex(('127.0.0.1', port)) == 0
    return values


def normalize(value, task, run_id):
    if isinstance(value, dict):
        return {k: normalize(v, task, run_id) for k, v in value.items() if k != 'run_id'}
    if isinstance(value, list):
        return [normalize(v, task, run_id) for v in value]
    return value.replace(task, '__TASK__').replace(run_id, '__RUN__') if isinstance(value, str) else value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--archive', type=Path, required=True)
    ap.add_argument('--lock', type=Path, required=True)
    ap.add_argument('--archive-sha', required=True)
    ap.add_argument('--lock-sha', required=True)
    args = ap.parse_args()
    assert socket.gethostname() == 'crdnotebook-2112186448839770114-ac3p4mowhz-97046'
    assert sha(args.archive) == args.archive_sha and sha(args.lock) == args.lock_sha
    lock = json.loads(args.lock.read_text())
    assert lock['archive_sha256'] == args.archive_sha and lock['root_selected_samples'] == 100
    assert not any(ports_open().values()), 'Existing service preserved; root replay not launched'
    assert not Path('/proc/64006').exists() and not Path('/proc/64007').exists()
    prior = json.loads(Path('/root/private_data/guoxin-session030/auk-performance-A001/progress.json').read_text())
    assert prior['performance_complete'] and prior['measured_requests'] == 96 and prior['owned_service_stopped']
    OUT.mkdir(parents=True, exist_ok=False)
    control = (OUT / 'exclusive.lock').open('x')
    fcntl.flock(control, fcntl.LOCK_EX | fcntl.LOCK_NB)
    project = OUT / 'project'
    project.mkdir()
    (OUT / 'executed-controller.py').write_bytes(Path(__file__).read_bytes())
    (OUT / 'frozen-project-lock.json').write_bytes(args.lock.read_bytes())
    run_id = uuid.uuid4().hex
    record = {'started_utc': now(), 'controller_pid': os.getpid(), 'controller_start_token': token(os.getpid()),
              'source_sha256': sha(__file__), 'state': 'extracting_exact_frozen_project_before_GPU',
              'git_commit': lock['commit'], 'source_archive_sha256': args.archive_sha,
              'exact_source_files': len(lock['files']), 'run_id': run_id, 'phases': {},
              'GPU_started': False, 'root100_generation_complete': False, 'root_replay_complete': False,
              'complete_formal_accuracy_preserved': 1200, 'new_performance_requests': 0,
              'technical_delivery_complete': False, 'submitted': False, 'Notebook_shutdown_requested': False}
    def save():
        tmp = OUT / 'progress.tmp'
        tmp.write_text(json.dumps(record, indent=2) + '\n')
        tmp.replace(OUT / 'progress.json')
    save()
    env = os.environ.copy()
    env.update(AUK_PYTHON_BIN=PYTHON, AUK_MAX_SECONDS='100', AUK_MAX_RUNTIME_SECONDS='10800', AUK_LOAD_TIMEOUT='1200')
    def step(name, argv, bound):
        with (OUT / (name + '.log')).open('xb') as stream:
            child = subprocess.Popen(['bash', str(project / argv[0]), *argv[1:]], env=env,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            record['current_child'] = {'pid': child.pid, 'start_token': token(child.pid), 'argv': argv, 'phase': name}
            save()
            try:
                code = child.wait(timeout=bound)
            except BaseException:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=90)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=30)
                raise
            finally:
                record['current_child'] = None
            record['phases'][name] = code
            save()
            return code
    try:
        with zipfile.ZipFile(args.archive) as bundle:
            assert bundle.testzip() is None
            assert len(bundle.namelist()) == len(set(bundle.namelist())) == len(lock['files'])
            assert set(bundle.namelist()) == {r['path'] for r in lock['files']}
            for item in lock['files']:
                target = project / item['path']
                assert target.resolve().is_relative_to(project.resolve())
                raw = bundle.read(item['path'])
                assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256']
                assert hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest() == item['git_blob_sha1']
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as stream:
                    stream.write(raw)
        assert step('selfcheck', ['verify.sh', '--selfcheck'], 300) == 0
        assert step('bench_plan', ['bench.sh', '--plan'], 300) == 0
        assert step('dry_run', ['run.sh', '--dry-run', '--run-id', run_id], 300) == 0
        planned = json.loads((OUT / 'dry_run.log').read_text())['configuration']
        baseline = json.loads((project / 'publisher031/complete1200-actual-profile.json').read_text())
        assert normalize(planned, planned['task'], run_id) == normalize(baseline, baseline['task'], baseline['run_id'])
        assert not any(ports_open().values())
        record.update(state='launching_fixed_root_profile', dry_run_profile_matches_full1200=True, GPU_started=True)
        save()
        assert step('launch', ['run.sh', '--run-id', run_id], 1500) == 0
        actual = json.loads((project / f'candidate/runtime/config-{run_id}.json').read_text())
        assert normalize(actual, actual['task'], run_id) == normalize(baseline, baseline['task'], baseline['run_id'])
        record['actual_profile_matches_full1200_except_task_UUID'] = True
        assert step('environment', ['env_check.sh'], 180) == 0
        environment = json.loads((project / 'env.log').read_text())
        assert environment['device_probe']['exit_code'] == 0
        assert json.loads(environment['device_probe']['stdout'])['gpu_available']
        assert step('smoke', ['smoke.sh'], 900) == 0
        record.update(state='generating_frozen_root100')
        save()
        assert step('verify', ['verify.sh', '--run-id', run_id, '--output', str(OUT / 'reproduction')], 7500) == 0
        result = json.loads((OUT / 'reproduction/progress.json').read_text())
        assert result['complete'] and result['generated'] == result['model_requests_executed'] == 100
        record['root100_generation_complete'] = True
    except BaseException as exc:
        record['error'] = repr(exc)
    finally:
        cfg = project / f'candidate/runtime/config-{run_id}.json'
        if cfg.exists():
            try:
                record['owned_service_stopped'] = step('stop', ['stop.sh', '--run-id', run_id], 120) == 0
            except BaseException as exc:
                record['cleanup_error'] = repr(exc)
        else:
            record['owned_service_stopped'] = not record['GPU_started']
        with socket.socket() as sock:
            record['owned_service_port_closed'] = sock.connect_ex(('127.0.0.1', 30139)) != 0
        record['root_GPU_execution_complete'] = bool(record['root100_generation_complete'] and record['owned_service_stopped']
                and record['owned_service_port_closed'] and not record.get('error') and not record.get('cleanup_error'))
        record.update(state='root_GPU100_complete_pending_independent_audit' if record['root_GPU_execution_complete'] else 'root_failed_partial_preserved', ended_utc=now())
        save()
    archive = Path('/root/session031-auk-root-replay001.zip')
    manifest = []
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        for folder, dirs, names in os.walk(OUT, onerror=lambda error: (_ for _ in ()).throw(error)):
            dirs[:] = sorted(n for n in dirs if n != '__pycache__')
            for name in sorted(names):
                path = Path(folder) / name
                assert not path.is_symlink()
                rel = path.relative_to(OUT).as_posix()
                bundle.write(path, rel)
                manifest.append({'path': rel, 'bytes': path.stat().st_size, 'sha256': sha(path)})
        bundle.writestr('MANIFEST.json', json.dumps({'files': manifest}, indent=2))
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
