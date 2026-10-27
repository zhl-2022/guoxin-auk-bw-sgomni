"""Read the finished owned BW run and automatically compute all three CPU CERs.

SSH credentials arrive only through stdin, remain in RAM, and are never saved.
Only reads the remote instance. No GPU launches or instance lifecycle actions.
"""
from pathlib import Path
import base64
import datetime as dt
import hashlib
import json
import os
import shlex
import stat
import subprocess
import sys
import time
import zipfile

import paramiko


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'evidence/session019'
STATE = OUT / 'auk-cpu-followup-progress.json'
REMOTE = '/root/private_data/guoxin-session019/auk-zhhard001'
ARCHIVE = '/root/session019-auk-zhhard001.zip'
HOST = 'ssh.zzai.scnet.cn'
PORT = 11909


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def save(record):
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(STATE)


def connect(password):
    known = paramiko.HostKeys(str(Path.home() / '.ssh/known_hosts'))
    client = paramiko.SSHClient()
    # Reuse the exact gateway key previously verified by the authorized SSH work.
    aliases = [keys for alias, keys in known.items() if HOST in alias]
    assert aliases
    for keys in aliases:
        for kind, key in keys.items():
            client.get_host_keys().add(f'[{HOST}]:{PORT}', kind, key)
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect(HOST, port=PORT, username='root', password=password, timeout=15,
                   auth_timeout=20, banner_timeout=20, allow_agent=False, look_for_keys=False)
    key = client.get_transport().get_remote_server_key()
    fingerprint = base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip('=')
    assert fingerprint == 'FDuS5UcY7KDwYbYutvA6xtGhtg9y9XDKmuVftEMWtxg'
    client.get_transport().set_keepalive(20)
    return client


def execute(client, code):
    _, output, error = client.exec_command('python3 -c ' + shlex.quote(code), timeout=40)
    raw = output.read()
    message = error.read()
    assert output.channel.recv_exit_status() == 0, message.decode(errors='replace')
    return json.loads(raw)


def main():
    assert not STATE.exists()
    if os.name == 'nt':
        import ctypes
        kernel = ctypes.windll.kernel32
        handle = kernel.GetStdHandle(-10)
        mode = ctypes.c_ulong()
        if kernel.GetConsoleMode(handle, ctypes.byref(mode)):
            assert kernel.SetConsoleMode(handle, mode.value & ~0x0004)
    print(json.dumps({'password_required_for_authorized_port': PORT, 'credentials_persisted': False}), flush=True)
    authentication = json.loads(sys.stdin.readline())
    password = authentication.pop('password')
    authentication.clear()
    record = {'started_utc': now(), 'watcher_pid': os.getpid(), 'state': 'waiting_owned_BW_generation_and_SIM',
              'credentials_persisted': False, 'source_sha256': sha(__file__), 'remote': REMOTE,
              'BW_model_requests': 0, 'GPU_launched_by_watcher': False, 'submitted': False}
    save(record)
    deadline = time.monotonic() + 172800
    client = None
    snapshot = None
    while True:
        try:
            if client is None:
                client = connect(password)
            code = f'''from pathlib import Path
import json
p=Path({REMOTE!r})/'progress.json'
if not p.exists(): print(json.dumps({{'exists':False}}))
else:
 r=json.loads(p.read_text()); pid=r['controller_pid']; c=Path('/proc')/str(pid)/'cmdline'
 alive=c.exists() and b'run_auk_zhhard_queue019.py' in c.read_bytes()
 print(json.dumps({{'exists':True,'alive':alive,'progress':r,'archive_exists':Path({ARCHIVE!r}).is_file()}}))
'''
            snapshot = execute(client, code)
            if snapshot.get('exists') and not snapshot['alive']:
                remote = snapshot['progress']
                assert remote['state'] == 'generation_and_SIM_complete_CER_pending', ('Remote candidate did not finish', remote)
                assert remote['generation_complete'] and remote['SIM_complete']
                assert remote['generation_rows'] == remote['SIM_rows'] == 1200
                assert snapshot['archive_exists'] and remote['owned_service_stopped'] and remote['phases']['stop'] == 0
                break
            record['last_successful_check_utc'] = now()
            record['remote_state'] = snapshot.get('progress', {}).get('state')
            record['remote_generation_rows'] = snapshot.get('progress', {}).get('generation_rows', 0)
            save(record)
        except (paramiko.AuthenticationException, paramiko.BadHostKeyException):
            raise
        except (OSError, EOFError, paramiko.SSHException) as exc:
            if client is not None:
                client.close()
            client = None
            record.update(last_connection_error_type=type(exc).__name__, last_connection_error_utc=now())
            save(record)
        assert time.monotonic() < deadline, 'Owned remote run has not completed in48h; no GPU was launched or stopped'
        time.sleep(45)
    record.update(state='copying_finished_raw_archive', remote_final=snapshot['progress'])
    save(record)
    local = ROOT / 'dist/session019-auk-zhhard001.zip'
    assert not local.exists()
    with client.open_sftp() as sftp:
        remote_size = sftp.stat(ARCHIVE).st_size
    _, source, error = client.exec_command('cat -- ' + shlex.quote(ARCHIVE), timeout=1800)
    with local.open('xb') as target:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            target.write(chunk)
    assert source.channel.recv_exit_status() == 0, error.read().decode(errors='replace')
    assert local.stat().st_size == remote_size
    client.close()
    password = None
    extracted = OUT / 'auk-zhhard001'
    extracted.mkdir(exist_ok=False)
    with zipfile.ZipFile(local) as bundle:
        assert bundle.testzip() is None
        rows = json.loads(bundle.read('MANIFEST.json'))['files']
        assert len(bundle.namelist()) == len(set(bundle.namelist())) == len(rows) + 1
        assert set(bundle.namelist()) == {r['path'] for r in rows} | {'MANIFEST.json'}
        for row in rows:
            p = extracted / row['path']
            assert not Path(row['path']).is_absolute() and p.resolve().is_relative_to(extracted.resolve())
            assert not stat.S_ISLNK(bundle.getinfo(row['path']).external_attr >> 16)
            raw = bundle.read(row['path'])
            assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open('xb') as stream:
                stream.write(raw)
    assert json.loads((extracted / 'progress.json').read_text()) == snapshot['progress']
    assert sha(extracted / 'executed-controller.py') == '187a55cbf1729f1773ca2527268169d776d6a35c977608702f769f9ae5e0bba6'
    python = ROOT / 'external/auk-zh-eval019/Scripts/python.exe'
    worker = ROOT / 'scripts/submission013/score_auk_zh019.py'
    assert sha(worker) == 'f6e56c5737488de65d0da44d63852abfd0eb8be393285561fbce5f4e797eb4e7'
    results = []
    record['state'] = 'CPU_scoring_three_complete_400_rounds'
    save(record)
    for iteration in (1, 2, 3):
        score_dir = OUT / f'auk-zhhard-CER-run{iteration}'
        with (OUT / f'auk-zhhard-CER-run{iteration}.log').open('xb') as log:
            code = subprocess.run([str(python), str(worker), '--model', str(ROOT / 'datasets/session019/paraformer-zh'),
                  '--source', str(ROOT / 'evidence/session010/sources/seedtts-run-wer.py'),
                  '--data', str(ROOT / 'datasets/session019/seedtts-zhhard'),
                  '--predictions', str(extracted / f'run{iteration}/requests.jsonl'),
                  '--audio-root', str(extracted / f'run{iteration}'), '--output', str(score_dir)],
                  stdout=log, stderr=subprocess.STDOUT, timeout=14400).returncode
        assert code == 0, ('CPU CER stage failed', iteration)
        value = json.loads((score_dir / 'cer-result.json').read_text())
        assert value['complete'] and value['samples'] == 400 and value['excluded'] == 0
        results.append(value['CER_percent'])
        record['complete_CER_rounds'] = len(results)
        save(record)
    scores = [json.loads(r) for r in (extracted / 'sim.jsonl').read_text().splitlines()]
    assert len(scores) == len({(r['iteration'], r['id']) for r in scores}) == 1200
    samples = [json.loads(r) for r in (ROOT / 'datasets/session019/seedtts-zhhard/samples.jsonl').read_text(encoding='utf-8').splitlines()]
    ids = {r['id'] for r in samples}
    for iteration in (1, 2, 3):
        generated = [json.loads(r) for r in (extracted / f'run{iteration}/requests.jsonl').read_text().splitlines()]
        assert len(generated) == len({r['id'] for r in generated}) == 400
        assert {r['id'] for r in generated} == {r['id'] for r in scores if r['iteration'] == iteration} == ids
        mapping = {r['id']: r for r in generated}
        for r in scores:
            if r['iteration'] == iteration:
                assert r['audio_sha256'] == mapping[r['id']]['sha256']
    cer = sum(results) / 3
    sim = sum(r['similarity'] for r in scores) / 1200
    assert abs(sim - snapshot['progress']['SIM_mean']) < 1e-12
    record.update(state='full_accuracy_complete', ended_utc=now(), full_round_samples=[400, 400, 400],
                  CER_percent_rounds=results, CER_percent_mean=cer, SIM_mean=sim,
                  CER_upper=6.0282, SIM_lower=.76636, numeric_pass=cer <= 6.0282 and sim >= .76636,
                  archive_sha256=sha(local), all1200_raw_audio_manifest_verified=True,
                  technical_delivery_complete=False, official_acceptance=False, submitted=False)
    save(record)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        if STATE.exists():
            record = json.loads(STATE.read_text())
            record.update(state='failed', error=repr(exc), ended_utc=now(), submitted=False)
            save(record)
        raise
