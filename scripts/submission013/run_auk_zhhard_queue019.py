"""After the owned Robo queue, generate all400 zh-hard cases three times on BW.

Preserve all audio and scores; compute SIM after stopping the owned AuK service.
Chinese CER is a separate CPU metric stage, never replaced by a model smoke.
"""
from pathlib import Path
import base64
import datetime as dt
import hashlib
import io
import json
import math
import os
import resource
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import uuid
import wave
import zipfile


STAGE = Path('/root/private_data/guoxin-session019/auk-zhhard-stage001')
OUT = Path('/root/private_data/guoxin-session019/auk-zhhard001')
ROBO = Path('/root/private_data/guoxin-session017/robo-refspatial001')
BASE = Path('/root/private_data/guoxin-session010')
SEEDS = (1234, 2345, 3456)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def save(record):
    p = OUT / 'progress.json'
    temporary = p.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(p)


def owned_alive(pid, script):
    try:
        p = Path('/proc') / str(pid)
        return p.joinpath('stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z' and script.encode() in p.joinpath('cmdline').read_bytes()
    except (FileNotFoundError, ProcessLookupError):
        return False


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    assert json.loads((STAGE / 'IMPORT-READY.json').read_text())['ready']
    data = STAGE / 'data'
    ready = json.loads((data / 'READY.json').read_text())
    assert ready['count'] == 400 and ready['full_split'] and ready['selection_before_any_predictions']
    assert sha(data / 'samples.jsonl') == ready['samples_sha256'] == 'ea61e12ac36e6c1520a7c41ed608ebcbc7471933d47f5964b1e5dde02cc98701'
    samples = [json.loads(r) for r in (data / 'samples.jsonl').read_text().splitlines()]
    assert len(samples) == len({r['id'] for r in samples}) == 400
    for row in samples:
        p = data / row['audio']
        assert p.resolve().is_relative_to(data.resolve()) and not p.is_symlink() and sha(p) == row['sha256']
        assert 0 < row['estimated_output_seconds'] <= 100
    # Shape preflight is the first normal scored sample, not an extra retry.
    selected = sorted(samples, key=lambda r: (-r['estimated_output_seconds'], r['id']))
    OUT.mkdir(parents=True, exist_ok=False)
    record = {'started_utc': now(), 'controller_pid': os.getpid(), 'state': 'waiting_owned_robo_queue',
              'source_sha256': sha(__file__), 'dataset_count': 400, 'complete_rounds_required': 3,
              'seeds': SEEDS, 'full_target_requests': 1200, 'data_sha256': sha(data / 'samples.jsonl'),
              'longest_case_first_before_predictions': True, 'max_seconds_profile': 100,
              'shape_validation_complete': False, 'GPU_started': False, 'phases': {},
              'generation_rows': 0, 'SIM_rows': 0, 'numeric_pass': None, 'submitted': False,
              'instance_shutdown_requested': False}
    save(record)
    shutil.copyfile(__file__, OUT / 'executed-controller.py')
    deadline = time.monotonic() + 172800
    while True:
        if not owned_alive(43964, 'run_refspatial_queue017.py') and (ROBO / 'progress.json').exists():
            prior = json.loads((ROBO / 'progress.json').read_text())
            assert prior['state'] in ('validated_pending_final_delivery', 'accuracy_or_performance_not_qualified', 'failed')
            assert prior.get('owned_service_stopped') and prior.get('phases', {}).get('stop') == 0, 'Robo cleanup not confirmed'
            record['previous_robo_progress_sha256'] = sha(ROBO / 'progress.json')
            break
        assert time.monotonic() < deadline, 'Existing owned work is still active; nothing was stopped'
        time.sleep(20)
    for port in (30000, 30001, 30002, 30109, 30119, 30129, 30139):
        with socket.socket() as sock:
            assert sock.connect_ex(('127.0.0.1', port)) != 0, ('Existing service left untouched', port)
    candidate = OUT / 'candidate'
    shutil.copytree(STAGE / 'candidate', candidate)
    assert sha(candidate / 'serve.py') == 'a7464a867538ebc9e9061dcd411874f5900fcb2ca529db973d7bc428f4051fe7'
    env = os.environ.copy()
    env.update(AUK_MAX_SECONDS='100', AUK_MAX_RUNTIME_SECONDS='10800', AUK_LOAD_TIMEOUT='1200',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', ROCM_PATH='/opt/dtk', HIP_PATH='/opt/dtk/hip')
    env['PATH'] = '/opt/ucx/bin:/opt/dtk/bin:/opt/dtk/llvm/bin:/opt/dtk/hip/bin:/opt/hyhal/bin:' + env.get('PATH', '/usr/bin:/bin')
    env['LD_LIBRARY_PATH'] = ':'.join(['/opt/ucx/lib', '/opt/dtk/dcc/gcvm/lib', '/opt/dtk/hip/lib',
        '/opt/dtk/llvm/lib', '/opt/dtk/lib', '/opt/dtk/lib64', '/opt/hyhal/lib', '/opt/hyhal/lib64',
        '/opt/dtk/dushmem/lib', '/opt/dtk/opencl/lib', '/opt/mpi/lib', '/opt/hwloc/lib', env.get('LD_LIBRARY_PATH', '')])

    def step(command, label, timeout):
        with (OUT / (label + '.log')).open('xb') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            try:
                code = child.wait(timeout=timeout)
            except BaseException:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                raise
        record['phases'][label] = code
        save(record)
        return code

    run_id = uuid.uuid4().hex
    record.update(state='generating_three_complete_rounds', GPU_started=True, run_id=run_id,
                  candidate_serve_sha256=sha(candidate / 'serve.py'), model_lock_sha256=sha(candidate / 'model-lock.json'))
    save(record)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    generated = []
    launched = False
    try:
        launched = True
        assert step([sys.executable, str(candidate / 'runtime.py'), 'launch', '--run-id', run_id], 'launch', 1500) == 0
        for iteration, seed in enumerate(SEEDS, 1):
            folder = OUT / f'run{iteration}'
            (folder / 'audio').mkdir(parents=True, exist_ok=False)
            with (folder / 'requests.jsonl').open('x') as log:
                for sample in selected:
                    body = {'model': 'tencent-auk-bw-port006', 'input': sample['text'], 'ref_text': sample['ref_text'],
                            'ref_audio': 'data:audio/wav;base64,' + base64.b64encode((data / sample['audio']).read_bytes()).decode(),
                            'seed': seed, 'response_format': 'wav'}
                    raw = json.dumps(body).encode()
                    req = urllib.request.Request('http://127.0.0.1:30139/v1/audio/speech', raw, {'Content-Type': 'application/json'})
                    record['active_sample'] = {'id': sample['id'], 'iteration': iteration, 'seed': seed,
                                               'request_sha256': hashlib.sha256(raw).hexdigest()}
                    save(record)
                    started = time.perf_counter()
                    with opener.open(req, timeout=900) as response:
                        audio = response.read(64 * 1024**2 + 1)
                        assert len(audio) <= 64 * 1024**2
                    path = folder / 'audio' / (sample['id'] + '.wav')
                    # Preserve the returned bytes even if the following header or
                    # duration check fails. Never discard a failed long shape.
                    with path.open('xb') as stream:
                        stream.write(audio)
                    with wave.open(io.BytesIO(audio), 'rb') as wav:
                        assert wav.getnchannels() == 1 and wav.getsampwidth() == 2 and wav.getframerate() == 24000
                        seconds = wav.getnframes() / 24000
                        assert wav.getnframes() > 0
                    expected = math.ceil(sample['estimated_output_seconds'] * 50) / 50
                    assert abs(seconds - expected) <= .041, ('Truncation or duration mismatch', sample['id'], seconds, expected)
                    row = {'id': sample['id'], 'iteration': iteration, 'seed': seed,
                           'file': path.relative_to(folder).as_posix(), 'sha256': hashlib.sha256(audio).hexdigest(),
                           'seconds': seconds, 'expected_seconds': expected, 'request_sha256': hashlib.sha256(raw).hexdigest(),
                           'wall_seconds': time.perf_counter() - started}
                    generated.append(dict(row, relative_to_output=path.relative_to(OUT).as_posix()))
                    log.write(json.dumps(row) + '\n')
                    log.flush()
                    record['generation_rows'] = len(generated)
                    record['shape_validation_complete'] = True
                    if len(generated) == 1 or len(generated) % 10 == 0:
                        save(record)
                        print(json.dumps({'generation_done': len(generated), 'target': 1200}), flush=True)
        record['generation_complete'] = len(generated) == 1200
    except BaseException as exc:
        record['generation_error'] = repr(exc)
    finally:
        if launched:
            try:
                step([sys.executable, str(candidate / 'runtime.py'), 'stop', '--run-id', run_id], 'stop', 100)
            except BaseException as exc:
                record['cleanup_error'] = repr(exc)
        with socket.socket() as sock:
            record['owned_service_stopped'] = sock.connect_ex(('127.0.0.1', 30139)) != 0
        save(record)
    if record.get('generation_complete') and record['owned_service_stopped'] and record['phases'].get('stop') == 0:
        record['state'] = 'complete_three_rounds_speaker_similarity'
        save(record)
        import torch
        torch.set_num_threads(4)
        assets = BASE / 'wavlm-sim'
        assert json.loads((assets / 'IMPORT-READY.json').read_text())['ready']
        asset_identity = json.loads((assets / 'READY.json').read_text())
        for item in asset_identity['files']:
            row = item['file']
            assert (assets / row['path']).stat().st_size == row['size']
            assert sha(assets / row['path']) == row['lfs']['oid']
        sys.path.insert(0, str(assets))
        from speaker_similarity import WavLMSpeakerSimilarity
        engine = WavLMSpeakerSimilarity(finetune_checkpoint=assets / 'wavlm_large_finetune.pth', wavlm_base=assets / 'wavlm_large.pt', device='cuda')
        mapped = {s['id']: s for s in samples}
        scores = []
        with (OUT / 'sim.jsonl').open('x') as log:
            for row in generated:
                sample = mapped[row['id']]
                path = OUT / row['relative_to_output']
                assert sha(path) == row['sha256']
                value, = engine.score_batch([str(data / sample['audio'])], [str(path)])
                assert math.isfinite(value) and -100.001 <= value <= 100.001
                score = {'id': row['id'], 'iteration': row['iteration'], 'similarity': value / 100,
                         'audio_sha256': row['sha256'], 'reference_sha256': sample['sha256']}
                scores.append(score)
                log.write(json.dumps(score) + '\n')
                log.flush()
                record['SIM_rows'] = len(scores)
                if len(scores) % 20 == 0:
                    save(record)
            assert len(scores) == 1200
        record['SIM_mean'] = sum(r['similarity'] for r in scores) / 1200
        record['SIM_complete'] = True
        record['SIM_scorer_sha256'] = sha(assets / 'speaker_similarity.py')
        del engine
        torch.cuda.empty_cache()
    record.update(state='generation_and_SIM_complete_CER_pending' if record.get('SIM_complete') else 'generation_or_SIM_failed',
                  ended_utc=now(), CER_complete=False, numeric_pass=None, technical_delivery_complete=False)
    save(record)
    entries = []
    with zipfile.ZipFile('/root/session019-auk-zhhard001.zip', 'x', zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        for directory, dirs, names in os.walk(OUT, onerror=lambda error: (_ for _ in ()).throw(error)):
            dirs[:] = sorted(d for d in dirs if d != '__pycache__')
            for name in sorted(names):
                p = Path(directory) / name
                assert not p.is_symlink()
                rel = p.relative_to(OUT).as_posix()
                bundle.write(p, rel)
                entries.append({'path': rel, 'bytes': p.stat().st_size, 'sha256': sha(p)})
        bundle.writestr('MANIFEST.json', json.dumps({'files': entries}, indent=2))
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        if (OUT / 'progress.json').exists():
            record = json.loads((OUT / 'progress.json').read_text())
            record.update(state='failed', error=repr(exc), ended_utc=now(), submitted=False)
            save(record)
        raise
