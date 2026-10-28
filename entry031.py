"""Portable root interfaces for the unchanged, measured AuK100s service profile.

The fixed100 root replay is separate from the preserved complete3x400 accuracy.
"""
from pathlib import Path
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import math
import os
import sys
import time
import urllib.request
import wave

import entry as original


ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'publisher031/data'
SAMPLE_SHA = 'd056bd4a0b0402ec7fa98887af0d303cda1d6cda7548e8017d31e58f6cc01a57'
FULL_SHA = 'ea61e12ac36e6c1520a7c41ed608ebcbc7471933d47f5964b1e5dde02cc98701'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def samples():
    full = DATA / 'samples.jsonl'
    subset = ROOT / 'publisher031/auk-root-quarter100-frozen-samples.jsonl'
    assert sha(full.read_bytes()) == FULL_SHA and sha(subset.read_bytes()) == SAMPLE_SHA
    all_rows, selected = jsonl(full), jsonl(subset)
    mapping = {r['id']: r for r in all_rows}
    assert len(mapping) == len(all_rows) == 400
    assert len(selected) == len({r['id'] for r in selected}) == 100
    for sample in all_rows:
        ref = DATA / sample['audio']
        assert ref.resolve().is_relative_to(DATA.resolve()) and not ref.is_symlink()
        assert sha(ref.read_bytes()) == sample['sha256']
    for row in selected:
        assert row == mapping[row['id']]
    return all_rows, selected


def body(sample, seed=1234):
    ref = DATA / sample['audio']
    assert sha(ref.read_bytes()) == sample['sha256']
    return {'model': 'tencent-auk-bw-port006', 'input': sample['text'], 'ref_text': sample['ref_text'],
            'ref_audio': 'data:audio/wav;base64,' + base64.b64encode(ref.read_bytes()).decode(),
            'seed': seed, 'response_format': 'wav'}


def normalize(value, task, run_id):
    if isinstance(value, dict):
        return {k: normalize(v, task, run_id) for k, v in value.items() if k != 'run_id'}
    if isinstance(value, list):
        return [normalize(v, task, run_id) for v in value]
    return value.replace(task, '__TASK__').replace(run_id, '__RUN__') if isinstance(value, str) else value


def check_owned(run_id):
    original.owned_supervisor.checked_run_id(run_id)
    state = load(ROOT / 'candidate/runtime/state.json')
    assert state['run_id'] == run_id and state['status'] == 'ready'
    assert original.owned_supervisor.matches(state['server_pid'], state['server_token'])
    config = load(ROOT / f'candidate/runtime/config-{run_id}.json')
    baseline = load(ROOT / 'publisher031/complete1200-actual-profile.json')
    assert normalize(config, config['task'], run_id) == normalize(baseline, baseline['task'], baseline['run_id'])
    return config


def request(sample, destination, url):
    raw = json.dumps(body(sample)).encode()
    start = time.perf_counter()
    req = urllib.request.Request(url + '/v1/audio/speech', raw, {'Content-Type': 'application/json'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=900) as response:
        assert response.status == 200
        first = response.read(1)
        first_at = time.perf_counter()
        audio = first + response.read(64 * 1024**2)
    ended = time.perf_counter()
    with wave.open(io.BytesIO(audio), 'rb') as wav:
        assert wav.getnchannels() == 1 and wav.getsampwidth() == 2 and wav.getframerate() == 24000
        seconds = wav.getnframes() / 24000
        assert seconds > 0 and any(wav.readframes(wav.getnframes()))
    expected = math.ceil(sample['estimated_output_seconds'] * 50) / 50
    assert abs(seconds - expected) <= .041, ('Audio duration differs from the unchanged profile', sample['id'])
    target = destination / (sample['id'] + '.wav')
    with target.open('xb') as stream:
        stream.write(audio)
    return {'id': sample['id'], 'seed': 1234, 'file': 'audio/' + target.name,
            'sha256': sha(audio), 'response_bytes': len(audio), 'seconds': seconds,
            'expected_seconds': expected, 'request_sha256': sha(raw), 'wall_seconds': ended - start,
            'first_body_byte_seconds': first_at - start, 'RTF': (ended - start) / seconds}


def selfcheck():
    all_rows, selected = samples()
    full_audit = load(ROOT / 'publisher031/auk-full1200-independent-audit.json')
    performance = load(ROOT / 'publisher031/auk-performance96-independent-audit.json')
    assert full_audit['audit_complete'] and full_audit['accuracy_numeric_pass']
    assert full_audit['total_generated_audio'] == 1200 and performance['measured_requests'] == 96
    assert performance['audit_complete'] and performance['same_service_config_except_task_UUID']
    return {'selfcheck_complete': True, 'source': original.source_binding(), 'public_split': len(all_rows),
            'root_replay_subset_count': len(selected), 'sample_list_sha256': SAMPLE_SHA,
            'full_accuracy_preserved': True, 'full_CER_percent_three_run_mean': full_audit['CER_percent_three_run_mean'],
            'full_SIM_three_run_mean': full_audit['SIM_three_run_mean'],
            'fresh_root_model_requests_executed': 0, 'GPU_executed': False}


def verify(args):
    _, selected = samples()
    plan = {'scope': 'Additional fixed-root reproduction subset, not the original split full accuracy',
            'original_samples': 400, 'selected_samples': 100, 'fraction': .25, 'seed': 1234,
            'sample_list_sha256': SAMPLE_SHA, 'input_request_sha256':
                {r['id']: sha(json.dumps(body(r)).encode()) for r in selected},
            'full_accuracy_result_preserved': True, 'model_requests_executed': 0}
    if args.plan or args.dry_run:
        return dict(plan, plan_only=True)
    if args.selfcheck:
        return selfcheck()
    if not args.run_id:
        raise ValueError('Root replay requires the exact --run-id of this project service')
    config = check_owned(args.run_id)
    output = Path(args.output or ROOT / 'root-replay031').absolute()
    output.mkdir(parents=True, exist_ok=False)
    audio = output / 'audio'
    audio.mkdir()
    (output / 'samples.jsonl').write_bytes((ROOT / 'publisher031/auk-root-quarter100-frozen-samples.jsonl').read_bytes())
    record = dict(plan, state='generating_fixed_root100', run_id=args.run_id, started_utc=original.stamp(),
                  generated=0, complete=False, output=str(output), actual_profile_matches_full1200=True)
    def save():
        tmp = output / 'progress.tmp'
        tmp.write_text(json.dumps(record, indent=2) + '\n')
        tmp.replace(output / 'progress.json')
    save()
    deadline = time.monotonic() + 7200
    with (output / 'requests.jsonl').open('x', encoding='utf-8') as stream:
        for sample in sorted(selected, key=lambda r: (-r['estimated_output_seconds'], r['id'])):
            assert time.monotonic() < deadline, 'Root replay bound; completed normal samples are retained'
            row = request(sample, audio, config['url'])
            stream.write(json.dumps(row) + '\n')
            stream.flush()
            record['generated'] += 1
            record['model_requests_executed'] += 1
            save()
    record.update(complete=True, state='root100_generation_complete_metrics_separate', ended_utc=original.stamp(),
                  accuracy_metric_computed_here=False, official_acceptance=False, submitted=False)
    save()
    original.canonical_log('verify_result.json', record)
    return record


def bench(args):
    all_rows, _ = samples()
    selected = sorted((r for r in all_rows if 1 <= r['estimated_output_seconds'] <= 15), key=lambda r: r['id'])[:16]
    plan = {'scope': 'Buffered-WAV AuK API performance; not text-token six-load benchmark',
            'concurrencies': [1, 2, 4], 'rounds': 2, 'requests_per_group': 16, 'measured_requests': 96,
            'separate_warmup_requests': 6, 'selected_ids': [r['id'] for r in selected],
            'TTFA_boundary': 'First response-body byte; not streaming speech TTFT', 'model_requests_executed': 0}
    if args.plan or args.dry_run:
        return dict(plan, plan_only=True)
    if not args.run_id:
        raise ValueError('Benchmark requires the exact owned --run-id')
    config = check_owned(args.run_id)
    output = Path(args.output or ROOT / 'bench-replay031').absolute()
    output.mkdir(parents=True, exist_ok=False)
    groups = []
    for iteration in (1, 2):
        for concurrency in (1, 2, 4):
            folder = output / f'round{iteration}_c{concurrency}'
            warm = folder / 'warmup'
            warm.mkdir(parents=True)
            original.write(folder / 'warmup.json', request(selected[0], warm, config['url']))
            audio = folder / 'audio'
            audio.mkdir()
            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                values = list(pool.map(lambda s: request(s, audio, config['url']), selected))
            elapsed = time.perf_counter() - started
            (folder / 'requests.jsonl').write_text(''.join(json.dumps(v) + '\n' for v in values))
            total = sum(v['seconds'] for v in values)
            summary = {'round': iteration, 'concurrency': concurrency, 'requests': 16, 'wall_seconds': elapsed,
                       'audio_seconds': total, 'RTFX': total / elapsed, 'requests_per_second': 16 / elapsed,
                       'mean_RTF': sum(v['RTF'] for v in values) / 16,
                       'mean_first_body_byte_seconds': sum(v['first_body_byte_seconds'] for v in values) / 16}
            original.write(folder / 'summary.json', summary)
            groups.append(summary)
    return dict(plan, groups=groups, measured_requests_complete=96, model_requests_executed=102, official_acceptance=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['prepare','launch','stop','smoke','env','verify','bench'])
    ap.add_argument('--python', default=os.environ.get('PYTHON_BIN', str(ROOT / 'runtime/native-venv/bin/python')))
    ap.add_argument('--run-id')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--selfcheck', action='store_true')
    ap.add_argument('--plan', action='store_true')
    ap.add_argument('--output')
    args = ap.parse_args()
    original.source_binding()
    if args.action == 'verify':
        print(json.dumps(verify(args), ensure_ascii=False, indent=2))
        return 0
    if args.action == 'bench':
        print(json.dumps(bench(args), ensure_ascii=False, indent=2))
        return 0
    if args.action == 'launch' and args.dry_run:
        run_id = args.run_id or '1' * 32
        config = original.candidate_runtime.configuration(run_id)
        config['command'][0] = args.python
        print(json.dumps({'dry_run': True, 'run_id': run_id, 'configuration': config, 'environment_modified': False}, indent=2))
        return 0
    forwarded = [args.action, '--python', args.python]
    if args.run_id:
        forwarded += ['--run-id', args.run_id]
    if args.dry_run:
        forwarded += ['--dry-run']
    return original.main(forwarded)


if __name__ == '__main__':
    raise SystemExit(main())
