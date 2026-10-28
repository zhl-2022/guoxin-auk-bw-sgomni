"""Score the existing1200 AuK waveforms; never request or regenerate model audio."""
from pathlib import Path
import base64
import datetime as dt
import hashlib
import json
import math
import os
import socket
import sys
import time
import wave
import zipfile

GEN = Path('/root/private_data/guoxin-session023/auk-zhhard-B001')
OUT = Path('/root/private_data/guoxin-session030/auk-SIM-resume001')
DATA = Path('/root/private_data/guoxin-session019/auk-zhhard-stage001/data')
ASSETS = Path('/root/private_data/guoxin-session010/wavlm-sim')
ARCHIVE = Path('/root/session030-auk-zhhard-resume001.zip')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def main():
    assert socket.gethostname() == 'crdnotebook-2112237945233408001-ac3p4mowhz-70506'
    assert not Path('/proc/17403').exists() and not Path('/proc/18406').exists()
    for port in (30000, 30001, 30002, 30109, 30119, 30129, 30139):
        with socket.socket() as sock:
            sock.settimeout(1)
            assert sock.connect_ex(('127.0.0.1', port)) != 0
    old = json.loads((GEN / 'progress.json').read_text())
    assert old['generation_complete'] and old['generation_rows'] == 1200
    assert old['owned_service_stopped'] and old['phases']['stop'] == 0
    assert not old.get('SIM_complete') and old['SIM_rows'] == 0
    assert sha(GEN / 'executed-controller.py') == 'd09f51267cd0803e0e3dacd01bf99c87fa072f62a06a9385bc6dc80a84a863a9'
    assert sha(DATA / 'samples.jsonl') == 'ea61e12ac36e6c1520a7c41ed608ebcbc7471933d47f5964b1e5dde02cc98701'
    samples = [json.loads(s) for s in (DATA / 'samples.jsonl').read_text().splitlines()]
    mapped = {s['id']: s for s in samples}
    assert len(mapped) == len(samples) == 400
    generated = []
    for iteration, seed in enumerate((1234, 2345, 3456), 1):
        rows = [json.loads(s) for s in (GEN / f'run{iteration}/requests.jsonl').read_text().splitlines()]
        assert len(rows) == len({r['id'] for r in rows}) == 400
        assert {r['id'] for r in rows} == set(mapped)
        for row in rows:
            sample = mapped[row['id']]
            assert row['iteration'] == iteration and row['seed'] == seed
            audio = GEN / f'run{iteration}' / row['file']
            assert audio.resolve().is_relative_to(GEN.resolve()) and not audio.is_symlink()
            assert sha(audio) == row['sha256']
            ref = DATA / sample['audio']
            assert sha(ref) == sample['sha256']
            with wave.open(str(audio), 'rb') as wav:
                assert wav.getnchannels() == 1 and wav.getsampwidth() == 2 and wav.getframerate() == 24000
                assert abs(wav.getnframes() / 24000 - row['expected_seconds']) <= .041
            body = {'model': 'tencent-auk-bw-port006', 'input': sample['text'], 'ref_text': sample['ref_text'],
                    'ref_audio': 'data:audio/wav;base64,' + base64.b64encode(ref.read_bytes()).decode(),
                    'seed': seed, 'response_format': 'wav'}
            assert hashlib.sha256(json.dumps(body).encode()).hexdigest() == row['request_sha256']
            generated.append(dict(row, absolute_audio=str(audio)))
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / 'executed-SIM-controller.py').write_bytes(Path(__file__).read_bytes())
    record = {'UTC': now(), 'state': 'scoring_existing_complete1200_SIM', 'controller_pid': os.getpid(),
              'source_sha256': sha(__file__), 'generation_complete': True, 'generation_rows': 1200,
              'all_original_audio_requests_and_shapes_verified': True, 'original_generation_root': str(GEN),
              'original_generation_progress_sha256': sha(GEN / 'progress.json'), 'old_error_preserved': old.get('error'),
              'SIM_rows': 0, 'SIM_complete': False, 'CER_complete': False,
              'owned_service_stopped': True, 'phases': {'stop': 0}, 'AuK_generation_requests': 0,
              'numeric_pass': None, 'technical_delivery_complete': False, 'submitted': False,
              'Notebook_shutdown_requested': False}
    def save():
        p = OUT / 'progress.tmp'
        p.write_text(json.dumps(record, indent=2) + '\n')
        p.replace(OUT / 'progress.json')
    save()
    engine = None
    try:
        import torch
        torch.set_num_threads(4)
        assert json.loads((ASSETS / 'IMPORT-READY.json').read_text())['ready']
        for item in json.loads((ASSETS / 'READY.json').read_text())['files']:
            r = item['file']
            assert (ASSETS / r['path']).stat().st_size == r['size']
            assert sha(ASSETS / r['path']) == r['lfs']['oid']
        sys.path.insert(0, str(ASSETS))
        from speaker_similarity import WavLMSpeakerSimilarity
        engine = WavLMSpeakerSimilarity(finetune_checkpoint=ASSETS / 'wavlm_large_finetune.pth',
                                      wavlm_base=ASSETS / 'wavlm_large.pt', device='cuda')
        record['SIM_scorer_sha256'] = sha(ASSETS / 'speaker_similarity.py')
        total = 0.0
        deadline = time.monotonic() + 7200
        with (OUT / 'sim.jsonl').open('x') as stream:
            for row in generated:
                assert time.monotonic() < deadline, 'SIM boundary; completed scores retained'
                sample = mapped[row['id']]
                value, = engine.score_batch([str(DATA / sample['audio'])], [row['absolute_audio']])
                assert math.isfinite(value) and -100.001 <= value <= 100.001
                score = {'id': row['id'], 'iteration': row['iteration'], 'similarity': value / 100,
                         'audio_sha256': row['sha256'], 'reference_sha256': sample['sha256']}
                stream.write(json.dumps(score) + '\n')
                stream.flush()
                total += value / 100
                record['SIM_rows'] += 1
                if record['SIM_rows'] % 20 == 0:
                    save()
        assert record['SIM_rows'] == 1200
        record.update(SIM_complete=True, SIM_mean=total / 1200)
    except BaseException as exc:
        record['SIM_error'] = repr(exc)
    finally:
        if engine is not None:
            del engine
            torch.cuda.empty_cache()
        record.update(state='generation_and_SIM_complete_CER_pending' if record['SIM_complete'] else 'SIM_failed_complete_generation_preserved', ended_utc=now())
        save()
    entries = []
    with zipfile.ZipFile(ARCHIVE, 'x', zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        for base, prefix in [(GEN, 'generation'), (OUT, 'SIM')]:
            for folder, dirs, names in os.walk(base, onerror=lambda error: (_ for _ in ()).throw(error)):
                dirs[:] = sorted(d for d in dirs if d != '__pycache__')
                for name in sorted(names):
                    p = Path(folder) / name
                    assert p.is_file() and not p.is_symlink()
                    rel = prefix + '/' + p.relative_to(base).as_posix()
                    bundle.write(p, rel)
                    entries.append({'path': rel, 'bytes': p.stat().st_size, 'sha256': sha(p)})
        bundle.writestr('MANIFEST.json', json.dumps({'files': entries, 'all_original_generation_preserved': True,
                        'new_AuK_generation_requests': 0}, indent=2))
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
