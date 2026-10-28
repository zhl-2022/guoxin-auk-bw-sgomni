"""Portable SeedTTS SIM replay over the existing three complete audio runs.

The published scorer and checkpoints remain unchanged. This wrapper never calls
the AuK service. Plan mode imports no neural framework and does not open audio.
"""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import time
import wave

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_SHA = 'ea61e12ac36e6c1520a7c41ed608ebcbc7471933d47f5964b1e5dde02cc98701'
SCORER_SHA = '18528106c718384e6e096b79031c78fbc90890344d80ba6ae3d4cbadc0d2545e'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def load_lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def confined(root, relative):
    target = root / relative
    assert not target.is_symlink() and target.resolve().is_relative_to(root.resolve())
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'check', 'score'])
    parser.add_argument('--data', type=Path, default=ROOT / 'publisher031/data')
    parser.add_argument('--requests-root', type=Path, default=ROOT / 'publisher032/results/full1200')
    parser.add_argument('--audio-root', type=Path, required=True,
                        help='Root containing run1/audio, run2/audio and run3/audio')
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--max-seconds', type=int, default=7200)
    args = parser.parse_args()
    assert 60 <= args.max_seconds <= 10800
    assert sha(args.data / 'samples.jsonl') == SAMPLE_SHA
    samples = load_lines(args.data / 'samples.jsonl')
    mapped = {r['id']: r for r in samples}
    assert len(mapped) == len(samples) == 400
    scorer = Path(__file__).with_name('speaker_similarity.py')
    assert sha(scorer) == SCORER_SHA
    records = []
    for iteration, seed in enumerate((1234, 2345, 3456), 1):
        request_file = args.requests_root / f'run{iteration}/generation-requests.jsonl'
        assert sha(request_file) == sha(ROOT / f'publisher032/results/full1200/run{iteration}/generation-requests.jsonl')
        rows = load_lines(request_file)
        assert len(rows) == len({r['id'] for r in rows}) == 400
        assert {r['id'] for r in rows} == set(mapped)
        for row in rows:
            assert row['iteration'] == iteration and row['seed'] == seed
            sample = mapped[row['id']]
            audio = confined(args.audio_root, f'run{iteration}/' + row['file'])
            reference = confined(args.data, sample['audio'])
            records.append((row, sample, audio, reference))
    lock = json.loads(Path(__file__).with_name('SIM-assets-READY.json').read_text())
    protocol = {'UTC': dt.datetime.now(dt.timezone.utc).isoformat(), 'action': args.action,
                'scorer_source_sha256': SCORER_SHA, 'sample_list_sha256': SAMPLE_SHA,
                'runs': 3, 'samples_per_run': 400, 'total': 1200, 'excluded': 0,
                'device': args.device, 'batch_size': 1, 'score_scale': 'Published percentage /100',
                'AuK_generation_requests': 0, 'neural_scoring_executed': False,
                'historical_results_overwritten': False,
                'audio_root': str(args.audio_root), 'assets': str(args.assets),
                'audio_and_asset_bytes_verified': False,
                'new_wrapper_not_a_claim_of_new_metric_inference': True}
    if args.action == 'plan':
        print(json.dumps(protocol, ensure_ascii=False, indent=2))
        return 0
    for asset in lock['files']:
        row = asset['file']
        path = confined(args.assets, row['path'])
        assert path.stat().st_size == row['size'] and sha(path) == row['lfs']['oid']
    for row, sample, audio, reference in records:
        assert sha(audio) == row['sha256'] and sha(reference) == sample['sha256']
        with wave.open(str(audio), 'rb') as wav:
            assert wav.getnchannels() == 1 and wav.getsampwidth() == 2 and wav.getframerate() == 24000
            assert abs(wav.getnframes() / 24000 - row['expected_seconds']) <= .041
    protocol['audio_and_asset_bytes_verified'] = True
    if args.action == 'check':
        print(json.dumps(protocol, ensure_ascii=False, indent=2))
        return 0
    assert args.output, 'An exclusive new output directory is required'
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    torch.set_num_threads(4)
    spec = importlib.util.spec_from_file_location('unchanged_speaker_similarity037', scorer)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    engine = module.WavLMSpeakerSimilarity(
        finetune_checkpoint=args.assets / 'wavlm_large_finetune.pth',
        wavlm_base=args.assets / 'wavlm_large.pt', device=args.device)
    protocol.update(neural_scoring_executed=True, wrapper_sha256=sha(__file__), torch_version=torch.__version__)
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    deadline = time.monotonic() + args.max_seconds
    values = []
    with (args.output / 'sim.jsonl').open('x', encoding='utf-8') as stream:
        for row, sample, audio, reference in records:
            assert time.monotonic() < deadline, 'Bound reached; retain existing partial rows'
            value, = engine.score_batch([str(reference)], [str(audio)])
            assert math.isfinite(value) and -100.001 <= value <= 100.001
            value /= 100
            result = {'id': row['id'], 'iteration': row['iteration'], 'similarity': value,
                      'audio_sha256': row['sha256'], 'reference_sha256': sample['sha256']}
            stream.write(json.dumps(result, allow_nan=False) + '\n')
            stream.flush()
            values.append(value)
    result = dict(protocol, complete=len(values) == 1200, SIM_mean=sum(values) / len(values),
                  SIM_lower_relative_two_percent=.76636, official_acceptance=False, submitted=False)
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
