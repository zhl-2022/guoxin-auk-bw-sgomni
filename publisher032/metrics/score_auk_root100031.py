"""CPU CER for the fixed100 root replay, using the exact published normalizer.

This is a metric model, not AuK inference or a replacement for BW-generated audio.
No target transcript is supplied to the recognizer.
"""
from pathlib import Path
import argparse
import ast
import datetime as dt
import hashlib
import importlib.metadata as metadata
import json
import os
import string
import time
import wave


SCORER_SHA = 'be5e608e3fa1d99ddeb92853dd9e768348633702b51c6a607ed557dcf7bbb2ea'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def original_metric(source):
    assert sha(source) == SCORER_SHA
    from jiwer import compute_measures
    from zhon.hanzi import punctuation
    tree = ast.parse(source.read_text(encoding='utf-8'))
    function, = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'process_one']
    module = ast.Module(body=[function], type_ignores=[])
    scope = {'compute_measures': compute_measures, 'punctuation_all': punctuation + string.punctuation, 'lang': 'zh'}
    exec(compile(module, str(source), 'exec'), scope)
    return scope['process_one']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--probe', action='store_true')
    ap.add_argument('--data', type=Path)
    ap.add_argument('--predictions', type=Path)
    ap.add_argument('--audio-root', type=Path)
    ap.add_argument('--selected-samples', type=Path, required=True)
    args = ap.parse_args()
    model = args.model.resolve()
    ready = load(model / 'READY.json')
    assert ready['revision'] == 'd7811ee3ac581fbcfdeb37c98c6ba674028433dc' and ready['all_files_hashed']
    for row in ready['files']:
        target = model / row['path']
        assert target.resolve().is_relative_to(model) and not target.is_symlink()
        assert target.stat().st_size == row['bytes'] and sha(target) == row['sha256']
    process_one = original_metric(args.source)
    assert process_one('你好，世界！', '你好世界')[2] == 0
    assert process_one('你们好', '你好')[2] == .5
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', MODELSCOPE_OFFLINE='1',
                      OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    import torch
    torch.set_num_threads(4)
    from funasr import AutoModel
    from zhconv import convert
    start = time.perf_counter()
    engine = AutoModel(model=str(model), device='cpu', disable_update=True, disable_pbar=True, log_level='ERROR')
    protocol = {'UTC': dt.datetime.now(dt.timezone.utc).isoformat(), 'metric': 'SeedTTS Chinese CER',
                'metric_device': 'cpu', 'metric_model': 'funasr/paraformer-zh',
                'metric_model_revision': ready['revision'], 'scorer_source_sha256': SCORER_SHA,
                'worker_sha256': sha(__file__), 'target_text_given_to_recognizer': False,
                'BW_model_requests': 0, 'GPU_executed': False,
                'versions': {name: metadata.version(name) for name in ('torch', 'torchaudio', 'numpy', 'scipy', 'funasr', 'jiwer', 'zhconv', 'zhon')},
                'model_cpu_initialization_seconds': time.perf_counter() - start}
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf-8')
    if args.probe:
        start = time.perf_counter()
        value = engine.generate(input=str(model / 'example/asr_example.wav'), batch_size_s=300)
        assert len(value) == 1 and isinstance(value[0]['text'], str) and value[0]['text'].strip()
        result = dict(protocol, probe_completed=True, sample='Official Paraformer example',
                      raw_result=value, seconds=time.perf_counter() - start,
                      AuK_accuracy_score_computed=False)
        (args.output / 'probe.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
        return
    assert args.data and args.predictions and args.audio_root
    data_ready = load(args.data / 'READY.json')
    assert data_ready['count'] == 400 and data_ready['full_split']
    assert sha(args.data / 'samples.jsonl') == data_ready['samples_sha256'] == 'ea61e12ac36e6c1520a7c41ed608ebcbc7471933d47f5964b1e5dde02cc98701'
    samples = [json.loads(r) for r in (args.data / 'samples.jsonl').read_text(encoding='utf-8').splitlines()]
    full_samples = {s['id']: s for s in samples}
    assert sha(args.selected_samples) == 'd056bd4a0b0402ec7fa98887af0d303cda1d6cda7548e8017d31e58f6cc01a57'
    samples = [json.loads(r) for r in args.selected_samples.read_text(encoding='utf-8').splitlines()]
    assert len(samples) == 100 and all(s == full_samples[s['id']] for s in samples)
    predictions = [json.loads(r) for r in args.predictions.read_text(encoding='utf-8').splitlines()]
    targets = {s['id']: s for s in samples}
    assert len(predictions) == len({r['id'] for r in predictions}) == len(targets) == 100
    assert {r['id'] for r in predictions} == set(targets)
    audio_root = args.audio_root.resolve()
    rows = []
    with (args.output / 'cer.jsonl').open('x', encoding='utf-8') as log:
        for prediction in predictions:
            sample = targets[prediction['id']]
            path = audio_root / prediction['file']
            assert path.resolve().is_relative_to(audio_root) and not path.is_symlink() and sha(path) == prediction['sha256']
            with wave.open(str(path), 'rb') as wav:
                assert wav.getframerate() == 24000 and wav.getsampwidth() == 2 and wav.getnchannels() == 1
                assert wav.getnframes() > 0
            value = engine.generate(input=str(path), batch_size_s=300)
            assert len(value) == 1 and isinstance(value[0]['text'], str)
            hypothesis = convert(value[0]['text'], 'zh-cn')
            truth, text, cer, substitutions, deletions, insertions = process_one(hypothesis, sample['text'])
            row = {'id': sample['id'], 'audio_sha256': prediction['sha256'],
                   'reference': truth, 'raw_transcript': value[0]['text'], 'transcript_simplified': text,
                   'CER': cer, 'substitutions_rate': substitutions,
                   'deletions_rate': deletions, 'insertions_rate': insertions, 'raw_result': value}
            rows.append(row)
            log.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')
            log.flush()
            if len(rows) % 20 == 0:
                print(json.dumps({'CPU_CER_done': len(rows), 'target': 100}), flush=True)
    result = {'complete': len(rows) == 100, 'samples': len(rows), 'excluded': 0,
              'CER_percent': 100 * sum(r['CER'] for r in rows) / 100,
              'original_split_count': 400, 'subset_fraction': .25, 'numeric_pass': None,
              'note': 'Additional fixed100 root-reproduction subset only; full3x400 formal accuracy is preserved separately. Do not compare this length-stratified subset directly to the full-split publisher baseline.',
              'AuK_technical_delivery_complete': False, 'submitted': False}
    (args.output / 'cer-result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
