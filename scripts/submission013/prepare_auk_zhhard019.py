"""Prepare every400 published SeedTTS Chinese hard case, without model calls."""
from pathlib import Path
import hashlib
import io
import json
import re
import wave


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'datasets/session019/seedtts-zhhard'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    import pyarrow.parquet as pq
    meta = json.loads((DATA / 'source-metadata.json').read_text())
    assert meta['revision'] == '81d1901582dee1293a537a6d945d084301712c41'
    parquet = DATA / 'raw/zh.parquet'
    assert parquet.stat().st_size == meta['file']['size']
    assert sha(parquet.read_bytes()) == meta['file']['lfs']['oid']
    hardcase = (DATA / 'raw/hardcase.lst').read_bytes()
    assert sha(hardcase) == meta['hardcase_sha256']
    values = pq.ParquetFile(parquet).read().to_pylist()
    assert len(values) == 2020
    references = {}
    for row in values:
        key = row['ref_audio_path']
        if key in references:
            previous = references[key]
            assert previous['ref_text'] == row['ref_text'] and previous['ref_audio']['bytes'] == row['ref_audio']['bytes']
        references[key] = row
    source_rows = [line.split('|') for line in hardcase.decode('utf-8').splitlines()]
    assert len(source_rows) == 400 and all(len(row) == 4 for row in source_rows)
    assert len({row[0] for row in source_rows}) == 400
    missing = [row[2] for row in source_rows if row[2] not in references]
    assert not missing, ('Do not silently exclude missing hardcase references', missing)
    out = DATA / 'reference'
    out.mkdir(exist_ok=False)
    samples = []
    for name, ref_text, audio_path, text in source_rows:
        assert re.fullmatch('[A-Za-z0-9_-]+', name) and text.strip() and ref_text.strip()
        value = references[audio_path]
        assert value['ref_text'] == ref_text
        audio = value['ref_audio']['bytes']
        with wave.open(io.BytesIO(audio), 'rb') as wav:
            assert wav.getnchannels() == 1 and wav.getsampwidth() == 2 and wav.getnframes() > 0
            rate, frames = wav.getframerate(), wav.getnframes()
        path = out / (name + '.wav')
        path.write_bytes(audio)
        seconds = frames / rate
        predicted = seconds * len(text.encode()) / len(ref_text.encode())
        samples.append({'id': name, 'ref_text': ref_text, 'text': text,
                        'audio': path.relative_to(DATA).as_posix(), 'sha256': sha(audio),
                        'source_ref_audio_path': audio_path, 'sample_rate': rate,
                        'reference_seconds': seconds, 'estimated_output_seconds': predicted})
    raw = ''.join(json.dumps(s, ensure_ascii=False) + '\n' for s in samples).encode()
    with (DATA / 'samples.jsonl').open('xb') as stream:
        stream.write(raw)
    ready = {'dataset': 'SeedTTS test-zh-hard', 'source_repack': meta['repository'],
             'revision': meta['revision'], 'full_split': True, 'count': 400,
             'selection_before_any_predictions': True, 'samples_sha256': sha(raw),
             'hardcase_sha256': sha(hardcase), 'reference_mapping_verified': 400,
             'publisher_paper': 'https://arxiv.org/html/2609.08936v1',
             'publisher_baseline_CER_percent': 5.91, 'CER_upper_relative_two_percent': 6.0282,
             'publisher_baseline_SIM': 0.782, 'SIM_lower_relative_two_percent': 0.76636,
             'publisher_aggregation': 'Three complete runs, mean; no prompt enhancement',
             'max_estimated_seconds': max(r['estimated_output_seconds'] for r in samples),
             'total_estimated_seconds_per_run': sum(r['estimated_output_seconds'] for r in samples),
             'over_default_30s': sum(r['estimated_output_seconds'] > 30 for r in samples),
             'original_Google_Drive_byte_identity_independently_verified': False,
             'GPU_executed': False, 'numeric_pass': None, 'submitted': False,
             'previous_English1088_WER_failure_retained': True}
    with (DATA / 'READY.json').open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(ready, indent=2) + '\n')
    print(json.dumps(ready, indent=2))


if __name__ == '__main__':
    main()
