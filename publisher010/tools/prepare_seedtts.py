"""Extract the complete pinned English split used by the SGLang-Omni cookbook.

The actual Arrow split contains 1088 rows. Never truncate it to the rounded
1000-sample description in the SeedTTS README.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import re
import urllib.request
import wave


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--mirror', action='store_true')
    args = ap.parse_args()
    data = args.data.resolve()
    meta = json.loads((data / 'source-metadata.json').read_text())
    assert meta['revision'] == '81d1901582dee1293a537a6d945d084301712c41'
    item, = meta['files']
    parquet = data / 'en.parquet'
    if not parquet.exists():
        host = 'hf-mirror.com' if args.mirror else 'huggingface.co'
        url = f"https://{host}/datasets/{meta['repository']}/resolve/{meta['revision']}/{item['path']}"
        partial = data / 'en.parquet.partial'
        with urllib.request.urlopen(url, timeout=120) as source, partial.open('wb') as target:
            while block := source.read(1024 * 1024):
                target.write(block)
        assert partial.stat().st_size == item['size']
        assert sha(partial) == item['lfs']['oid']
        partial.rename(parquet)
    assert parquet.stat().st_size == item['size'] and sha(parquet) == item['lfs']['oid']
    import pyarrow.parquet as pq
    values = pq.ParquetFile(parquet).read().to_pylist()
    assert len(values) == len({r['sample_id'] for r in values}) == 1088
    audio_dir = data / 'reference'
    audio_dir.mkdir(exist_ok=True)
    rows = []
    for row in values:
        name = row['sample_id']
        assert re.fullmatch(r'[A-Za-z0-9_-]+', name)
        assert row['ref_text'].strip() and row['target_text'].strip()
        raw = row['ref_audio']['bytes']
        with wave.open(io.BytesIO(raw), 'rb') as source_audio:
            frames = source_audio.getnframes()
            sample_rate = source_audio.getframerate()
            assert frames > 0 and source_audio.getnchannels() == 1
        path = audio_dir / (name + '.wav')
        if path.exists():
            assert path.read_bytes() == raw
        else:
            path.write_bytes(raw)
        seconds = frames / sample_rate
        predicted_seconds = seconds * len(row['target_text'].encode()) / len(row['ref_text'].encode())
        rows.append({'id': name, 'ref_text': row['ref_text'], 'text': row['target_text'],
                     'audio': path.relative_to(data).as_posix(), 'sha256': hashlib.sha256(raw).hexdigest(),
                     'sample_rate': sample_rate, 'reference_seconds': seconds,
                     'estimated_output_seconds': predicted_seconds})
    samples = data / 'samples.jsonl'
    samples.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    ready = {'repository': meta['repository'], 'revision': meta['revision'], 'count': len(rows),
             'samples_sha256': sha(samples), 'full_split_no_subsetting': True,
             'total_estimated_seconds': sum(r['estimated_output_seconds'] for r in rows),
             'max_estimated_seconds': max(r['estimated_output_seconds'] for r in rows),
             'over_default_30s': sum(r['estimated_output_seconds'] > 30 for r in rows),
             'note': 'The pinned full Arrow English split has 1088 rows, despite nominal 1000 in original README.'}
    (data / 'READY.json').write_text(json.dumps(ready, indent=2) + '\n')
    print(json.dumps(ready), flush=True)


if __name__ == '__main__':
    main()
