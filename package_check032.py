"""Verify the exact sealed source and stored evaluation arrays without inference."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import zipfile

ROOT = Path(__file__).resolve().parent
MANIFEST = 'publisher032/project-manifest.json'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--allow-git', action='store_true')
    ap.add_argument('--arrays', action='store_true')
    args = ap.parse_args()
    manifest = load(ROOT / MANIFEST)['files']
    found = set()
    for folder, dirs, names in os.walk(ROOT, onerror=lambda error: (_ for _ in ()).throw(error)):
        if args.allow_git:
            dirs[:] = [n for n in dirs if n != '.git']
        for name in names:
            path = Path(folder) / name
            assert not path.is_symlink()
            found.add(path.relative_to(ROOT).as_posix())
    assert found == set(manifest) | {MANIFEST}, 'Source inventory differs from the sealed manifest'
    for rel, entry in manifest.items():
        raw = (ROOT / rel).read_bytes()
        assert len(raw) == entry['bytes'] and sha(raw) == entry['sha256'], rel
    if args.arrays:
        pub = ROOT / 'publisher032'
        means = []
        for iteration in (1, 2, 3):
            folder = pub / f'results/full1200/run{iteration}'
            generated, cer = rows(folder / 'generation-requests.jsonl'), rows(folder / 'cer.jsonl')
            assert len(generated) == len(cer) == len({r['id'] for r in cer}) == 400
            mapping = {r['id']: r for r in generated}
            assert set(mapping) == {r['id'] for r in cer}
            assert all(r['audio_sha256'] == mapping[r['id']]['sha256'] for r in cer)
            mean = 100 * sum(r['CER'] for r in cer) / 400
            assert math.isclose(mean, load(folder / 'cer-result.json')['CER_percent'], abs_tol=1e-12)
            means.append(mean)
        sim = rows(pub / 'results/full1200/sim.jsonl')
        assert len(sim) == len({(r['iteration'], r['id']) for r in sim}) == 1200
        summary = load(ROOT / 'verify_result.json')
        assert math.isclose(sum(means) / 3, summary['CER_percent_mean'], abs_tol=1e-12)
        assert math.isclose(sum(r['similarity'] for r in sim) / 1200, summary['SIM_mean'], abs_tol=1e-12)
        root = rows(pub / 'results/root100/cer.jsonl')
        assert len(root) == len({r['id'] for r in root}) == 100
        assert math.isclose(sum(r['CER'] for r in root), summary['root_subset_CER_percent'], abs_tol=1e-12)
        with zipfile.ZipFile(pub / 'originals/session030-auk-performance-A001.zip') as bundle:
            assert bundle.testzip() is None
            progress = json.loads(bundle.read('progress.json'))
            assert progress['performance_complete'] and progress['measured_requests'] == 96
            for item in json.loads(bundle.read('MANIFEST.json'))['files']:
                assert sha(bundle.read(item['path'])) == item['sha256']
    print(json.dumps({'sealed_source_verified': True, 'files_including_manifest': len(found),
                      'stored_arrays_checked': args.arrays, 'model_requests_executed': 0,
                      'official_acceptance': False, 'submitted': False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
