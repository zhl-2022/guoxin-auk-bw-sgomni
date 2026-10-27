"""Prepare a local CPU metric environment, leaving both BW experiments alone."""
from pathlib import Path
import hashlib
import importlib.metadata as metadata
import json
import subprocess
import sys
import venv


ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / 'external/auk-zh-eval019'
OUT = ROOT / 'evidence/session019'


def protected():
    value = {}
    for name in ('torch', 'numpy', 'scipy', 'transformers', 'modelscope'):
        dist = metadata.distribution(name)
        text = dist.read_text('METADATA')
        value[name] = {'version': dist.version, 'root': str(dist.locate_file('')),
                       'metadata_sha256': hashlib.sha256(text.encode()).hexdigest()}
    return value


def main():
    assert not TARGET.exists()
    OUT.mkdir(parents=True, exist_ok=True)
    before = protected()
    (OUT / 'zh-eval-protected-before.json').write_text(json.dumps(before, indent=2) + '\n')
    venv.EnvBuilder(system_site_packages=True, with_pip=True).create(TARGET)
    python = TARGET / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    constraints = OUT / 'zh-eval-protected-constraints.txt'
    constraints.write_text(''.join(name + '==' + row['version'] + '\n' for name, row in before.items() if name != 'numpy') + 'numpy==1.26.4\n')
    commands = [
        [str(python), '-m', 'pip', 'install', '--disable-pip-version-check', '--index-url', 'https://pypi.org/simple',
         '-c', str(constraints), 'numpy==1.26.4', 'funasr==1.4.16', 'jiwer==3.0.5', 'zhconv==1.4.3', 'zhon==2.1.1'],
        [str(python), '-m', 'pip', 'install', '--disable-pip-version-check', '--no-deps',
         '--index-url', 'https://download.pytorch.org/whl/cpu', 'torchaudio==2.10.0+cpu'],
    ]
    with (OUT / 'zh-eval-setup.log').open('xb') as log:
        for command in commands:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=900)
            assert result.returncode == 0, 'See task-local setup log; base environment unchanged'
    assert protected() == before
    freeze = subprocess.check_output([str(python), '-m', 'pip', 'freeze'], text=True)
    (OUT / 'zh-eval-freeze.txt').write_text(freeze, encoding='utf-8')
    result = {'environment': str(TARGET), 'python': str(python), 'protected_base_packages_unchanged': True,
              'metric_inference_device': 'cpu', 'BW_model_requests': 0, 'GPU_executed': False,
              'model_import_and_inference_verified': False}
    (OUT / 'zh-eval-setup.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
