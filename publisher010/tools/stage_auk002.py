"""Prepare AuK benchmark inputs and an isolated serving profile on the CPU."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request

BASE=Path('/root/private_data/guoxin-session010')
HERE=Path(__file__).resolve().parent


def identity(path, item):
    raw=path.read_bytes()
    digest=hashlib.sha256(raw).hexdigest() if item.get('lfs') else hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()
    return len(raw)==item['size'] and digest==(item['lfs']['oid'] if item.get('lfs') else item['oid'])


def main():
    data=BASE/'seedtts';data.mkdir(exist_ok=True)
    shutil.copy2(HERE/'seedtts-source-metadata.json',data/'source-metadata.json')
    subprocess.run([sys.executable,str(HERE/'prepare_seedtts002.py'),'--data',str(data),'--mirror'],check=True)
    model=BASE/'whisper-large-v3';model.mkdir(exist_ok=True)
    meta=json.loads((HERE/'whisper-source-metadata.json').read_text())
    assert meta['revision']=='06f233fe06e710322aca913c1bc4249a0d71fce1'
    def fetch(item):
        target=model/item['path']
        assert target.parent==model
        if target.exists():
            assert identity(target,item)
            return
        url=f"https://hf-mirror.com/{meta['repository']}/resolve/{meta['revision']}/{item['path']}"
        partial=target.with_name(target.name+'.partial')
        with urllib.request.urlopen(url,timeout=180) as source, partial.open('wb') as output:
            while block:=source.read(4*1024*1024):output.write(block)
        assert identity(partial,item)
        partial.rename(target)
        print(json.dumps({'verified':item['path'],'bytes':item['size']}),flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(fetch,meta['files']))
    shutil.copy2(HERE/'whisper-source-metadata.json',model/'SOURCE-IDENTITY.json')
    source=Path('/opt/guoxin-reviewed008/projects/guoxin-auk-bw-sgomni')
    candidates=list(source.rglob('candidate/runtime.py'))
    assert len(candidates)==1, candidates
    candidate=BASE/'auk-candidate'
    assert not candidate.exists(), 'Never overwrite a prepared candidate'
    shutil.copytree(candidates[0].parent,candidate,ignore=shutil.ignore_patterns('runtime','__pycache__'))
    path=candidate/'serve.py';old=path.read_text()
    text=old.replace('seconds=3','seconds=30').replace('0 < seconds <= 3','0 < seconds <= 30').replace('First pilot output must be <=3 seconds','AuK output cap must be <=30 seconds')
    text=text.replace('max_concurrency=1','max_concurrency=4').replace('max_batch_size=1','max_batch_size=4')
    text=text.replace('weight_dtype="bfloat16"','weight_dtype="float32"')
    assert text!=old and 'seconds=30' in text and 'weight_dtype="float32"' in text
    path.write_text(text)
    path=candidate/'runtime.py';text=path.read_text().replace('runtime<=3600','runtime<=10800')
    path.write_text(text)
    ready={'ready':True,'candidate':str(candidate),'dataset':str(data),'whisper':str(model),
           'data_count':json.loads((data/'READY.json').read_text())['count'],
           'changes':['30-second cap, reference-duration estimate unmodified','max batch/concurrency4',
                      'FP32 stored DiT weights with BF16 autocast, matching upstream recipe','owned runtime cap10800'],
           'serve_before_sha256':hashlib.sha256(old.encode()).hexdigest(),
           'serve_after_sha256':hashlib.sha256((candidate/'serve.py').read_bytes()).hexdigest()}
    (BASE/'AUK-READY.json').write_text(json.dumps(ready,indent=2)+'\n')
    print(json.dumps(ready),flush=True)


if __name__=='__main__':main()
