from pathlib import Path
import hashlib,json,os,stat,sys,zipfile
archive=Path('/root/session019-auk-zhhard-queue001.zip')
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='7531ea92033be065b6a664dc3d5006e619e59aeb0568ff5b4e4edc42bd6df263'
stage=Path('/root/private_data/guoxin-session019/auk-zhhard-stage001');stage.mkdir(parents=True,exist_ok=False)
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 rows=json.loads(z.read('MANIFEST.json'))['files']
 assert len(z.namelist())==len(set(z.namelist()))==len(rows)+1
 assert set(z.namelist())=={r['path'] for r in rows}|{'MANIFEST.json'}
 for r in rows:
  p=stage/r['path'];assert not Path(r['path']).is_absolute() and p.resolve().is_relative_to(stage.resolve())
  assert not stat.S_ISLNK(z.getinfo(r['path']).external_attr>>16)
  raw=z.read(r['path']);assert len(raw)==r['bytes'] and hashlib.sha256(raw).hexdigest()==r['sha256']
  p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(raw)
(stage/'IMPORT-READY.json').write_text(json.dumps({'ready':True,'files':len(rows),'GPU_started':False}))
python='/root/private_data/guoxin-auk-port006/venv002/bin/python'
os.execv(python,[python,'-u',str(stage/'tools/run_auk_zhhard_queue019.py')])
