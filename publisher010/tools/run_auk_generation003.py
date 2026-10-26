"""Bounded owned AuK generation experiment; independent of the Kimi instance."""
from pathlib import Path
import argparse
import shutil
import datetime as dt
import hashlib
import json
import os
import socket
import subprocess
import uuid
import zipfile

BASE=Path('/root/private_data/guoxin-session010')
PYTHON='/root/private_data/guoxin-auk-port006/venv002/bin/python'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--worker',choices=['A','B'],required=True);ap.add_argument('--runs',nargs='+',type=int,choices=[1,2,3],required=True);args=ap.parse_args()
    assert len(args.runs)==len(set(args.runs))
    assert Path('/dev/kfd').exists()
    assert json.loads((BASE/'AUK-READY.json').read_text())['ready']
    for port in (30000,30001,30002,30119,30129,30139):
        with socket.socket() as s:assert s.connect_ex(('127.0.0.1',port))!=0
    out=BASE/('auk-experiment003-'+args.worker);out.mkdir(exist_ok=False)
    candidate=BASE/('auk-candidate003-'+args.worker);assert not candidate.exists();shutil.copytree(BASE/'auk-candidate003',candidate,ignore=shutil.ignore_patterns('runtime','__pycache__'));owner=uuid.uuid4().hex
    env=os.environ.copy();env.update(AUK_MAX_RUNTIME_SECONDS='9000',AUK_LOAD_TIMEOUT='1200',
                                    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1')
    record={'started_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'run_id':owner,'worker':args.worker,'runs':args.runs,'phases':{},
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'run-start.json').write_text(json.dumps(record,indent=2)+'\n')
    def step(command,label,seconds):
        with (out/(label+'.log')).open('xb') as log:
            child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            try:code=child.wait(timeout=seconds)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                code=124
        record['phases'][label]=code;print(json.dumps({'phase':label,'exit':code}),flush=True)
        return code
    try:
        if step([PYTHON,'-c',"import torch,soundfile,numpy;assert torch.cuda.is_available();print(torch.__version__,torch.cuda.get_device_name(0))"],'environment',90):
            raise RuntimeError('AuK GPU/environment preflight failed')
        if step([PYTHON,str(candidate/'runtime.py'),'launch','--run-id',owner],'launch',1500):
            raise RuntimeError('AuK launch failed')
        if args.worker=='B' and step([PYTHON,'-u',str(BASE/'auk-tools/pilot_auk003.py')],'integration-pilot',600):
            raise RuntimeError('Batched reference-audio pilot failed')
        if step([PYTHON,'-u',str(BASE/'auk-tools/generate_auk003.py'),'--runs',*map(str,args.runs)],'generation',7800):
            raise RuntimeError('Full generation incomplete')
        record['generation_complete']=True
    except Exception as error:record['error']=str(error)
    finally:
        step([PYTHON,str(candidate/'runtime.py'),'stop','--run-id',owner],'stop',90)
        with socket.socket() as s:record['service_stopped']=s.connect_ex(('127.0.0.1',30139))!=0
        record.update(ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(),scoring_complete=False,submitted=False)
        (out/'run-final.json').write_text(json.dumps(record,indent=2)+'\n')
        with zipfile.ZipFile(f'/root/session010-auk-generation003-{args.worker}-results.zip','x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            for directory in (out,*[BASE/f'auk-generation003/run{i}' for i in args.runs],candidate/'runtime'):
                if directory.exists():
                    for p in directory.rglob('*'):
                        if p.is_file():z.write(p,p.relative_to(BASE).as_posix())
        print(json.dumps(record),flush=True)


if __name__=='__main__':main()
