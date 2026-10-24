"""Standard competition entrypoints around the unchanged frozen AuK candidate004."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent;CANDIDATE=ROOT/"candidate"
sys.path.insert(0,str(CANDIDATE))
import owned_supervisor
import runtime as candidate_runtime
from protocol import inspect_wav
from pilot import http

CORE={"omegaconf":"2.3.0","antlr4-python3-runtime":"4.9.3","x-transformers":"1.31.14","soundfile":"0.13.1","av":"14.2.0"}
PROTECTED=("torch","torchvision","vllm","vllm-hcu","transformers","triton","numpy","pydantic","tokenizers","safetensors")

def stamp():return datetime.now(timezone.utc).isoformat()
def sha(data):return hashlib.sha256(data).hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
def canonical_log(name,value):
    path=ROOT/name
    if path.is_symlink() or (path.exists() and not path.is_file()):raise ValueError("Unsafe canonical log path")
    if path.exists():
        prior=path.read_bytes();archive=ROOT/"runtime/log-archive"/(sha(prior)+"-"+name)
        archive.parent.mkdir(parents=True,exist_ok=True)
        if archive.exists():
            if archive.read_bytes()!=prior:raise ValueError("Archived raw-log identity differs")
        else:
            with archive.open("xb") as f:f.write(prior)
    data=(json.dumps(value,ensure_ascii=False,indent=2)+"\n").encode()
    temporary=path.with_name(path.name+".tmp-"+secrets.token_hex(8))
    with temporary.open("xb") as f:f.write(data)
    temporary.replace(path)
def source_binding():
    binding=json.loads((ROOT/"source-binding.json").read_text())
    for row in binding["files"]:
        p=ROOT/row["path"]
        if p.is_symlink() or not p.resolve().is_relative_to(CANDIDATE.resolve()):raise ValueError("Unsafe frozen candidate path")
        data=p.read_bytes()
        if len(data)!=row["bytes"] or sha(data)!=row["sha256"]:raise ValueError("Frozen candidate source changed: "+row["path"])
    return {"candidate004_zip_sha256":binding["candidate004_zip_sha256"],"files_verified":len(binding["files"])}
def distributions(executable,names):
    script="import importlib.metadata as m,json,sys; out={};\nfor n in json.loads(sys.argv[1]):\n try:\n  d=m.distribution(n); r=d.read_text('METADATA'); import hashlib; out[n]={'version':d.version,'root':str(d.locate_file('')),'metadata_sha256':hashlib.sha256(r.encode()).hexdigest() if r else None}\n except m.PackageNotFoundError:out[n]=None\nprint(json.dumps({'prefix':sys.prefix,'base_prefix':sys.base_prefix,'packages':out}))"
    result=subprocess.run([str(executable),"-B","-c",script,json.dumps(list(names))],capture_output=True,text=True,timeout=45)
    if result.returncode:raise RuntimeError("Metadata subprocess failed: "+result.stderr[-2000:])
    return json.loads(result.stdout)
def wheel_lock():
    lock=json.loads((ROOT/"cpu-wheel-lock.json").read_text())
    names={r["name"] for r in lock["files"]}
    allowed=set(CORE)|{"librosa","soxr","audioread","pooch","scikit-learn","threadpoolctl","lazy-loader"}
    if names!=allowed or len(lock["files"])!=len(allowed):raise ValueError("Unexpected CPU dependency set")
    for row in lock["files"]:
        p=ROOT/"cpu-wheels"/row["filename"]
        if p.is_symlink() or not p.resolve().is_relative_to((ROOT/"cpu-wheels").resolve()):raise ValueError("Unsafe CPU wheel path")
        if p.stat().st_size!=row["size"] or sha(p.read_bytes())!=row["sha256"]:raise ValueError("CPU wheel mismatch")
    return lock
def prepare(executable,dry_run=False):
    lock=wheel_lock();wanted=[r["name"]+"=="+r["version"] for r in lock["files"]]
    # Preserve the interpreter symlink: resolving venv/bin/python to the base
    # executable would silently discard the virtual environment at invocation.
    executable=Path(os.path.abspath(executable))
    default=Path(os.path.abspath(ROOT/"runtime/native-venv/bin/python"))
    if dry_run:return {"dry_run":True,"python":str(executable),"packages":wanted,"native_modified":False}
    if executable.exists():
        state=distributions(executable,CORE)
        if state["prefix"]==state["base_prefix"]:raise ValueError("A dedicated venv is required; refusing native installation")
        missing=[n for n,v in CORE.items() if not state["packages"].get(n) or state["packages"][n]["version"]!=v]
        if missing:raise ValueError("Existing venv lacks pinned dependencies; preserve it and select a new venv: "+repr(missing))
        return {"existing_venv_checked_readonly":True,"python":str(executable),"metadata":state}
    if sys.platform!="linux" or executable!=default:raise ValueError("Only a new task-local default Linux venv is prepared automatically")
    venv=executable.parent.parent
    if venv.exists() or venv.is_symlink():raise ValueError("Unmarked existing venv directory is preserved")
    before=distributions(sys.executable,PROTECTED)
    venv.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,"-B","-m","venv","--without-pip","--system-site-packages",str(venv)],check=True,timeout=90)
    state=distributions(executable,())
    if Path(state["prefix"]).resolve()!=venv.resolve() or state["prefix"]==state["base_prefix"]:raise ValueError("New interpreter prefix mismatch")
    env=os.environ.copy()
    for key in ("PIP_TARGET","PIP_PREFIX","PIP_USER"):env.pop(key,None)
    env["PIP_CONFIG_FILE"]=os.devnull
    command=[str(executable),"-B","-m","pip","install","--require-virtualenv","--disable-pip-version-check","--no-index","--no-deps","--find-links",str(ROOT/"cpu-wheels"),*wanted]
    with (ROOT/"runtime/dependency-install.log").open("x",encoding="utf-8") as log:
        process=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=240)
    after=distributions(sys.executable,PROTECTED)
    result={"python":str(executable),"command":command,"exit_code":process.returncode,"native_protected_metadata_unchanged":before==after,"native_before":before,"native_after":after}
    write(ROOT/"runtime/dependency-install-result.json",result)
    if process.returncode or before!=after:raise RuntimeError("Dependency preparation failed or native metadata changed; see runtime logs")
    write(venv/"auk-delivery006-ready.json",result)
    return result
def smoke(url,dry_run=False):
    request={"model":"tencent-auk-bw-port006","input":"Welcome home.","instructions":"A clear, natural voice.","stage_params":{"auk_engine":{"gen_seconds":3}},"seed":1234,"response_format":"wav"}
    if dry_run:return {"dry_run":True,"url":url,"request":request,"official_accuracy_verified":False}
    output=ROOT/"smoke-evidence"/secrets.token_hex(16);output.mkdir(parents=True)
    result={"started_utc":stamp(),"scope":"three-second development WAV only","official_accuracy_verified":False,"official_performance_verified":False}
    try:
        result["health"],_=http(url+"/health",output/"health.body")
        result["models"],_=http(url+"/v1/models",output/"models.body")
        result["speech"],body=http(url+"/v1/audio/speech",output/"speech.wav",request,timeout=600)
        if any(result[n]["status"]!=200 for n in ("health","models","speech")):raise ValueError("HTTP status failure")
        result["wav"]=inspect_wav(body,sample_rate=24000,channels=1,seconds=3)
        if result["wav"]["all_zero_samples"]:raise ValueError("WAV is entirely zero")
        result["passed"]=True
    except BaseException as exc:result.update(passed=False,error=repr(exc))
    result.update(ended_utc=stamp(),output=str(output));write(output/"result.json",result)
    return result
def environment(executable):
    result={"checked_utc":stamp(),"source":source_binding(),"image_full_reference":os.environ.get("IMAGE") or None,"image_candidate_name":os.environ.get("IMAGE_CANDIDATE_NAME") or None,"metadata":distributions(executable,(*PROTECTED,*CORE)),"framework":"SG-Omni AuK-only port of f615735fd2a4d42daf80f2b81bd8eea4cf5a9b88"}
    if sys.platform=="linux" and Path("/dev/kfd").exists():
        script="import torch,json; available=torch.cuda.is_available(); print(json.dumps({'torch':torch.__version__,'HIP':torch.version.hip,'gpu_available':available,'device':torch.cuda.get_device_name(0) if available else None,'count':torch.cuda.device_count()}))"
        p=subprocess.run([str(executable),"-B","-c",script],capture_output=True,text=True,timeout=90)
        result["device_probe"]={"exit_code":p.returncode,"stdout":p.stdout,"stderr":p.stderr}
        try:
            p=subprocess.run(["hy-smi"],capture_output=True,text=True,timeout=30);result["hy_smi"]={"exit_code":p.returncode,"stdout":p.stdout,"stderr":p.stderr}
        except OSError as exc:result["hy_smi"]={"error":repr(exc)}
    else:result["device_probe"]={"executed":False,"reason":"No allocated native BW device"}
    return result
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument("action",choices=["prepare","launch","stop","smoke","env","verify","bench"])
    ap.add_argument("--python",default=os.environ.get("PYTHON_BIN",str(ROOT/"runtime/native-venv/bin/python")))
    ap.add_argument("--run-id");ap.add_argument("--dry-run",action="store_true");ap.add_argument("--selfcheck",action="store_true");args=ap.parse_args(argv)
    source_binding()
    if args.action=="prepare":print(json.dumps(prepare(args.python,args.dry_run),indent=2));return 0
    if args.action in {"verify","bench"}:
        original=owned_supervisor.configuration;owned_supervisor.configuration=lambda task:candidate_runtime.configuration("a"*32)
        try:return owned_supervisor.official(ROOT,args.action,args.dry_run,args.selfcheck)
        finally:owned_supervisor.configuration=original
    if args.action=="env":
        result=environment(args.python);canonical_log("env.log",result);print(json.dumps(result,indent=2));return 0
    if args.action=="smoke":
        result=smoke(candidate_runtime.configuration("a"*32)["url"],args.dry_run)
        if not args.dry_run:canonical_log("smoke.log",result)
        print(json.dumps(result,indent=2));return 0 if args.dry_run or result["passed"] else 1
    if args.action=="stop" and not args.run_id:raise ValueError("stop requires the exact --run-id printed by run.sh")
    run_id=args.run_id or secrets.token_hex(16);owned_supervisor.checked_run_id(run_id)
    if args.action=="launch":
        if not args.dry_run:prepare(args.python)
        command=[str(args.python),"-B",str(CANDIDATE/"runtime.py"),"launch","--run-id",run_id]
        if args.dry_run:
            print(json.dumps({"run_id":run_id,"command":command,"configuration":candidate_runtime.configuration(run_id),"dry_run":True,"environment_modified":False},indent=2));return 0
    else:command=[str(args.python),"-B",str(CANDIDATE/"runtime.py"),"stop","--run-id",run_id]
    if args.dry_run:command.append("--dry-run")
    code=subprocess.run(command).returncode
    print(json.dumps({"action":args.action,"run_id":run_id,"exit_code":code,"official_accuracy_verified":False}))
    return code
if __name__=="__main__":raise SystemExit(main())
