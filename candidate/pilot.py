"""Bounded one-request real AuK HTTP pilot with exact-owner cleanup and raw WAV."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import owned_supervisor as supervisor
from protocol import inspect_wav
from runtime import configuration, ipc_runtime_path

ROOT=Path(__file__).resolve().parent
PROTECTED=("torch","torchvision","vllm","vllm-hcu","transformers","triton","numpy","pydantic","tokenizers","safetensors")

def stamp():return datetime.now(timezone.utc).isoformat()
def digest(data):return hashlib.sha256(data).hexdigest()
def write(path,value):
    with path.open("x",encoding="utf-8") as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
def metadata():
    out={}
    for name in PROTECTED:
        try:
            d=importlib.metadata.distribution(name);raw=d.read_text("METADATA")
            out[name]={"version":d.version,"root":str(d.locate_file("")),"metadata_sha256":digest(raw.encode()) if raw else None}
        except importlib.metadata.PackageNotFoundError:out[name]=None
    return out
def sources():
    lock=json.loads((ROOT/"source-lock.json").read_text())
    expected={"vendor/"+r["path"]:r["sha256"] for r in lock["files"]}
    for name in ("runtime.py","serve.py","pilot.py","owned_supervisor.py","protocol.py","model-lock.json","source-lock.json","compatibility.patch"):
        expected[name]=digest((ROOT/name).read_bytes())
    for name,sha in expected.items():
        if digest((ROOT/name).read_bytes())!=sha:raise ValueError("Source-lock mismatch: "+name)
    return expected
def step(argv,path,seconds):
    start=time.monotonic();timeout=False
    with path.open("xb") as log:
        child=subprocess.Popen(argv,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=child.wait(timeout=seconds)
        except (subprocess.TimeoutExpired,KeyboardInterrupt) as exc:
            timeout=isinstance(exc,subprocess.TimeoutExpired)
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:child.wait(timeout=70)
            except subprocess.TimeoutExpired:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                child.wait()
            if not timeout:raise
            code=124
    return {"argv":argv,"exit_code":code,"timed_out":timeout,"child_reaped":True,"wall_seconds":time.monotonic()-start,"log_sha256":digest(path.read_bytes())}
def http(url,path,body=None,timeout=10):
    payload=json.dumps(body,ensure_ascii=False,separators=(",",":")).encode() if body is not None else None
    if payload is not None:(path.with_suffix(".request.json")).write_bytes(payload)
    req=urllib.request.Request(url,data=payload,headers={"Content-Type":"application/json"} if payload is not None else {},method="POST" if payload is not None else "GET")
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));start=time.monotonic()
    try:
        with opener.open(req,timeout=timeout) as response:
            data=response.read(20*1024*1024+1);status=response.status;headers=dict(response.headers)
    except urllib.error.HTTPError as response:
        data=response.read(20*1024*1024+1);status=response.code;headers=dict(response.headers)
    if len(data)>20*1024*1024:raise ValueError("HTTP body exceeds pilot bound")
    path.write_bytes(data)
    record={"url":url,"method":req.method,"status":status,"headers":headers,"body_bytes":len(data),"body_sha256":digest(data),"wall_seconds":time.monotonic()-start}
    write(path.with_suffix(path.suffix+".http.json"),record)
    return record,data

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--dry-run",action="store_true");args=ap.parse_args()
    run_id=secrets.token_hex(16);supervisor.checked_run_id(run_id);config=configuration(run_id)
    request={"model":"tencent-auk-bw-port006","input":"Welcome home.","instructions":"A clear, natural voice.",
             "stage_params":{"auk_engine":{"gen_seconds":3}},"seed":1234,"response_format":"wav"}
    if args.dry_run:
        print(json.dumps({"run_id":run_id,"configuration":config,"request":request,"executed":False},indent=2));return 0
    if sys.platform!="linux":raise ValueError("Actual pilot requires SCNet Linux")
    output=ROOT/"pilot-evidence"/run_id;output.mkdir(parents=True)
    before_sources=sources();before_metadata=metadata();write(output/"sources-before.json",before_sources);write(output/"native-before.json",before_metadata)
    record={"run_id":run_id,"started_utc":stamp(),"configuration":config,"scope":"one 3-second text-only development request; no official scoring", "official_accuracy_verified":False,"official_performance_verified":False,"submission_ready":False}
    write(output/"pilot-plan.json",record)
    old=signal.getsignal(signal.SIGTERM)
    def interrupted(number,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    try:
        record["start"]=step([sys.executable,"-B",str(ROOT/"runtime.py"),"launch","--run-id",run_id],output/"start.log",config["load_timeout"]+300)
        if record["start"]["exit_code"]:raise RuntimeError("AuK startup failed; original service log preserved")
        record["health"],_=http(config["url"]+"/health",output/"health.body")
        record["models"],_=http(config["url"]+"/v1/models",output/"models.body")
        record["speech"],body=http(config["url"]+"/v1/audio/speech",output/"speech.wav",request,timeout=600)
        if record["health"]["status"]!=200 or record["models"]["status"]!=200 or record["speech"]["status"]!=200:
            raise RuntimeError("HTTP status failure; raw body preserved")
        record["wav"]=inspect_wav(body,sample_rate=24000,channels=1,seconds=3)
        if record["wav"]["all_zero_samples"]:raise ValueError("Generated WAV is entirely zero")
        record["functional_wav_passed"]=True
    except BaseException as exc:
        record["error"]={"type":type(exc).__name__,"message":str(exc)}
        record["functional_wav_passed"]=False
    finally:
        signal.signal(signal.SIGTERM,old)
        try:
            record["stop"]=step([sys.executable,"-B",str(ROOT/"runtime.py"),"stop","--run-id",run_id],output/"stop.log",70)
        except BaseException as exc:
            record["stop"]={"exit_code":127,"error":repr(exc)}
        state_path=ROOT/"runtime/state.json"
        try:state=json.loads(state_path.read_text()) if state_path.is_file() else None
        except (ValueError,OSError) as exc:state=None;record["state_error"]=repr(exc)
        owned=state is not None and state.get("run_id")==run_id
        tokens={key:supervisor.process_token(state.get(key+"_pid",0)) for key in ("supervisor","server")} if owned else None
        with socket.socket() as sock:sock.settimeout(2);port_open=sock.connect_ex(("127.0.0.1",config["port"]))==0
        record["cleanup"]={"state_belongs_to_run":owned,"process_tokens":tokens,"port_open":port_open,
                            "verified":bool(owned and record["stop"]["exit_code"]==0 and all(v is None for v in tokens.values()) and not port_open)}
        if state is None and not record.get("state_error") and not (ROOT/"runtime"/("config-"+run_id+".json")).exists():
            record["cleanup"].update(verified=not port_open,reason="No launch config/state was created; no owner signal required")
        for name in ["state.json","config-"+run_id+".json","checkpoint-"+run_id+".json","server.log","supervisor.log"]:
            path=ROOT/"runtime"/name
            if path.is_file() and path.stat().st_size<=50*1024*1024:shutil.copyfile(path,output/name)
        resolved=ipc_runtime_path(run_id)/"resolved-config.json"
        if resolved.is_file():shutil.copyfile(resolved,output/"resolved-config.json")
        after_metadata=metadata()
        try:after_sources=sources()
        except (ValueError,OSError) as exc:after_sources={"source_verification_error":repr(exc)}
        write(output/"native-after.json",after_metadata);write(output/"sources-after.json",after_sources)
        record.update(ended_utc=stamp(),native_protected_metadata_unchanged=before_metadata==after_metadata,source_unchanged=before_sources==after_sources)
        record["passed"]=bool(record.get("functional_wav_passed") and record["cleanup"]["verified"] and record["native_protected_metadata_unchanged"] and record["source_unchanged"])
        write(output/"pilot-result.json",record)
        print(json.dumps({"output":str(output),"run_id":run_id,"passed":record["passed"],"cleanup":record["cleanup"]},indent=2))
    return 0 if record["passed"] else 1
if __name__=="__main__":raise SystemExit(main())
