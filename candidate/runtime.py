"""Use the unchanged tested owned supervisor for the real AuK-only Omni service."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import owned_supervisor as supervisor

ROOT=Path(__file__).resolve().parent
MODEL=Path("/root/private_data/guoxin-candidates/models/auk006")
ENCODER=Path("/root/private_data/guoxin-candidates/models/qwen25-omni3b006")

def verify_models(hash_weights=True):
    lock=json.loads((ROOT/"model-lock.json").read_text())
    directories={"tencent/AuK":MODEL,"Qwen/Qwen2.5-Omni-3B":ENCODER}
    records=[]
    for model in lock["models"]:
        base=directories[model["id"]]
        for item in model["files"]:
            if not item.get("required_for_conservative_snapshot"):continue
            rel=PurePosixPath(item["path"])
            if rel.is_absolute() or ".." in rel.parts:raise ValueError("Unsafe lock path")
            path=base.joinpath(*rel.parts)
            if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(base.resolve()):
                raise ValueError("Missing/unsafe locked checkpoint file: "+str(path))
            before=path.stat()
            if before.st_size!=item["bytes"]:raise ValueError("Checkpoint size mismatch: "+str(path))
            row={"model":model["id"],"revision":model["revision"],"path":str(path),"bytes":before.st_size}
            if hash_weights or item.get("role") not in {"AuK_DiT_with_layer_fusion","AuK_VAE","Qwen_full_mixed_checkpoint_shard"}:
                sha=hashlib.sha256();blob=hashlib.sha1(f"blob {before.st_size}\0".encode())
                with path.open("rb") as stream:
                    for chunk in iter(lambda:stream.read(4*1024*1024),b""):sha.update(chunk);blob.update(chunk)
                after=path.stat()
                if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError("Checkpoint changed during verification")
                actual=sha.hexdigest() if item.get("sha256") else blob.hexdigest()
                expected=item.get("sha256") or item.get("git_blob_sha1")
                if actual!=expected:raise ValueError("Checkpoint content mismatch: "+str(path))
                row.update(identity_verified=True,actual_hash=actual)
            else:row.update(identity_verified=False,reason="weight hashing omitted for diagnostic only")
            records.append(row)
    return {"files":records,"complete":all(r["identity_verified"] for r in records),"weights_loaded":False}

def ipc_runtime_path(run_id):
    if run_id is None: raise ValueError("Explicit owner run identity required")
    supervisor.checked_run_id(run_id)
    return Path("/tmp") / ("a006-" + run_id)

def configuration(run_id):
    ipc_runtime = ipc_runtime_path(run_id)
    runtime=int(os.environ.get("AUK_MAX_RUNTIME_SECONDS","1800"));load=int(os.environ.get("AUK_LOAD_TIMEOUT","900"))
    if not 60<=load<=runtime<=3600:raise ValueError("AuK service bounds must satisfy 60<=load<=runtime<=3600")
    port=int(os.environ.get("AUK_PORT","30139"))
    if not 1024<=port<=65535:raise ValueError("Invalid AuK port")
    command=[sys.executable,"-B",str(ROOT/"serve.py"),"--model-path",str(MODEL),"--encoder-path",str(ENCODER),
             "--runtime-dir",ipc_runtime.as_posix(),"--port",str(port)]
    return {"task":str(ROOT),"framework":"sglang-omni-AuK-native-Torch-port006","precision":"bf16",
            "model_id":"tencent/AuK","model_revision":"790742b71a4430120daf2b2099192abae449eb9f",
            "model_path":str(MODEL),"encoder_path":str(ENCODER),"host":"127.0.0.1","port":port,
            "url":f"http://127.0.0.1:{port}","load_timeout":load,"max_runtime":runtime,"stop_grace":45,
            "tp":1,"command":command,"official_accuracy_verified":False,"official_performance_verified":False}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("action",choices=["launch","stop","verify","plan"])
    ap.add_argument("--run-id");ap.add_argument("--dry-run",action="store_true");args=ap.parse_args()
    if args.action in {"launch","stop","plan"}:
        if not args.run_id:raise ValueError("Explicit 32-hex run identity required")
        supervisor.checked_run_id(args.run_id)
    if args.action=="stop":
        if args.dry_run:print(json.dumps({"plan_only":True,"process_signalled":False,"run_id":args.run_id}));return 0
        return supervisor.stop(ROOT,args.run_id)
    if args.action=="verify":print(json.dumps(verify_models(True),indent=2));return 0
    if not args.run_id:raise ValueError("Explicit 32-hex run identity required")
    config=configuration(args.run_id)
    if args.action=="plan" or args.dry_run:print(json.dumps(config,indent=2));return 0
    if sys.platform!="linux":raise ValueError("Actual launch requires SCNet Linux")
    if not Path("/dev/kfd").exists():raise ValueError("Native BW device is not allocated; refusing checkpoint loading in the CPU-only environment")
    verify=verify_models(True)
    runtime=ROOT/"runtime";runtime.mkdir(exist_ok=True)
    with (runtime/("checkpoint-"+args.run_id+".json")).open("x",encoding="utf-8") as f:json.dump(verify,f,indent=2)
    os.environ.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",HF_DATASETS_OFFLINE="1")
    original=supervisor.configuration;supervisor.configuration=lambda task:config
    try:return supervisor.launch(ROOT,False,args.run_id)
    finally:supervisor.configuration=original
if __name__=="__main__":raise SystemExit(main())
