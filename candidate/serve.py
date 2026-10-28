"""Launch the actual pinned SG-Omni AuK pipeline with a restricted native profile."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")

def checked_runtime_directory(value):
    # Check the lexical input before resolve() can hide a symlink or traversal.
    text = str(value)
    parsed = PurePosixPath(text)
    if str(parsed) != text or parsed.parent != PurePosixPath("/tmp") or not re.fullmatch(r"a006-[0-9a-f]{32}", parsed.name):
        raise ValueError("Runtime directory must be /tmp/a006-<exact 32-hex owner ID>")
    path = Path(text)
    if path.is_symlink() or path.exists():
        raise ValueError("Runtime directory already exists or is a symlink; refusing reuse")
    for parent in path.parents:
        if parent.is_symlink(): raise ValueError("Runtime parent must not be a symlink")
    return path

def claim_runtime_directory(path):
    # The final mkdir is exclusive: a racing existing target is never reused.
    if path.is_symlink() or path.exists(): raise ValueError("Runtime directory already exists")
    for parent in path.parents:
        if parent.is_symlink(): raise ValueError("Runtime parent must not be a symlink")
    path.mkdir(mode=0o700, exist_ok=False)
    return path

def configuration(model_path, encoder_path, runtime_dir, seconds=30):
    runtime_dir = checked_runtime_directory(str(runtime_dir).replace("\\", "/"))
    from sglang_omni.config import CommConfig, EndpointsConfig, FactoryArgs
    from sglang_omni.models.auk.config import AuKPipelineConfig
    from sglang_omni.platforms import current_platform
    if not 0 < seconds <= 100: raise ValueError("AuK output cap must be in (0,100] seconds")
    config = AuKPipelineConfig(model_path=str(Path(model_path).resolve()), name="tencent-auk-bw-port006")
    for stage in config.stages:
        values = stage.factory.model_dump(exclude_none=True)
        if stage.name == "preprocessing":
            values.update(max_concurrency=1, default_seconds=seconds, max_seconds=seconds)
        else:
            values.update(device=current_platform.device_type, max_batch_size=1)
        if stage.name == "conditioning":
            values.update(text_encoder_path=str(Path(encoder_path).resolve()), dtype="bfloat16")
        if stage.name == "auk_engine":
            values.update(nfe=32, cfg_strength=2.0, sway_sampling_coef=-1.0, max_seconds=seconds,
                          dtype="bfloat16", weight_dtype="float32", enable_dit_torch_compile=False,
                          enable_dit_cuda_graph=False, enable_dit_fused_qk_norm_rope=False)
        stage.factory = FactoryArgs(**values)
        stage.comm = CommConfig(slot_size_mb=16, credits=2)
    config.endpoints = EndpointsConfig(base_path=str(Path(runtime_dir).resolve() / "ipc"))
    config.mps = "off"
    config.weight_share = "off"
    if [s.name for s in config.stages] != ["preprocessing", "conditioning", "auk_engine", "decode"]:
        raise RuntimeError("Unexpected stage topology")
    if any(s.process != "pipeline" or s.tp_size != 1 for s in config.stages):
        raise RuntimeError("AuK port006 supports one native process on one device only")
    # Use the pinned framework's own pathname calculation before spawning workers.
    from sglang_omni.pipeline.runtime_config import (
        truncate_ipc_namespace_prefix, ipc_dir_len, longest_endpoint_suffix_len,
        _IPC_SUN_PATH_BUDGET,
    )
    ipc_base = Path(config.endpoints.base_path)
    prefix = truncate_ipc_namespace_prefix(config.name, base_root=ipc_base, stages=config.stages)
    length = ipc_dir_len(base_root=ipc_base, namespace_prefix_len=len(prefix),
                         endpoint_suffix_len=longest_endpoint_suffix_len(config.stages))
    if length > _IPC_SUN_PATH_BUDGET: raise ValueError("IPC pathname budget exceeded")
    return config

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--model-path",required=True);ap.add_argument("--encoder-path",required=True)
    ap.add_argument("--runtime-dir",required=True);ap.add_argument("--port",type=int,default=30139);ap.add_argument("--resolve",action="store_true")
    args=ap.parse_args()
    if not 1024 <= args.port <= 65535:raise ValueError("Invalid loopback port")
    runtime=checked_runtime_directory(args.runtime_dir)
    seconds=float(os.environ.get('AUK_MAX_SECONDS','30'))
    config=configuration(args.model_path,args.encoder_path,runtime,seconds=seconds)
    document={"scope":"SG-Omni AuK-only native Torch compatibility fork; no SRT model engines", "profile":config.model_dump(),
              "framework_revision":"f615735fd2a4d42daf80f2b81bd8eea4cf5a9b88","GPU_verified":False}
    if args.resolve:
        print(json.dumps(document,indent=2));return 0
    import torch
    if not torch.cuda.is_available(): raise RuntimeError("Actual pilot requires the allocated native BW device; CPU weight loading is disabled")
    claim_runtime_directory(runtime)
    with (runtime/"resolved-config.json").open("x",encoding="utf-8") as stream:json.dump(document,stream,indent=2)
    from sglang_omni.serve.launcher import launch_server
    launch_server(config,host="127.0.0.1",port=args.port,model_name="tencent-auk-bw-port006",
                  skip_server_warmup=True,enable_realtime=False,tts_batch_max_items=1)
    return 0
if __name__=="__main__":raise SystemExit(main())
