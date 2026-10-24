"""CPU-only import and small real-network checks; never load checkpoints."""
from __future__ import annotations
import argparse
import ast
from datetime import datetime, timezone
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--functional", action="store_true")
    ap.add_argument("--launcher", action="store_true")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    result = {"checked_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
              "scope": "CPU compatibility only, no weights, no GPU and no formal evaluation",
              "checks": [], "distributions": {}}
    for name in ["torch", "torchaudio", "transformers", "numpy", "pydantic", "msgpack", "msgspec", "pyzmq", "pybase64", "omegaconf", "x-transformers", "einops", "soundfile", "scipy", "safetensors", "fastapi", "uvicorn"]:
        try: result["distributions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: result["distributions"][name] = None
    def check(name, fn):
        try:
            data = fn()
            result["checks"].append({"name": name, "passed": True, "detail": data})
        except Exception as exc:
            result["checks"].append({"name": name, "passed": False, "error": repr(exc), "traceback": traceback.format_exc()})
    def syntax():
        files = list((ROOT / "vendor").rglob("*.py"))
        for f in files: ast.parse(f.read_text(encoding="utf-8"), feature_version=(3, 10))
        return {"files": len(files), "python_grammar": "3.10"}
    check("python310_syntax", syntax)
    def stages():
        mod = importlib.import_module("sglang_omni.models.auk.stages")
        return {"file": mod.__file__, "factories": [getattr(mod, n).__name__ for n in ["create_preprocessing_executor", "create_conditioning_executor", "create_auk_engine_executor", "create_decode_executor"]]}
    check("real_upstream_four_stage_import", stages)
    def qwen():
        from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
        return [Qwen2_5OmniProcessor.__name__, Qwen2_5OmniThinkerForConditionalGeneration.__name__]
    check("qwen25_omni_classes_import", qwen)
    if args.launcher:
        def launcher():
            from sglang_omni.serve.launcher import launch_server
            return {"module": launch_server.__module__, "executed": False}
        check("real_upstream_http_launcher_import", launcher)
    if args.functional:
        def flow_forward():
            import torch
            from sglang_omni.models.auk.dit import AuKDit
            from sglang_omni.models.auk.flow_matching import AuKFlowMatching, AuKSampleItem
            if torch.cuda.is_available():
                # Deliberately construct all tensors/model on CPU regardless of host GPUs.
                pass
            torch.manual_seed(19)
            model = AuKDit(dim=32, heads=2, dim_head=16, num_layers=1, num_single_layers=1,
                           latent_dim=4, text_hidden_dim=8, dropout=0)
            flow = AuKFlowMatching(model, num_llm_layers=2).eval()
            conditioning = torch.randn(3, 8, device="cpu")
            item = AuKSampleItem(conditioning, torch.ones(3, dtype=torch.bool), 4, seed=5)
            with torch.inference_mode():
                out = flow.sample_batch([item], steps=2, cfg_strength=2, sway_sampling_coef=-1)[0]
                repeated = flow.sample_batch([item], steps=2, cfg_strength=2, sway_sampling_coef=-1)[0]
                ref_item = AuKSampleItem(conditioning, torch.ones(3, dtype=torch.bool), 4,
                                        ref_latent=torch.randn(2, 4), ref_length=2, seed=7)
                referenced = flow.sample_batch([ref_item], steps=2, cfg_strength=2, sway_sampling_coef=-1)[0]
            assert out.shape == (4, 4) and referenced.shape == (4, 4)
            assert torch.isfinite(out).all() and torch.isfinite(referenced).all()
            assert torch.equal(out, repeated), "same seeded CPU eager path must be deterministic"
            return {"device": str(out.device), "shape": list(out.shape), "finite": True,
                    "seeded_repeat_identical": True, "reference_path_finite": True,
                    "checkpoint_loaded": False, "small_random_network_only": True}
        check("real_AuK_DiT_Flow_CPU_forward", flow_forward)
        def state_roundtrip():
            import torch
            from sglang_omni.models.auk.payload_types import AuKState
            from sglang_omni.proto.request import StagePayload, OmniRequest
            from sglang_omni.scheduling.pipeline_state import load_state, store_state
            original = AuKState(instruction="test", conditioning=torch.arange(12).reshape(3, 4), gen_frames=150, seed=1234)
            payload = StagePayload("cpu-test", OmniRequest("test"), {})
            restored = load_state(store_state(payload, original), AuKState)
            assert torch.equal(restored.conditioning, original.conditioning)
            assert restored.gen_frames == 150 and restored.seed == 1234
            assert restored.conditioning.device.type == "cpu"
            return {"tensor_codec": "upstream tensor_cpu", "device": "cpu", "relay_used": False}
        check("upstream_cpu_state_codec", state_roundtrip)
    result["passed"] = all(row["passed"] for row in result["checks"])
    result["sglang_modules_actually_imported"] = sorted(n for n in sys.modules if n == "sglang" or n.startswith("sglang."))
    result["GPU_execution"] = False
    result["weights_loaded"] = False
    payload = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        if args.output.exists(): raise SystemExit("Refusing to overwrite probe output")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)
    return 0 if result["passed"] else 1

if __name__ == "__main__": raise SystemExit(main())
