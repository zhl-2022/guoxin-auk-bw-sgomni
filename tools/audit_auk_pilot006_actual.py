"""Audit downloaded pilot004 evidence without loading Torch or model checkpoints."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "a3e59176c8456dbbefd89b2189cb4165"
ZIP_SHA = "f087c025de08e7f764118d44f187ff7de78e9ca265539a89d353b18b95c0bdfb"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def audit(directory, *, source_archive=None, protocol_path=None):
    base = Path(directory).resolve()
    manifest = {}

    def raw(name):
        path = base / name
        if path.is_symlink() or not path.resolve().is_relative_to(base):
            raise ValueError("Unsafe evidence member: " + name)
        data = path.read_bytes()
        manifest[name] = {"bytes": len(data), "sha256": digest(data)}
        return data

    def read(name):
        return json.loads(raw(name))

    def require(condition, reason):
        if not condition:
            raise ValueError(reason)

    result = read("pilot-result.json")
    require(result["run_id"] == RUN_ID and result["passed"] is True, "Actual pilot identity/result mismatch")
    require(result["functional_wav_passed"] is True, "Pilot did not pass WAV diagnostics")
    for key in ("official_accuracy_verified", "official_performance_verified", "submission_ready"):
        require(result[key] is False, "Development pilot must not claim formal scoring/submission")
    require(result["start"]["exit_code"] == 0 and result["stop"]["exit_code"] == 0, "Start/stop did not succeed")
    for key, filename in (("health", "health.body"), ("models", "models.body"), ("speech", "speech.wav")):
        data = raw(filename)
        item = read(filename + ".http.json")
        require(item == result[key], "HTTP record differs from pilot result: " + key)
        require(item["status"] == 200 and item["body_sha256"] == digest(data) and item["body_bytes"] == len(data), "HTTP body/status identity failure: " + key)
    require(any(row["id"] == "tencent-auk-bw-port006" for row in json.loads(raw("models.body"))["data"]), "Actual model list does not expose the reviewed AuK service")
    request = read("speech.request.json")
    require(request["model"] == "tencent-auk-bw-port006" and request["input"] == "Welcome home." and request["instructions"] == "A clear, natural voice." and request["seed"] == 1234 and request["stage_params"]["auk_engine"]["gen_seconds"] == 3, "Pilot request differs from reviewed one")

    protocol_path = protocol_path or ROOT / "tasks/auk-bw-sgomni-port006-004/protocol.py"
    spec = importlib.util.spec_from_file_location("auk_actual_protocol", protocol_path)
    protocol = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = protocol
    spec.loader.exec_module(protocol)
    wav = protocol.inspect_wav(raw("speech.wav"), sample_rate=24000, channels=1, seconds=3)
    require(wav == result["wav"] and not wav["all_zero_samples"], "WAV reinspection differs from raw result")

    before = read("sources-before.json")
    after = read("sources-after.json")
    require(before == after and result["source_unchanged"] is True, "Sources changed during actual run")
    archive = source_archive or ROOT / "dist/session006-auk-port-cpu-004.zip"
    require(digest(archive.read_bytes()) == ZIP_SHA, "Local immutable source ZIP identity differs")
    with zipfile.ZipFile(archive) as frozen:
        require(frozen.testzip() is None, "Source archive CRC failure")
        lock = json.loads(frozen.read("source-lock.json"))
        expected = {"vendor/" + r["path"] for r in lock["files"]}
        expected.update(("runtime.py", "serve.py", "pilot.py", "owned_supervisor.py", "protocol.py", "model-lock.json", "source-lock.json", "compatibility.patch"))
        require(set(before) == expected, "Actual source snapshot membership differs")
        for name, sha in before.items():
            require(digest(frozen.read(name)) == sha, "Actual executed source differs from frozen004: " + name)
        models = json.loads(frozen.read("model-lock.json"))

    native_before = read("native-before.json")
    native_after = read("native-after.json")
    require(native_before == native_after and result["native_protected_metadata_unchanged"] is True, "Protected native metadata changed")
    require(native_before["transformers"]["version"] == "5.5.0", "Native Transformers differs from expected validated stack")
    checkpoint = read("checkpoint-" + RUN_ID + ".json")
    require(checkpoint["complete"] is True and len(checkpoint["files"]) == 20, "Incomplete checkpoint identity evidence")
    expected_models = {(m["id"], f["path"]): (m["revision"], f["bytes"], f.get("sha256") or f.get("git_blob_sha1")) for m in models["models"] for f in m["files"] if f.get("required_for_conservative_snapshot")}
    actual_keys = set()
    for row in checkpoint["files"]:
        key = row["model"], PurePosixPath(row["path"]).name
        actual_keys.add(key)
        require((row["revision"], row["bytes"], row["actual_hash"]) == expected_models[key] and row["identity_verified"] is True, "Checkpoint identity does not match fixed model lock")
    require(actual_keys == set(expected_models), "Checkpoint membership differs from full conservative snapshot")

    config = read("resolved-config.json")
    require(config["framework_revision"] == "f615735fd2a4d42daf80f2b81bd8eea4cf5a9b88", "Framework revision differs")
    profile = config["profile"]
    require(profile["endpoints"]["base_path"] == "/tmp/a006-" + RUN_ID + "/ipc", "Unexpected actual IPC path")
    stages = profile["stages"]
    require([s["name"] for s in stages] == ["preprocessing", "conditioning", "auk_engine", "decode"], "Actual four-stage topology differs")
    require(all(s["process"] == "pipeline" and s["tp_size"] == 1 for s in stages), "Actual process/TP differs")
    require(all(s["factory"]["device"] == "cuda" for s in stages if s["name"] != "preprocessing"), "Actual stage devices were not native CUDA-compatible backend")
    factories = {s["name"]: s["factory"] for s in stages}
    expected_engine = {"nfe": 32, "cfg_strength": 2.0, "sway_sampling_coef": -1.0,
                       "dtype": "bfloat16", "weight_dtype": "bfloat16", "max_seconds": 3,
                       "max_batch_size": 1, "enable_dit_torch_compile": False,
                       "enable_dit_cuda_graph": False, "enable_dit_fused_qk_norm_rope": False}
    require(all(factories["auk_engine"][key] == value for key, value in expected_engine.items()), "Actual AuK sampling/precision profile differs")
    require(profile["mps"] == "off" and profile["weight_share"] == "off", "Actual optional platform mechanisms differ")
    require(factories["conditioning"]["text_encoder_path"] == "/root/private_data/guoxin-candidates/models/qwen25-omni3b006", "Actual encoder path differs from fixed checkpoint")

    cleanup = result["cleanup"]
    require(cleanup["verified"] is True and cleanup["state_belongs_to_run"] is True and cleanup["port_open"] is False and cleanup["process_tokens"] == {"server": None, "supervisor": None}, "Exact-owner cleanup evidence incomplete")
    state = read("state.json")
    require(state["run_id"] == RUN_ID, "Final state belongs to another owner")
    launch_config = read("config-" + RUN_ID + ".json")
    require(launch_config["run_id"] == RUN_ID and
            {key: value for key, value in launch_config.items() if key != "run_id"} == result["configuration"],
            "Supervised launch configuration differs from actual pilot configuration")
    require(state["command"] == launch_config["command"] and state["url"] == launch_config["url"],
            "Final supervisor state is not bound to actual command/endpoint")
    plan = read("pilot-plan.json")
    require(plan["run_id"] == RUN_ID and plan["configuration"] == result["configuration"],
            "Initial pilot plan differs from actual result configuration")
    server_log = raw("server.log").decode("utf-8", errors="replace")
    require("=== run " + RUN_ID + " at " in server_log and
            json.dumps(launch_config["command"]) in server_log,
            "Raw server log is not bound to actual owner and command")
    require("1442/1442" in server_log and "hip/cutlassfa_adapter.h" in server_log,
            "Raw server log lacks observed checkpoint load or DAS HIP attention execution")
    require('"POST /v1/audio/speech HTTP/1.1" 200 OK' in server_log and
            "Application shutdown complete." in server_log,
            "Raw server log lacks actual generation response or application shutdown")
    runtime_observations = {
        "checkpoint_load_1442_of_1442": True,
        "DAS_HIP_SDPA_kernel_path_observed": True,
        "recorded_stage_device": "cuda (native DAS Torch backend)",
        "fresh_device_probe_in_this_local_audit": False,
        "application_shutdown_complete": True,
        "resource_tracker_leaked_semaphore_warning": "1 leaked semaphore objects" in server_log,
        "mooncake_and_nixl_missing_warnings": "Failed to import mooncake" in server_log and "Failed to import nixl" in server_log,
        "conditioner_system_prompt_warning": "System prompt modified" in server_log,
    }
    for name in ("supervisor.log", "start.log", "stop.log"):
        raw(name)
    require(manifest["start.log"]["sha256"] == result["start"]["log_sha256"] and manifest["stop.log"]["sha256"] == result["stop"]["log_sha256"], "Start/stop raw logs differ from result identities")
    return {"scope": "Downloaded pilot004 source/checkpoint identity, raw HTTP/WAV, metadata and recorded cleanup audit; no fresh remote-process/device probe and no official scoring", "run_id": RUN_ID, "pilot_directory": str(base), "passed": True, "source_files_bound_to_frozen004": len(before), "checkpoint_files_verified_against_original_lock": len(actual_keys), "actual_stage_devices": {s["name"]: s["factory"].get("device") for s in stages}, "server_log_owner_and_command_bound": True, "runtime_observations": runtime_observations, "native_metadata": native_before, "wav": wav, "configuration": result["configuration"], "timings": {"launch_step_wall_seconds": result["start"]["wall_seconds"], "speech_http_wall_seconds": result["speech"]["wall_seconds"], "raw_speech_headers": result["speech"]["headers"]}, "cleanup": cleanup, "raw_files": manifest, "official_accuracy_verified": False, "official_performance_verified": False, "official_acceptance": False, "submitted": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--protocol", type=Path)
    args = parser.parse_args()
    result = audit(args.pilot_dir, source_archive=args.source_archive, protocol_path=args.protocol)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "run_id": RUN_ID, "passed": True, "raw_files_verified": len(result["raw_files"])}))


if __name__ == "__main__":
    main()
