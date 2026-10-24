#!/usr/bin/env python3
"""AuK CPU preparation: no network, framework imports, model loading, or serving.

Builds distinct upstream Omni request contracts, inspects WAV containers, and
verifies previously downloaded checkpoint bytes against the explicit source lock.
These checks cannot establish GPU compatibility, speech quality, or acceptance.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import struct
import sys

MAX_AUDIO_BYTES = 20 * 1024 * 1024


def file_hashes(path: Path) -> tuple[int, str, str]:
    size = path.stat().st_size
    sha = hashlib.sha256()
    blob = hashlib.sha1(b"blob " + str(size).encode("ascii") + b"\0")
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            sha.update(chunk)
            blob.update(chunk)
    if path.stat().st_size != size:
        raise ValueError(f"file size changed while reading: {path}")
    return size, sha.hexdigest(), blob.hexdigest()


def inspect_wav(data: bytes, *, sample_rate: int | None = None,
                channels: int | None = None, seconds: float | None = None) -> dict:
    """Validate bounded RIFF PCM/IEEE-float audio without decoding with native libs."""
    if seconds is not None and (not math.isfinite(seconds) or seconds <= 0):
        raise ValueError("expected duration must be finite and positive")
    if len(data) > MAX_AUDIO_BYTES:
        raise ValueError("audio exceeds 20 MiB inspection bound")
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("expected RIFF/WAVE, not an HTTP error or other container")
    if struct.unpack_from("<I", data, 4)[0] + 8 != len(data):
        raise ValueError("RIFF length differs from archived byte length")
    offset, fmt, chunks = 12, None, []
    while offset < len(data):
        if offset + 8 > len(data):
            raise ValueError("truncated RIFF chunk header")
        name, size = struct.unpack_from("<4sI", data, offset)
        start, end = offset + 8, offset + 8 + size
        if end + (size & 1) > len(data):
            raise ValueError("truncated RIFF chunk payload/padding")
        payload = data[start:end]
        if name == b"fmt ":
            if fmt is not None or size < 16:
                raise ValueError("duplicate or short fmt chunk")
            code, count, rate, byte_rate, align, bits = struct.unpack_from("<HHIIHH", payload)
            if code == 65534:
                if size < 40 or struct.unpack_from("<H", payload, 16)[0] < 22:
                    raise ValueError("short WAVE_FORMAT_EXTENSIBLE")
                sub = payload[24:40]
                if sub[4:] != bytes.fromhex("00001000800000aa00389b71"):
                    raise ValueError("unsupported extensible subformat")
                code = struct.unpack_from("<I", sub)[0]
            if code not in (1, 3) or bits not in ((8, 16, 24, 32) if code == 1 else (32, 64)):
                raise ValueError("only PCM8/16/24/32 or IEEE float32/64 supported")
            if count <= 0 or rate <= 0 or align != count * bits // 8 or byte_rate != rate * align:
                raise ValueError("inconsistent WAV format/block alignment")
            fmt = (code, count, rate, align, bits)
        elif name == b"data":
            chunks.append(payload)
        offset = end + (size & 1)
    if fmt is None or not chunks:
        raise ValueError("missing format or data chunk")
    code, count, rate, align, bits = fmt
    if any(len(chunk) % align for chunk in chunks):
        raise ValueError("partial audio frame")
    audio = b"".join(chunks)
    frames = len(audio) // align
    if frames <= 0:
        raise ValueError("empty audio")
    if sample_rate is not None and rate != sample_rate:
        raise ValueError(f"sample rate {rate} != required {sample_rate}")
    if channels is not None and count != channels:
        raise ValueError(f"channels {count} != required {channels}")
    duration = frames / rate
    if seconds is not None and abs(duration - seconds) > 0.04:
        raise ValueError("duration differs by more than the local pilot tolerance (40 ms)")
    peak, squared, values = 0.0, 0.0, 0
    width = bits // 8
    for pos in range(0, len(audio), width):
        raw = audio[pos:pos + width]
        if code == 3:
            value = struct.unpack("<f" if bits == 32 else "<d", raw)[0]
        elif bits == 8:
            value = (raw[0] - 128) / 128
        else:
            value = int.from_bytes(raw, "little", signed=True) / (2 ** (bits - 1))
        if not math.isfinite(value):
            raise ValueError("nonfinite audio sample")
        peak = max(peak, abs(value))
        squared += value * value
        values += 1
    return {"container": "RIFF/WAVE", "format_code": code, "bits_per_sample": bits,
            "channels": count, "sample_rate": rate, "frames": frames,
            "duration_seconds": duration, "finite_samples": True,
            "peak_absolute": peak, "rms": math.sqrt(squared / values) if math.isfinite(squared) else None,
            "all_zero_samples": peak == 0.0, "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "scope": "audio_structure_only", "semantic_quality_verified": False,
            "GPU_verified": False, "official_accuracy_pass": False}


def make_request(*, framework: str, mode: str, text: str, description: str,
                 duration: float, seed: int, model: str,
                 reference: bytes | None = None) -> dict:
    if framework not in ("sglang-omni", "vllm-omni") or mode not in ("description", "clone", "edit"):
        raise ValueError("unsupported framework/mode")
    if not isinstance(text, str) or not text.strip() or len(text.encode("utf8")) > 16384:
        raise ValueError("nonempty text <=16 KiB required by this pilot tool")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or not 0 < duration <= 30:
        raise ValueError("local pilot duration must be finite and in (0,30] seconds")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("integer seed required")
    if mode != "description" and reference is None:
        raise ValueError("clone/edit requires actual reference WAV bytes")
    if mode == "description" and reference is not None:
        raise ValueError("description mode does not consume reference audio")
    ref_uri = None
    if reference is not None:
        inspect_wav(reference)
        ref_uri = "data:audio/wav;base64," + base64.b64encode(reference).decode("ascii")
    if framework == "sglang-omni":
        if mode == "edit":
            payload = {"prompt": text, "metadata": {"tts_params": {"ref_audio": ref_uri}},
                       "output_modalities": ["audio"], "return_logprob": False,
                       "stage_params": {"auk_engine": {"gen_seconds": duration}}, "seed": seed}
            endpoint, output = "/generate", "upstream_generate_json_audio_schema_unverified_locally"
        else:
            payload = {"input": text, "instructions": description,
                       "stage_params": {"auk_engine": {"gen_seconds": duration}},
                       "seed": seed, "response_format": "wav"}
            if ref_uri is not None:
                payload["ref_audio"] = ref_uri
            endpoint, output = "/v1/audio/speech", "RIFF/WAVE, 24000 Hz, mono"
    else:
        if not model:
            raise ValueError("vLLM-Omni requires the actual served model name/path")
        instruction = text if mode == "edit" else (
            f'Say the following with the same voice: "{text}"' if mode == "clone" else
            f'Generate speech based on the following description: "{description}". The content to speak is: "{text}".')
        payload = {"model": model, "input": "", "voice": "default", "response_format": "wav",
                   "instructions": instruction, "duration_seconds": duration, "seed": seed}
        if ref_uri is not None:
            payload["ref_audio"] = ref_uri
        endpoint, output = "/v1/audio/speech", "RIFF/WAVE, 24000 Hz, mono"
    return {"framework": framework, "endpoint": endpoint, "method": "POST",
            "content_type": "application/json", "payload": payload,
            "expected_output_contract": output, "request_sent": False,
            "GPU_loaded": False, "official_evaluated": False}


def safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or any(x in ("", ".", "..") for x in value.split("/")) or "\\" in value or ":" in value:
        raise ValueError(f"unsafe relative path: {value}")
    return path


def verify_records(records: list[dict], root: Path, path_key: str) -> dict:
    checked = []
    for entry in records:
        relative = safe_relative(entry[path_key])
        logical = root.joinpath(*relative.parts)
        row = {"path": str(logical), "exists": logical.is_file(), "matches": False}
        if row["exists"]:
            size, sha, blob = file_hashes(logical)
            row.update(bytes=size, sha256=sha, git_blob_sha1=blob, resolved_path=str(logical.resolve()))
            row["matches"] = size == entry["bytes"] and (
                (not entry.get("sha256") or sha == entry["sha256"]) and
                (not entry.get("git_blob_sha1") or blob == entry["git_blob_sha1"]))
        checked.append(row)
    return {"files": checked, "complete": bool(checked) and all(x["matches"] for x in checked),
            "weights_loaded": False, "GPU_verified": False, "official_evaluated": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    request = commands.add_parser("request")
    request.add_argument("--framework", choices=["sglang-omni", "vllm-omni"], required=True)
    request.add_argument("--mode", choices=["description", "clone", "edit"], required=True)
    request.add_argument("--text", required=True)
    request.add_argument("--description", default="A clear, natural voice.")
    request.add_argument("--duration-seconds", type=float, default=3)
    request.add_argument("--seed", type=int, default=1234)
    request.add_argument("--model", default="")
    request.add_argument("--reference-wav", type=Path)
    request.add_argument("--output", type=Path, required=True)
    wav = commands.add_parser("inspect-wav")
    wav.add_argument("path", type=Path)
    wav.add_argument("--sample-rate", type=int)
    wav.add_argument("--channels", type=int)
    wav.add_argument("--seconds", type=float)
    source = commands.add_parser("verify-sources")
    source.add_argument("--lock", type=Path, required=True)
    source.add_argument("--workspace", type=Path, default=Path.cwd())
    checkpoint = commands.add_parser("verify-checkpoint")
    checkpoint.add_argument("--lock", type=Path, required=True)
    checkpoint.add_argument("--model", choices=["tencent/AuK", "Qwen/Qwen2.5-Omni-3B"], required=True)
    checkpoint.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "request":
            ref = None
            if args.reference_wav:
                if args.reference_wav.stat().st_size > MAX_AUDIO_BYTES:
                    raise ValueError("reference exceeds inspection bound")
                ref = args.reference_wav.read_bytes()
            result = make_request(framework=args.framework, mode=args.mode, text=args.text,
                                  description=args.description, duration=args.duration_seconds,
                                  seed=args.seed, model=args.model, reference=ref)
            with args.output.open("x", encoding="utf8", newline="\n") as out:
                json.dump(result, out, ensure_ascii=False, indent=2, allow_nan=False)
                out.write("\n")
        elif args.command == "inspect-wav":
            if args.path.stat().st_size > MAX_AUDIO_BYTES:
                raise ValueError("audio exceeds inspection bound")
            result = inspect_wav(args.path.read_bytes(), sample_rate=args.sample_rate,
                                 channels=args.channels, seconds=args.seconds)
        else:
            lock = json.loads(args.lock.read_text(encoding="utf8"))
            if args.command == "verify-sources":
                result = verify_records(lock["captured_files"], args.workspace, "local_path")
            else:
                selected = next(m for m in lock["models"] if m["id"] == args.model)
                result = verify_records([e for e in selected["files"] if e["required_for_conservative_snapshot"]], args.directory, "path")
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if result.get("complete", True) else 2
    except (ValueError, OSError, KeyError, StopIteration) as exc:
        print(json.dumps({"error": str(exc), "executed_GPU": False}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
