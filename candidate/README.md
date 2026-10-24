# AuK native Torch compatibility candidate for BW

This candidate keeps the pinned SGLang-Omni AuK model and its four real stages, SimpleScheduler, Coordinator, multiprocess pipeline, and HTTP service. It replaces the small SRT-coupled platform bootstrap and OS / URL helpers, and permits the existing SoundFile / SciPy fallback when a native torchaudio build is absent. It does not provide SRT model engines or claim all-model Omni support.

Upstream revision: `f615735fd2a4d42daf80f2b81bd8eea4cf5a9b88`. See `source-lock.json`, `compatibility.patch`, `vendor/LICENSE`, and `vendor/sglang_omni/models/auk/LICENSE`. New work retains copyright; no new repository-wide license has been selected.

The model identities are fixed by `model-lock.json`:

- `tencent/AuK`, revision `790742b71a4430120daf2b2099192abae449eb9f`.
- `Qwen/Qwen2.5-Omni-3B`, revision `f75b40e3da2003cdd6e1829b1f420ca70797c34e`.

Do not install the upstream `pyproject.toml` into the native DAS environment. Use a new system-site-packages venv and install only missing CPU dependencies with `--no-deps`. The initial native inventory needs `omegaconf==2.3.0`, `antlr4-python3-runtime==4.9.3`, `x-transformers==1.31.14`, and `soundfile==0.13.1`. Reference audio additionally uses librosa and its CPU dependencies. Preserve native Torch, Transformers, vLLM and Triton.

CPU preparation checks are separate from device execution:

```bash
python -B probe.py --functional --output cpu-check.json
python -B tiny_thinker_probe.py --processor-path /root/private_data/guoxin-candidates/models/qwen25-omni3b006 --output tiny-check.json
python -B pilot.py --dry-run
```

Windows checks proved real small random CPU DiT / Flow and Thinker paths. Full Linux / DAS execution must be assessed from actual evidence; these checks do not load real checkpoints or prove official accuracy.

After the exact 20 checkpoint files are present, native compatibility is checked, and one BW device is allocated:

```bash
AUK_MAX_RUNTIME_SECONDS=1800 AUK_LOAD_TIMEOUT=900 \
timeout -s TERM -k 80s 2400s /absolute/isolated-venv/bin/python -B pilot.py
```

The actual pilot hashes the checkpoint files, uses the original owned supervisor, runs a single three-second text-only request, preserves raw HTTP / WAV / timing / source and native metadata, and stops the matching run in `finally`. Output is under `pilot-evidence/<32-hex-run-id>/`. The service is bound to `127.0.0.1:30139`, BF16, one device, concurrency one, NFE 32 / CFG 2 / sway -1, with compile / graph / fused RoPE disabled. CPU-only checkpoint loading is refused.

The final contest form is intentionally left to the participant. A development WAV with valid structure does not establish speech quality, official correctness, official performance, or acceptance.


## Candidate 004 runtime path correction

Candidate 003 reached the real native launcher but its nested IPC base exceeded Linux Unix-domain socket limits. Candidate 004 uses `/tmp/a006-<exact 32-hex owner ID>` and validates the path with the pinned framework's own endpoint-budget calculation before worker creation. Existing paths, symlink targets and symlink ancestors are refused; the final runtime-directory creation is exclusive. The supervisor logs and identity state remain under the candidate directory. The pilot copies the resolved config from the new short runtime path. All 313 vendor files, model files, compatibility patch and source lock remain byte-identical to candidate 003. A full checkpoint / HTTP result still requires the next actual BW run.
