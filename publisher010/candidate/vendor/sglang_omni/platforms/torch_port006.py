# SPDX-License-Identifier: Apache-2.0
"""Native Torch device adapter restricted to AuK's single-device eager path."""
import random
import numpy as np
import torch
from sglang_omni.platforms.interface import OmniPlatform

class TorchPort006Platform(OmniPlatform):
    @property
    def device_type(self):
        return "cuda" if torch.cuda.is_available() else "cpu"
    def is_cpu(self): return self.device_type == "cpu"
    def is_cuda(self): return self.device_type == "cuda" and not self.is_rocm()
    def is_rocm(self): return self.device_type == "cuda" and bool(torch.version.hip)
    def is_cuda_alike(self): return self.device_type == "cuda"
    def is_npu(self): return False
    def is_xpu(self): return False
    def is_mps(self): return False
    def is_musa(self): return False
    def get_device(self, device_id=0):
        if self.is_cpu(): return torch.device("cpu")
        return torch.device("cuda", device_id)
    def set_device(self, device):
        if not self.is_cpu(): torch.cuda.set_device(device)
    def synchronize(self):
        if not self.is_cpu(): torch.cuda.synchronize()
    def empty_cache(self):
        if not self.is_cpu(): torch.cuda.empty_cache()
    def get_available_memory(self, device_id=0):
        if self.is_cpu(): raise NotImplementedError("CPU free memory is not a GPU budget")
        return torch.cuda.mem_get_info(device_id)
    def seed_everything(self, seed):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if not self.is_cpu(): torch.cuda.manual_seed_all(seed)
    def get_intra_node_transport(self):
        from sglang_omni.comm.data_ref import TransportKind
        return TransportKind.SHM
    def supports_torchaudio_resample(self):
        from sglang_omni.utils.audio import torchaudio
        return torchaudio is not None
    def enable_code2wav_graph(self): return False
    def enable_codec_decode_graph(self): return False
    def enable_talker_graph(self): return False
    def enable_tts_predictor_graph(self): return False
    def enable_tts_vocoder_fast_path(self): return False
    def enable_thinker_decode_graph(self): return False
    def supports_fp8_moe(self): return False
