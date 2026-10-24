# SPDX-License-Identifier: Apache-2.0
"""AuK-only port006 platform bootstrap using the installed PyTorch runtime.

This is a compatibility fork, not a claim that SRT's CUDA engines are available.
The AuK four-stage SimpleScheduler pipeline does not use an SRT model engine.
"""
import os
import pkgutil
from sglang_omni.platforms.interface import OmniPlatform
from sglang_omni.platforms.torch_port006 import TorchPort006Platform

def load_platform_class(qualname):
    cls = pkgutil.resolve_name(qualname)
    if not isinstance(cls, type) or not issubclass(cls, OmniPlatform):
        raise TypeError(f"Not an OmniPlatform: {qualname}")
    return cls

def get_platform_spec(platform):
    return f"{type(platform).__module__}.{type(platform).__qualname__}"

platform_spec = os.environ.get("SGLANG_OMNI_PLATFORM_SPEC")
current_platform = load_platform_class(platform_spec)() if platform_spec else TorchPort006Platform()
