# SPDX-License-Identifier: Apache-2.0
"""Small OS/network helpers so the AuK-only fork need not import SRT engines."""
from dataclasses import dataclass
import ctypes
import os
import signal
import sys

@dataclass(frozen=True)
class NetworkAddress:
    host: str
    port: int
    def to_url(self):
        host = f"[{self.host}]" if ":" in self.host and not self.host.startswith("[") else self.host
        return f"http://{host}:{self.port}"

def kill_itself_when_parent_died():
    # Linux parent-death signal, with the documented setup race checked.
    # Windows imports are supported for CPU unit checks; production target is Linux.
    if sys.platform != "linux": return
    parent = os.getppid()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
    if os.getppid() != parent or parent == 1:
        os.kill(os.getpid(), signal.SIGTERM)
