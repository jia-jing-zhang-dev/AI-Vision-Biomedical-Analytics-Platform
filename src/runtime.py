"""
runtime.py
==========
Resource settings for the Streamlit app, which serves every visitor from one
process on a small machine (Streamlit Community Cloud lists 690 MB-2.7 GB of
memory and up to 2 CPU cores, subject to change).

- ``heavy_work()``: a process-wide lock around memory-hungry steps (decoding
  uploads, segmentation, model inference). Concurrent visitors queue for a
  second instead of stacking their peak memory (two simultaneous 25 MP
  uploads reached ~1.16 GB without it).
- ``limit_torch_threads()``: PyTorch otherwise sizes its thread pools from
  the host's cores, not the container's CPU share.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager

_HEAVY = threading.Lock()
MAX_TORCH_THREADS = 2


@contextmanager
def heavy_work():
    """Serialize memory-hungry work across all sessions of the app process."""
    with _HEAVY:
        yield


def limit_torch_threads(n: int = MAX_TORCH_THREADS) -> None:
    """Call at the top of every page script: PyTorch's thread setting applies
    to the calling thread, and Streamlit runs each script run in its own."""
    import torch
    if torch.get_num_threads() > n:
        torch.set_num_threads(n)
