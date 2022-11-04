import os
import socket

import torch
from torch.utils.cpp_extension import load

from gb.util import FALLBACK_SRC_PATH


def get():
    global _module
    if _module is None:
        this_dir = os.path.dirname(__file__)
        cache_dir = os.path.join(this_dir, "__cudacache__", torch.__version__, socket.gethostname())
        os.makedirs(cache_dir, exist_ok=True)
        csrc_dir = os.path.join(this_dir, "csrc")
        if not os.path.exists(csrc_dir):
            # Fallback for runs via SEML on the GPU cluster.
            csrc_dir = f"{FALLBACK_SRC_PATH}/gb/kernels/csrc"
        _module = load(
            name="kernels",
            sources=[os.path.join(csrc_dir, "custom.cpp"), os.path.join(csrc_dir, "custom_kernel.cu")],
            extra_ldflags=['-lcusparse'],
            build_directory=cache_dir
        )
    return _module


_module = None
