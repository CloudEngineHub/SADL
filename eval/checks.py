"""Provenance records for an evaluation run: numerics flags, packages, GPU, evaluator file and imported sources."""
from __future__ import annotations

import hashlib

import json

import os

import re

import socket

import subprocess

import sys

import time

from pathlib import Path

RULE = "lewm-goal25-budget50"

EVALUATOR_WORLD_SHA256 = "39318b81ed151d8556d8540f460a63eedaed0ce4b2211ce0af9f8200e7d83bde"

ENV_PREFIXES = ("INVERSE_", "INTACT_", "MUJOCO_", "PYOPENGL_", "CUDA_VISIBLE_DEVICES", "CUBLAS_", "OMP_NUM",
                "MKL_NUM", "OPENBLAS_NUM", "NUMEXPR_NUM", "PYTHONHASHSEED", "EVAL_",
                "STABLEWM_HOME", "LD_LIBRARY_PATH", "LD_PRELOAD", "TORCH_", "NVIDIA_TF32", "PYTHONPATH")

def sha256_file(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

def numerics_snapshot() -> dict:
    """Effective numerics flags (recorded; nothing is set here)."""
    import torch
    cuda = torch.backends.cuda
    rec = {
        "matmul_allow_tf32": bool(cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "flash_sdp_enabled": bool(cuda.flash_sdp_enabled()),
        "mem_efficient_sdp_enabled": bool(cuda.mem_efficient_sdp_enabled()),
        "math_sdp_enabled": bool(cuda.math_sdp_enabled()),
    }
    if hasattr(cuda, "cudnn_sdp_enabled"):
        rec["cudnn_sdp_enabled"] = bool(cuda.cudnn_sdp_enabled())
    rec["CUBLAS_WORKSPACE_CONFIG"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    return rec

def _packages() -> dict:
    from importlib import metadata
    out = {}
    for d in metadata.distributions():
        name = (d.metadata.get("Name") or "").lower().replace("_", "-")
        if name:
            out[name] = d.version
    return dict(sorted(out.items()))

def env_record() -> dict:
    return {k: v for k, v in sorted(os.environ.items()) if k.startswith(ENV_PREFIXES)}

def process_snapshot() -> dict:
    """Software/GPU identity of the evaluation process."""
    import torch
    rec = {
        "python": sys.version.split()[0], "python_executable": sys.executable,
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "packages": _packages(), "env": env_record(),
        "hostname": socket.gethostname(), "sitecustomize": None,
    }
    site_mod = sys.modules.get("sitecustomize")
    if site_mod is not None and getattr(site_mod, "__file__", None):
        rec["sitecustomize"] = os.path.realpath(site_mod.__file__)
    if torch.cuda.is_available():
        rec["gpu_name"] = torch.cuda.get_device_name(0)
        rec["gpu_capability"] = list(torch.cuda.get_device_capability(0))
    else:
        rec["gpu_name"] = None
    try:
        import stable_worldmodel.world.world as w
        rec["world_py"] = os.path.realpath(w.__file__)
        rec["world_py_sha256"] = sha256_file(w.__file__)
    except Exception as exc:  # noqa: BLE001
        rec["world_py_error"] = f"{type(exc).__name__}: {exc}"
    return rec

def imported_sources(roots) -> dict:
    """sha256 of every imported module file located under one of `roots` (realpaths)."""
    roots = [os.path.realpath(str(r)) for r in roots if r]
    out = {}
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        f = os.path.realpath(f)
        if f.endswith((".pyc", ".pyo")):
            continue
        if any(f == r or f.startswith(r + os.sep) for r in roots) and os.path.isfile(f):
            out[f] = sha256_file(f)
    return dict(sorted(out.items()))

class CostCallCounter:
    """Count model.get_cost calls and candidate action steps (Direct identity check).

    Installed on the model instance before the solver is built; numerically transparent.
    """

    def __init__(self, model):
        self.calls = 0
        self.candidate_sequences = 0
        self.candidate_action_steps = 0
        self.installed = False
        original = getattr(model, "get_cost", None)
        if original is None:
            return
        counter = self

        def get_cost(info_dict, action_candidates, *a, **k):
            counter.calls += 1
            shape = getattr(action_candidates, "shape", ())
            if len(shape) >= 3:
                b, n, h = shape[:3]
                counter.candidate_sequences += int(b * n)
                counter.candidate_action_steps += int(b * n * h)
            return original(info_dict, action_candidates, *a, **k)

        object.__setattr__(model, "get_cost", get_cost)
        self.installed = True

    def record(self) -> dict:
        return {"get_cost_calls": self.calls, "candidate_sequences": self.candidate_sequences,
                "candidate_action_steps": self.candidate_action_steps, "installed": self.installed}

