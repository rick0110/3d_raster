from __future__ import annotations

import math as _math
import random as _random
from collections.abc import Sequence
from dataclasses import dataclass
import os
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

try:
    import numpy as _np
except ImportError:  # pragma: no cover - optional dependency
    _np = None


_DEFAULT_DEVICE = os.environ.get("RAYTRACER_DEVICE")


def set_default_device(device: str | None) -> None:
    global _DEFAULT_DEVICE
    if device in (None, "", "cpu"):
        _DEFAULT_DEVICE = None
    else:
        _DEFAULT_DEVICE = str(device)


def get_default_device() -> str | None:
    return _DEFAULT_DEVICE


def _torch_device():
    if torch is None:
        return None
    if _DEFAULT_DEVICE is None:
        return None
    return torch.device(_DEFAULT_DEVICE)


def _should_use_torch(value: Any | None = None) -> bool:
    if torch is None:
        return False
    if _DEFAULT_DEVICE is not None:
        return True
    return isinstance(value, torch.Tensor)


def _to_python(value: Any):
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().item()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_to_python(item) for item in value]
    if _np is not None and isinstance(value, _np.ndarray):
        return value.tolist()
    return value


def coerce_component(value: Any):
    if torch is not None and isinstance(value, torch.Tensor):
        device = _torch_device()
        if device is not None:
            return value.to(device=device, dtype=torch.float32)
        return value.to(dtype=torch.float32)

    if _should_use_torch():
        device = _torch_device()
        return torch.tensor(float(value), device=device, dtype=torch.float32)

    return float(value)


def to_float(value: Any) -> float:
    if torch is not None and isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def abs_value(value: Any):
    if torch is not None and isinstance(value, torch.Tensor):
        return torch.abs(value)
    return abs(value)


def max_value(*values: Any):
    if len(values) == 0:
        raise TypeError("max expected at least 1 argument, got 0")
    if len(values) == 1:
        return values[0]
    if torch is not None and any(_should_use_torch(value) for value in values):
        device = _torch_device()
        result = values[0] if isinstance(values[0], torch.Tensor) else torch.tensor(float(values[0]), device=device, dtype=torch.float32)
        for value in values[1:]:
            other = value if isinstance(value, torch.Tensor) else torch.tensor(float(value), device=device, dtype=torch.float32)
            result = torch.maximum(result, other)
        return result
    return max(values)


def min_value(*values: Any):
    if len(values) == 0:
        raise TypeError("min expected at least 1 argument, got 0")
    if len(values) == 1:
        return values[0]
    if torch is not None and any(_should_use_torch(value) for value in values):
        device = _torch_device()
        result = values[0] if isinstance(values[0], torch.Tensor) else torch.tensor(float(values[0]), device=device, dtype=torch.float32)
        for value in values[1:]:
            other = value if isinstance(value, torch.Tensor) else torch.tensor(float(value), device=device, dtype=torch.float32)
            result = torch.minimum(result, other)
        return result
    return min(values)


def clamp_value(value: Any, min_limit: float = 0.0, max_limit: float = 1.0):
    if torch is not None and isinstance(value, torch.Tensor):
        return torch.clamp(value, min_limit, max_limit)
    return max(min(value, max_limit), min_limit)


def sqrt_value(value: Any):
    if torch is not None and isinstance(value, torch.Tensor):
        return torch.sqrt(value)
    return _math.sqrt(value)


def floor_value(value: Any):
    if torch is not None and isinstance(value, torch.Tensor):
        return float(torch.floor(value).detach().cpu().item())
    return _math.floor(value)


def radians_value(value: Any):
    if torch is not None and isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item()) * _math.pi / 180.0
    return _math.radians(value)


def tan_value(value: Any):
    if torch is not None and isinstance(value, torch.Tensor):
        return float(torch.tan(value).detach().cpu().item())
    return _math.tan(value)


def random_value():
    if _should_use_torch():
        device = _torch_device()
        return float(torch.rand((), device=device, dtype=torch.float32).detach().cpu().item())
    return _random.random()


class _MathProxy:
    pi = _math.pi

    @staticmethod
    def sqrt(value):
        return sqrt_value(value)

    @staticmethod
    def floor(value):
        return floor_value(value)

    @staticmethod
    def radians(value):
        return radians_value(value)

    @staticmethod
    def tan(value):
        return tan_value(value)


class _RandomProxy:
    @staticmethod
    def random():
        return random_value()


@dataclass
class _LinalgProxy:
    def inv(self, matrix):
        if torch is not None and (_should_use_torch(matrix) or isinstance(matrix, torch.Tensor)):
            device = _torch_device()
            tensor = matrix if isinstance(matrix, torch.Tensor) else torch.tensor(_to_python(matrix), device=device, dtype=torch.float32)
            return torch.linalg.inv(tensor)
        if _np is None:
            raise RuntimeError("NumPy is required for matrix inversion on the CPU")
        return _np.linalg.inv(matrix)

    def norm(self, value):
        if torch is not None and (_should_use_torch(value) or isinstance(value, torch.Tensor)):
            device = _torch_device()
            tensor = value if isinstance(value, torch.Tensor) else torch.tensor(_to_python(value), device=device, dtype=torch.float32)
            return torch.linalg.norm(tensor)
        if _np is None:
            raise RuntimeError("NumPy is required for vector norms on the CPU")
        return _np.linalg.norm(value)


class _NumPyProxy:
    def __init__(self):
        self.linalg = _LinalgProxy()

    def array(self, value, dtype=float):
        if torch is not None and _should_use_torch(value):
            device = _torch_device()
            return torch.tensor(_to_python(value), device=device, dtype=torch.float32)
        if _np is None:
            raise RuntimeError("NumPy is required when PyTorch is not available")
        return _np.array(value, dtype=dtype)

    def linspace(self, start, stop, num):
        if torch is not None and (_should_use_torch(start) or _should_use_torch(stop)):
            device = _torch_device()
            return torch.linspace(to_float(start), to_float(stop), steps=num, device=device, dtype=torch.float32)
        if _np is None:
            raise RuntimeError("NumPy is required when PyTorch is not available")
        return _np.linspace(start, stop, num)

    def clip(self, value, min_value, max_value):
        if torch is not None and _should_use_torch(value):
            return torch.clamp(value, min_value, max_value)
        if _np is None:
            raise RuntimeError("NumPy is required when PyTorch is not available")
        return _np.clip(value, min_value, max_value)


math = _MathProxy()
random = _RandomProxy()
np = _NumPyProxy()