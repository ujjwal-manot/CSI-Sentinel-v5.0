import random
import time
import numpy as np
import torch
from typing import Optional, Any, Tuple
from contextlib import contextmanager


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_device(device: Optional[str] = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Timer:
    __slots__ = ('start_time', 'elapsed')

    def __init__(self) -> None:
        self.start_time: Optional[float] = None
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        if self.start_time is not None:
            self.elapsed = time.perf_counter() - self.start_time

    def start(self) -> None:
        self.start_time = time.perf_counter()

    def stop(self) -> float:
        if self.start_time is None:
            return 0.0
        self.elapsed = time.perf_counter() - self.start_time
        return self.elapsed

    def reset(self) -> None:
        self.start_time = None
        self.elapsed = 0.0


class AverageMeter:
    __slots__ = ('val', 'avg', 'sum', 'count')

    def __init__(self) -> None:
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        if n <= 0:
            return
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    @property
    def total(self) -> float:
        return self.sum


class EarlyStopping:
    __slots__ = ('patience', 'min_delta', 'counter', 'best_score', 'should_stop')

    def __init__(self, patience: int = 10, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter: int = 0
        self.best_score: Optional[float] = None
        self.should_stop: bool = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if score < self.best_score - self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def reset(self) -> None:
        self.counter = 0
        self.best_score = None
        self.should_stop = False


def format_time(seconds: float) -> str:
    if seconds < 0:
        return "0s"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def safe_load_checkpoint(
    path: str,
    device: Optional[torch.device] = None,
    weights_only: bool = True
) -> dict:
    map_location = device if device else "cpu"
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=weights_only)
        return checkpoint
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {path}: {e}") from e


def validate_tensor_shape(
    tensor: torch.Tensor,
    expected_dims: int,
    name: str = "tensor"
) -> None:
    if tensor.dim() != expected_dims:
        raise ValueError(
            f"{name} has {tensor.dim()} dimensions, expected {expected_dims}"
        )


def ensure_batch_dim(tensor: torch.Tensor, expected_dims: int = 4) -> torch.Tensor:
    while tensor.dim() < expected_dims:
        tensor = tensor.unsqueeze(0)
    return tensor


@contextmanager
def torch_eval_mode(model: torch.nn.Module):
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        if was_training:
            model.train()
