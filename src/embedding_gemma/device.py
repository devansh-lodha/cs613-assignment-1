"""Device selection and numerical precision dispatch utilities."""

import torch


def get_optimal_device() -> torch.device:
    """Detect and select the fastest available hardware accelerator.

    Selection priority:
    1. NVIDIA CUDA GPU (torch.cuda)
    2. Apple Silicon GPU / Metal Performance Shaders (torch.backends.mps)
    3. CPU fallback

    Returns:
        torch.device configured for the best available compute backend.

    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_optimal_dtype(device: torch.device) -> torch.dtype:
    """Select the optimal floating-point precision for the given compute device.

    Numerical stability notes for Gemma architecture:
    - Gemma models exhibit large hidden activation magnitudes (~60,000 in deep layers).
    - Standard IEEE 754 float16 overflows at 65,504, leading to NaN activations.
    - bfloat16 shares the 8-bit dynamic exponent range of float32 (~10^38), preventing
      overflow while halving memory consumption compared to float32.

    Returns:
        torch.bfloat16 for CUDA (when supported) and Apple Silicon MPS.
        torch.float16 for legacy CUDA devices without bfloat16 support.
        torch.float32 for standard CPU execution.

    """
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.bfloat16
    return torch.float32
