"""Modular framework for EmbeddingGemma inference and geometric representation.

This package provides utilities for running EmbeddingGemma models and extracting
representations across layers.
"""

from embedding_gemma.config import ModelConfig
from embedding_gemma.data import load_stsb_benchmark
from embedding_gemma.device import get_optimal_device, get_optimal_dtype
from embedding_gemma.geometry import (
    LayerGeometryRecord,
    analyze_layer_geometry,
    compute_cosine_anisotropy,
    compute_isoscore,
    compute_rogue_dimension_ratio,
    compute_spearman_correlation,
)
from embedding_gemma.model import EmbeddingGemmaWrapper, EmbeddingOutput
from embedding_gemma.utils import set_seed

__all__ = [
    "EmbeddingGemmaWrapper",
    "EmbeddingOutput",
    "LayerGeometryRecord",
    "ModelConfig",
    "analyze_layer_geometry",
    "compute_cosine_anisotropy",
    "compute_isoscore",
    "compute_rogue_dimension_ratio",
    "compute_spearman_correlation",
    "get_optimal_device",
    "get_optimal_dtype",
    "load_stsb_benchmark",
    "set_seed",
]
