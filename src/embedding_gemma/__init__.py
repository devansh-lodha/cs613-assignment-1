"""Modular framework for embedding inference and representation geometry analysis.

This package provides utilities for running text embedding models and analyzing
the isotropy of their hidden representations layer by layer.
"""

from embedding_gemma.config import ModelConfig
from embedding_gemma.data import load_bpcc_parallel
from embedding_gemma.device import get_optimal_device, get_optimal_dtype
from embedding_gemma.geometry import (
    LayerIsotropyRecord,
    analyze_layer_clouds,
    analyze_point_cloud,
    compute_avg_cosine_similarity,
    compute_id_score,
    compute_isoscore,
    compute_svd_ratio,
)
from embedding_gemma.model import EmbeddingGemmaWrapper, EmbeddingOutput
from embedding_gemma.utils import set_seed

__all__ = [
    "EmbeddingGemmaWrapper",
    "EmbeddingOutput",
    "LayerIsotropyRecord",
    "ModelConfig",
    "analyze_layer_clouds",
    "analyze_point_cloud",
    "compute_avg_cosine_similarity",
    "compute_id_score",
    "compute_isoscore",
    "compute_svd_ratio",
    "get_optimal_device",
    "get_optimal_dtype",
    "load_bpcc_parallel",
    "set_seed",
]
