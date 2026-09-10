"""Configuration models and type definitions for EmbeddingGemma inference."""

from dataclasses import dataclass
from typing import Literal

# Supported task specializations for EmbeddingGemma asymmetric prompt templates.
TaskType = Literal["query", "document", "raw"]


@dataclass(frozen=True)
class ModelConfig:
    """Configuration parameters for EmbeddingGemma model initialization and inference.

    EmbeddingGemma is an asymmetric dual-encoder text embedding model based on
    the Gemma architecture. It supports task-specific instruction prefixes for
    search queries versus target corpus documents.

    Attributes:
        model_id: Hugging Face Hub repository identifier.
        max_length: Maximum sequence length (in tokens) before truncation occurs.
        batch_size: Number of texts evaluated in a single forward pass.
        normalize_embeddings: Whether to project final vectors to the unit hypersphere
            (L2 norm = 1.0), enabling dot-product cosine similarity.
        task_type: Operating mode ("query", "document", or "raw").
        query_template: Prompt template applied when task_type is "query".
        document_template: Prompt template applied when task_type is "document".
        task_description: High-level intent string injected into the query template.
        document_title: Passage title metadata injected into the document template.
        attn_implementation: Attention backend ("sdpa" or "eager").
        seed: Deterministic random seed for reproducibility across backends.

    """

    model_id: str = "google/embeddinggemma-300m"
    max_length: int = 2048
    batch_size: int = 32
    normalize_embeddings: bool = True
    # [LEGACY FLAG]: Defaults to False; not used by collect_layer_sentence_embeddings.
    normalize_layers: bool = False
    task_type: TaskType = "raw"
    # [AUXILIARY / RETRIEVAL]: Used for instruction-tuned retrieval benchmarks.
    query_template: str = "task: {description} | query: {text}"
    document_template: str = "title: {title} | text: {text}"
    task_description: str = "search result"
    document_title: str = "none"
    attn_implementation: str = "sdpa"
    seed: int = 42
