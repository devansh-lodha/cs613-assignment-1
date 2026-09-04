"""Geometric representation analysis metrics for embedding space evaluation."""

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from embedding_gemma.model import EmbeddingOutput

EPSILON: float = 1e-12


@dataclass(frozen=True)
class LayerGeometryRecord:
    """Quantitative geometry metrics for a single neural network layer.

    Attributes:
        layer_index: Depth index of the layer (0 = input embedding layer).
        layer_name: Display label for the layer.
        cosine_anisotropy: Average pairwise cosine similarity in [0, 1].
        isoscore: IsoScore isotropic space coverage metric in [0, 1].
        rogue_ratio: Variance fraction captured by the top singular value.
        spearman_correlation: Spearman rank correlation against gold scores.

    """

    layer_index: int
    layer_name: str
    cosine_anisotropy: float
    isoscore: float
    rogue_ratio: float
    spearman_correlation: float

    def to_dict(self) -> dict[str, float | int | str]:
        """Convert metric record to a serializable dictionary."""
        return asdict(self)


def compute_cosine_anisotropy(embeddings: torch.Tensor) -> float:
    """Compute exact average pairwise cosine similarity (Godey et al., 2024).

    Measures the directional cone angle of representation vectors:
        Anisotropy(H) = (1 / (N * (N - 1))) * sum_{i != j} cos(h_i, h_j)

    Using the algebraic identity for normalized vectors u_i = h_i / ||h_i||:
        sum_{i != j} (u_i . u_j) = ||sum_{i=1}^N u_i||_2^2 - N
    This allows exact calculation in O(N * d) time without quadratic memory.

    Args:
        embeddings: Tensor of shape (N, d) representing embedding vectors.

    Returns:
        Average pairwise cosine similarity float in [-1.0, 1.0].

    """
    n = embeddings.size(0)
    if n <= 1:
        return 0.0

    normalized = F.normalize(embeddings.float(), p=2, dim=-1)
    mean_vector = normalized.mean(dim=0)
    norm_sq = float((mean_vector**2).sum().item())

    # Exact all-pairs average excluding self-similarity diagonal:
    anisotropy = (n * norm_sq - 1.0) / (n - 1)
    return float(np.clip(anisotropy, -1.0, 1.0))


def compute_isoscore(embeddings: np.ndarray | torch.Tensor) -> float:
    """Compute IsoScore quantifying variance uniformity (Rudman et al., 2024).

    Quantifies whether variance is uniformly distributed across all d dimensions
    (isotropic, IsoScore -> 1.0) or collapsed into a low-dimensional subspace
    (anisotropic, IsoScore -> 0.0).

    Args:
        embeddings: Array or tensor of shape (N, d) of representation vectors.

    Returns:
        IsoScore scalar in [0.0, 1.0].

    """
    if isinstance(embeddings, torch.Tensor):
        x = embeddings.float().cpu().numpy()
    else:
        x = np.asarray(embeddings, dtype=np.float64)

    n_samples, d_dims = x.shape
    if n_samples <= 1 or d_dims <= 1:
        return 0.0

    # Center representations along each feature axis.
    centered = x - np.mean(x, axis=0)
    covariance = np.cov(centered, rowvar=False)

    eigenvalues = np.linalg.eigvalsh(covariance)
    # Filter negative numerical noise and sort descending.
    eigenvalues = np.sort(np.maximum(eigenvalues, 0.0))[::-1]

    norm_val = float(np.linalg.norm(eigenvalues))
    if norm_val <= EPSILON:
        return 0.0

    d = float(d_dims)
    sigma_hat = np.sqrt(d) * (eigenvalues / norm_val)
    ones = np.ones(d_dims, dtype=np.float64)

    denom_delta = np.sqrt(2.0 * (d - np.sqrt(d)))
    if denom_delta <= EPSILON:
        return 0.0

    delta = float(np.linalg.norm(sigma_hat - ones) / denom_delta)
    phi = ((d - (delta**2) * (d - np.sqrt(d))) ** 2) / (d**2)
    score = (d * phi - 1.0) / (d - 1.0)

    return float(np.clip(score, 0.0, 1.0))


def compute_rogue_dimension_ratio(embeddings: np.ndarray | torch.Tensor) -> float:
    """Compute top singular value variance fraction (Timkey & van Schijndel, 2021).

    Anisotropy is frequently driven by 1 to 3 'rogue dimensions' with massive
    variance that dominate vector dot products:
        Ratio_1 = sigma_1^2 / sum_{k=1}^d sigma_k^2

    Args:
        embeddings: Array or tensor of shape (N, d) of representation vectors.

    Returns:
        Ratio of top singular value variance to total variance in [0.0, 1.0].

    """
    if isinstance(embeddings, torch.Tensor):
        x = embeddings.float().cpu().numpy()
    else:
        x = np.asarray(embeddings, dtype=np.float64)

    if x.shape[0] <= 1:
        return 0.0

    centered = x - np.mean(x, axis=0)
    # SVD on centered data gives principal component variances.
    _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = singular_values**2
    total_variance = float(np.sum(eigenvalues))

    if total_variance <= EPSILON:
        return 0.0

    return float(eigenvalues[0] / total_variance)


def compute_spearman_correlation(
    predicted_similarities: np.ndarray | torch.Tensor,
    gold_scores: np.ndarray | Sequence[float],
) -> float:
    """Compute Spearman rank correlation against gold human judgments.

    Args:
        predicted_similarities: Cosine similarities between sentence pairs.
        gold_scores: Ground-truth human similarity ratings.

    Returns:
        Spearman rank correlation coefficient rho in [-1.0, 1.0].

    """
    if isinstance(predicted_similarities, torch.Tensor):
        preds = predicted_similarities.float().cpu().numpy()
    else:
        preds = np.asarray(predicted_similarities, dtype=np.float64)

    golds = np.asarray(gold_scores, dtype=np.float64)
    result = spearmanr(preds, golds)
    return float(result.statistic)


def analyze_layer_geometry(
    output_s1: EmbeddingOutput,
    output_s2: EmbeddingOutput,
    gold_scores: np.ndarray | Sequence[float],
) -> list[LayerGeometryRecord]:
    """Compute layer-wise geometric and semantic evaluation metrics across depth.

    Args:
        output_s1: EmbeddingOutput containing hidden states for sentence 1 split.
        output_s2: EmbeddingOutput containing hidden states for sentence 2 split.
        gold_scores: Ground-truth similarity ratings.

    Returns:
        List of LayerGeometryRecord objects for all model layers.

    Raises:
        ValueError: If hidden states are missing or layer counts mismatch.

    """
    if output_s1.layer_hidden_states is None or output_s2.layer_hidden_states is None:
        msg = "Both inputs must contain layer_hidden_states (return_hidden_states=True)"
        raise ValueError(msg)

    num_layers_s1 = len(output_s1.layer_hidden_states)
    num_layers_s2 = len(output_s2.layer_hidden_states)
    if num_layers_s1 != num_layers_s2:
        msg = f"Layer count mismatch: {num_layers_s1} vs {num_layers_s2}"
        raise ValueError(msg)

    records: list[LayerGeometryRecord] = []

    for layer_idx in range(num_layers_s1):
        h1 = output_s1.layer_hidden_states[layer_idx]
        h2 = output_s2.layer_hidden_states[layer_idx]

        # Combine both sentence distributions for corpus geometry metrics.
        h_combined = torch.cat([h1, h2], dim=0)

        # 1. Cosine Anisotropy (The Cone Angle)
        anisotropy = compute_cosine_anisotropy(h_combined)

        # 2. IsoScore (Isotropic Variance Uniformity)
        isoscore = compute_isoscore(h_combined)

        # 3. Rogue Dimension Ratio (Top Eigenvalue Share)
        rogue_ratio = compute_rogue_dimension_ratio(h_combined)

        # 4. STS-B Spearman Correlation (Cosine Similarity vs Human Gold Scores)
        h1_norm = F.normalize(h1.float(), p=2, dim=-1)
        h2_norm = F.normalize(h2.float(), p=2, dim=-1)
        pair_similarities = (h1_norm * h2_norm).sum(dim=-1).float().cpu().numpy()
        spearman_rho = compute_spearman_correlation(pair_similarities, gold_scores)

        name = f"{layer_idx} (Embed)" if layer_idx == 0 else str(layer_idx)
        records.append(
            LayerGeometryRecord(
                layer_index=layer_idx,
                layer_name=name,
                cosine_anisotropy=anisotropy,
                isoscore=isoscore,
                rogue_ratio=rogue_ratio,
                spearman_correlation=spearman_rho,
            ),
        )

    return records
