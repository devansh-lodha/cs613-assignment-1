"""Geometric representation analysis metrics for embedding space evaluation."""

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

from embedding_gemma.model import EmbeddingOutput


EPSILON: float = 1e-12


@dataclass(frozen=True)
class LayerGeometryRecord:
    """Quantitative geometry metrics for a single neural network layer."""

    layer_index: int
    layer_name: str
    cosine_anisotropy: float
    isoscore: float
    rogue_ratio: float
    id_score: float
    spearman_correlation: float

    def to_dict(self) -> dict[str, float | int | str]:
        """Convert metric record to a serializable dictionary."""
        return asdict(self)


def compute_cosine_anisotropy(embeddings: torch.Tensor) -> float:
    """Compute exact average pairwise cosine similarity."""

    n = embeddings.size(0)

    if n <= 1:
        return 0.0

    normalized = F.normalize(
        embeddings.float(),
        p=2,
        dim=-1,
    )

    mean_vector = normalized.mean(dim=0)
    norm_sq = float((mean_vector**2).sum().item())

    # Exact all-pairs average excluding self-similarity diagonal.
    anisotropy = (n * norm_sq - 1.0) / (n - 1)

    return float(np.clip(anisotropy, -1.0, 1.0))


def compute_isoscore(
    embeddings: np.ndarray | torch.Tensor,
) -> float:
    """Compute IsoScore quantifying variance uniformity."""

    if isinstance(embeddings, torch.Tensor):
        x = embeddings.detach().float().cpu().numpy()
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
    eigenvalues = np.sort(
        np.maximum(eigenvalues, 0.0),
    )[::-1]

    norm_val = float(np.linalg.norm(eigenvalues))

    if norm_val <= EPSILON:
        return 0.0

    d = float(d_dims)

    sigma_hat = np.sqrt(d) * (
        eigenvalues / norm_val
    )

    ones = np.ones(
        d_dims,
        dtype=np.float64,
    )

    denom_delta = np.sqrt(
        2.0 * (d - np.sqrt(d)),
    )

    if denom_delta <= EPSILON:
        return 0.0

    delta = float(
        np.linalg.norm(sigma_hat - ones)
        / denom_delta
    )

    phi = (
        (
            d
            - (delta**2)
            * (d - np.sqrt(d))
        )
        ** 2
    ) / (d**2)

    score = (
        d * phi - 1.0
    ) / (d - 1.0)

    return float(
        np.clip(score, 0.0, 1.0),
    )


def compute_rogue_dimension_ratio(
    embeddings: np.ndarray | torch.Tensor,
) -> float:
    """Compute top singular value variance fraction."""

    if isinstance(embeddings, torch.Tensor):
        x = embeddings.detach().float().cpu().numpy()
    else:
        x = np.asarray(
            embeddings,
            dtype=np.float64,
        )

    if x.shape[0] <= 1:
        return 0.0

    centered = x - np.mean(
        x,
        axis=0,
    )

    # SVD on centered data gives principal component variances.
    _, singular_values, _ = np.linalg.svd(
        centered,
        full_matrices=False,
    )

    eigenvalues = singular_values**2

    total_variance = float(
        np.sum(eigenvalues),
    )

    if total_variance <= EPSILON:
        return 0.0

    return float(
        eigenvalues[0] / total_variance,
    )


def compute_spearman_correlation(
    predicted_similarities: np.ndarray | torch.Tensor,
    gold_scores: np.ndarray | Sequence[float],
) -> float:
    """Compute Spearman rank correlation against gold scores."""

    if isinstance(predicted_similarities, torch.Tensor):
        preds = (
            predicted_similarities
            .detach()
            .float()
            .cpu()
            .numpy()
        )
    else:
        preds = np.asarray(
            predicted_similarities,
            dtype=np.float64,
        )

    golds = np.asarray(
        gold_scores,
        dtype=np.float64,
    )

    result = spearmanr(
        preds,
        golds,
    )

    return float(result.statistic)


def compute_id_score(
    hidden_states: np.ndarray | torch.Tensor,
    k: int = 10,
) -> float:
    """Compute normalized intrinsic dimensionality score."""

    # Convert PyTorch tensors, including bfloat16,
    # to CPU float32 NumPy arrays for scikit-learn.
    if isinstance(hidden_states, torch.Tensor):
        hidden_states = (
            hidden_states
            .detach()
            .float()
            .cpu()
            .numpy()
        )
    else:
        hidden_states = np.asarray(
            hidden_states,
            dtype=np.float32,
        )

    n_samples, d = hidden_states.shape

    if n_samples <= k:
        return 0.0

    nn = NearestNeighbors(
        n_neighbors=k + 1,
    ).fit(hidden_states)

    distances, _ = nn.kneighbors(
        hidden_states,
    )

    # MLE calculation for intrinsic dimensionality.
    eps = 1e-8

    r_k = distances[:, k]
    r_j = distances[:, 1:k]

    log_ratios = np.log(
        (r_k[:, None] + eps)
        / (r_j + eps),
    )

    id_i = (
        (k - 1)
        / (
            np.sum(
                log_ratios,
                axis=1,
            )
            + eps
        )
    )

    id_x = np.mean(id_i)

    # Normalize by embedding dimension.
    return float(id_x / d)


def analyze_layer_geometry(
    output_s1: EmbeddingOutput,
    output_s2: EmbeddingOutput,
    gold_scores: np.ndarray | Sequence[float],
) -> list[LayerGeometryRecord]:
    """Compute geometric and semantic metrics across all layers."""

    if (
        output_s1.layer_hidden_states is None
        or output_s2.layer_hidden_states is None
    ):
        msg = (
            "Both inputs must contain layer_hidden_states "
            "(return_hidden_states=True)"
        )
        raise ValueError(msg)

    num_layers_s1 = len(
        output_s1.layer_hidden_states,
    )

    num_layers_s2 = len(
        output_s2.layer_hidden_states,
    )

    if num_layers_s1 != num_layers_s2:
        msg = (
            f"Layer count mismatch: "
            f"{num_layers_s1} vs {num_layers_s2}"
        )
        raise ValueError(msg)

    records: list[LayerGeometryRecord] = []

    for layer_idx in range(num_layers_s1):

        h1 = output_s1.layer_hidden_states[
            layer_idx
        ]

        h2 = output_s2.layer_hidden_states[
            layer_idx
        ]

        # Combine both sentence distributions
        # for corpus-level geometry metrics.
        h_combined = torch.cat(
            [h1, h2],
            dim=0,
        )

        # 1. Cosine anisotropy.
        anisotropy = compute_cosine_anisotropy(
            h_combined,
        )

        # 2. IsoScore.
        isoscore = compute_isoscore(
            h_combined,
        )

        # 3. Rogue dimension ratio.
        rogue_ratio = (
            compute_rogue_dimension_ratio(
                h_combined,
            )
        )

        # 4. Intrinsic dimensionality score.
        id_score = compute_id_score(
            h_combined,
        )

        # 5. Semantic similarity correlation.
        h1_norm = F.normalize(
            h1.float(),
            p=2,
            dim=-1,
        )

        h2_norm = F.normalize(
            h2.float(),
            p=2,
            dim=-1,
        )

        pair_similarities = (
            h1_norm * h2_norm
        ).sum(
            dim=-1,
        )

        # Skip Spearman correlation when gold scores are
        # missing or constant, as correlation is undefined.
        if gold_scores is not None and len(set(gold_scores)) > 1:
            spearman_rho = compute_spearman_correlation(
            pair_similarities,
            gold_scores,
        )
        else:
            spearman_rho = float("nan")
        # Layer display name.
        name = (
            f"{layer_idx} (Embed)"
            if layer_idx == 0
            else str(layer_idx)
        )

        # Create exactly one record per layer.
        records.append(
            LayerGeometryRecord(
                layer_index=layer_idx,
                layer_name=name,
                cosine_anisotropy=anisotropy,
                isoscore=isoscore,
                rogue_ratio=rogue_ratio,
                id_score=id_score,
                spearman_correlation=spearman_rho,
            ),
        )

    return records
