"""Isotropy metrics for layer-wise representation-space geometry analysis.

Each metric operates on a point cloud ``X`` of shape ``(n, d)``: ``n`` token
representations living in a ``d``-dimensional embedding space, extracted after a
given transformer layer. Together they quantify how uniformly the layer spreads
its representations across the available space (isotropy) versus collapsing them
onto a narrow cone or a few dominant directions (anisotropy).
"""

from dataclasses import asdict, dataclass

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

EPSILON: float = 1e-12
_POINT_CLOUD_NDIM: int = 2


@dataclass(frozen=True)
class LayerIsotropyRecord:
    """Isotropy metrics for the point cloud produced by a single layer."""

    layer_index: int
    layer_name: str
    isoscore: float
    avg_cosine_similarity: float
    id_score: float
    mev: float

    def to_dict(self) -> dict[str, float | int | str]:
        """Convert the record to a JSON/CSV-serializable dictionary."""
        return asdict(self)


def _to_numpy(embeddings: np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert to a float64 array, dropping any rows with non-finite values.

    Non-finite rows (e.g. from float16 overflow upstream) would otherwise make the
    covariance matrix non-finite and break the eigendecomposition.
    """
    if isinstance(embeddings, torch.Tensor):
        x = embeddings.detach().float().cpu().numpy().astype(np.float64)
    else:
        x = np.asarray(embeddings, dtype=np.float64)
    if x.ndim == _POINT_CLOUD_NDIM:
        x = x[np.isfinite(x).all(axis=1)]
    return x


def _covariance_eigenvalues(x: np.ndarray) -> np.ndarray:
    """Return the non-negative covariance eigenvalues (PCA variances) of ``x``.

    These eigenvalues are shared by IsoScore and the SVD Ratio, so computing them
    once avoids a redundant decomposition of the ``d x d`` covariance matrix.
    """
    centered = x - np.mean(x, axis=0)
    covariance = np.cov(centered, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)
    return np.maximum(eigenvalues, 0.0)


def _isoscore_from_eigenvalues(eigenvalues: np.ndarray, d_dims: int) -> float:
    """Compute IsoScore from precomputed covariance eigenvalues."""
    norm_val = float(np.linalg.norm(eigenvalues))
    if norm_val <= EPSILON:
        return 0.0

    d = float(d_dims)

    # Normalize so a perfectly isotropic cloud maps to the all-ones vector.
    sigma_hat = np.sqrt(d) * (eigenvalues / norm_val)
    ones = np.ones(d_dims, dtype=np.float64)

    denom_delta = np.sqrt(2.0 * (d - np.sqrt(d)))
    if denom_delta <= EPSILON:
        return 0.0

    delta = float(np.linalg.norm(sigma_hat - ones) / denom_delta)

    phi = ((d - (delta**2) * (d - np.sqrt(d))) ** 2) / (d**2)
    score = (d * phi - 1.0) / (d - 1.0)

    return float(np.clip(score, 0.0, 1.0))


def _mev_from_eigenvalues(eigenvalues: np.ndarray) -> float:
    """Compute the maximum explained variance ratio (MEV) from eigenvalues."""
    total_variance = float(np.sum(eigenvalues))
    if total_variance <= EPSILON:
        return 0.0
    return float(np.max(eigenvalues) / total_variance)


def compute_isoscore(embeddings: np.ndarray | torch.Tensor) -> float:
    """Compute IsoScore (Rudman et al., 2022) in ``[0, 1]``.

    Reorients the cloud with PCA, normalizes the resulting variance vector, and
    measures its Euclidean distance from the isotropic identity. Returns 0 for
    maximal anisotropy (all variance on one axis) and 1 for perfect isotropy.
    """
    x = _to_numpy(embeddings)
    n_samples, d_dims = x.shape
    if n_samples <= 1 or d_dims <= 1:
        return 0.0
    return _isoscore_from_eigenvalues(_covariance_eigenvalues(x), d_dims)


def compute_avg_cosine_similarity(
    embeddings: np.ndarray | torch.Tensor,
    num_pairs: int = 100_000,
    seed: int = 42,
) -> float:
    """Compute ``1 - |mean random-pair cosine similarity|`` (Ethayarajh, 2019).

    Approximates angular spread by averaging the cosine similarity of randomly
    sampled distinct point pairs. Returns 0 when vectors point in similar
    directions (minimal isotropy) and 1 for maximal isotropy.
    """
    x = _to_numpy(embeddings)
    n_samples = x.shape[0]
    if n_samples <= 1:
        return 0.0

    norms = np.linalg.norm(x, axis=1, keepdims=True)
    normalized = x / np.maximum(norms, EPSILON)

    rng = np.random.default_rng(seed)
    idx_a = rng.integers(0, n_samples, size=num_pairs)
    idx_b = rng.integers(0, n_samples, size=num_pairs)

    distinct = idx_a != idx_b
    idx_a, idx_b = idx_a[distinct], idx_b[distinct]
    if idx_a.size == 0:
        return 0.0

    cosines = np.sum(normalized[idx_a] * normalized[idx_b], axis=1)
    return float(1.0 - abs(float(np.mean(cosines))))


def compute_mev(embeddings: np.ndarray | torch.Tensor) -> float:
    """Compute Maximum Explained Variance (MEV): top eigenvalue share.

    Calculates ``lambda_1 / sum(lambda)``, isolating the fraction of variance
    captured by the single dominant "rogue" direction (formerly referred to as SVD
    ratio). A high value signals that one dimension absorbs a disproportionate
    share of the representational capacity.
    """
    x = _to_numpy(embeddings)
    if x.shape[0] <= 1:
        return 0.0
    return _mev_from_eigenvalues(_covariance_eigenvalues(x))


def compute_id_score(
    embeddings: np.ndarray | torch.Tensor,
    k: int = 10,
    sample_size: int = 5000,
    seed: int = 42,
) -> float:
    """Compute normalized intrinsic dimensionality: ``ID(X) / d``.

    Estimates the manifold dimension via the Levina-Bickel (2004) maximum
    likelihood estimator over ``k`` nearest neighbors, then normalizes by the
    ambient dimension ``d``. Points whose neighborhood collapses to zero distance
    (exact duplicates) are excluded to keep the estimate finite. The neighbor
    graph is capped at ``sample_size`` points to bound the quadratic cost in high
    dimensions.
    """
    if isinstance(embeddings, torch.Tensor):
        x = embeddings.detach().float().cpu().numpy().astype(np.float32)
    else:
        x = np.asarray(embeddings, dtype=np.float32)

    if x.shape[0] > sample_size:
        rng = np.random.default_rng(seed)
        x = x[rng.choice(x.shape[0], size=sample_size, replace=False)]

    n_samples, d = x.shape
    if n_samples <= k:
        return 0.0

    nn = NearestNeighbors(n_neighbors=k + 1).fit(x)
    distances, _ = nn.kneighbors(x)

    r_k = distances[:, k]
    r_j = distances[:, 1:k]

    # Exclude degenerate points whose k-th neighbor coincides with the point.
    valid = r_k > EPSILON
    if not np.any(valid):
        return 0.0

    r_k = r_k[valid]
    r_j = r_j[valid]

    log_ratios = np.log(r_k[:, None] / np.maximum(r_j, EPSILON))
    sum_logs = np.sum(log_ratios, axis=1)

    finite = sum_logs > EPSILON
    if not np.any(finite):
        return 0.0

    id_i = (k - 1) / sum_logs[finite]
    id_x = float(np.mean(id_i))

    return float(id_x / d)


def analyze_point_cloud(embeddings: np.ndarray | torch.Tensor) -> tuple[float, ...]:
    """Compute all four isotropy metrics for a single point cloud.

    The covariance eigendecomposition is shared between IsoScore and MEV
    to avoid decomposing the ``d x d`` covariance matrix twice.

    Returns:
        Tuple ``(isoscore, avg_cosine_similarity, id_score, mev)``.

    """
    x = _to_numpy(embeddings)
    n_samples, d_dims = x.shape
    if n_samples <= 1 or d_dims <= 1:
        return (0.0, 0.0, 0.0, 0.0)

    eigenvalues = _covariance_eigenvalues(x)
    isoscore = _isoscore_from_eigenvalues(eigenvalues, d_dims)
    mev = _mev_from_eigenvalues(eigenvalues)
    avg_cos = compute_avg_cosine_similarity(x)
    id_score = compute_id_score(x)

    return (isoscore, avg_cos, id_score, mev)


def analyze_layer_clouds(
    layer_clouds: list[np.ndarray],
) -> list[LayerIsotropyRecord]:
    """Compute isotropy metrics for each per-layer token point cloud.

    Args:
        layer_clouds: List indexed by layer (0 = input embeddings), each an
            ``(n, d)`` array of token representations at that layer.

    Returns:
        One :class:`LayerIsotropyRecord` per layer.

    """
    records: list[LayerIsotropyRecord] = []
    for layer_idx, cloud in enumerate(layer_clouds):
        isoscore, avg_cos, id_score, mev = analyze_point_cloud(cloud)
        name = f"{layer_idx} (Embed)" if layer_idx == 0 else str(layer_idx)
        records.append(
            LayerIsotropyRecord(
                layer_index=layer_idx,
                layer_name=name,
                isoscore=isoscore,
                avg_cosine_similarity=avg_cos,
                id_score=id_score,
                mev=mev,
            ),
        )
    return records
