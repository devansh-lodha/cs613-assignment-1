# Layer-wise Isotropy of Multilingual Sentence Embeddings on BPCC (English vs Hindi)

We measure how two embedding models organize their representation space at each
layer, and whether they treat English and Hindi differently. Every measurement
uses the same random parallel subset of BPCC, so any gap between the two
languages comes from the model, not from the text.

## Dataset

The Bharat Parallel Corpus Collection (BPCC), released by AI4Bharat, is a large
collection of English-Indic sentence pairs. We use its Wikipedia split for Hindi
(`wiki/hin_Deva.tsv`), where each row holds an English sentence and its Hindi
translation. From roughly 40,000 pairs we draw a fixed random sample of 10,000,
which gives us 10,000 English sentences and the 10,000 Hindi sentences that mean
the same thing. Because the two sides are exact translations, differences we see
between them reflect how a model encodes the two languages rather than any
difference in what is being said.

## Models and Method

We compare EmbeddingGemma-300m (24 layers plus the input embedding, hidden size
768) and Qwen3-Embedding-0.6B (28 layers plus the input embedding, hidden size
1024). Both models run in full native `bfloat16` precision without quantization,
adhering strictly to default architectures and native pooling specifications:

- **EmbeddingGemma:** Evaluated using attention-masked mean pooling, matching
  Google's official specification (`pooling_mode_mean_tokens: true`).
- **Qwen3-Embedding:** Evaluated using last-token pooling, matching Alibaba Qwen's
  official specification (`pooling_mode_lasttoken: true`).

At each layer, we compute the representation matrix $X \in \mathbb{R}^{N \times d}$
across the $N = 10,000$ parallel sentences. Treating the $N$ row vectors as a
point cloud $X \subseteq \mathbb{R}^d$, we calculate four intrinsic geometric
metrics:

### IsoScore

Introduced by Rudman et al. (2022), this metric quantifies the exact degree to
which the point cloud $X \subseteq \mathbb{R}^d$ uniformly utilizes the ambient
vector space. The method first reorients the data using PCA to decorrelate
dimensions, and then normalizes the resulting diagonal covariance vector
$\hat{\Sigma}_D$. It computes an isotropy defect $\delta(X)$ measuring the
Euclidean distance between this variance vector and the identity matrix $\mathbf{1}$:

$$\delta(X) := \frac{\|\hat{\Sigma}_D - \mathbf{1}\|}{\sqrt{2(d-\sqrt{d})}}$$

$$\iota(X) := \frac{(d - \delta(X)\sqrt{2(d-\sqrt{d})})^2 - d}{d(d - 1)}$$

The final IsoScore $\iota(X)$ maps this defect to a range where 0 indicates
minimal isotropy (maximal anisotropy) and 1 indicates maximal isotropy.

### Average Random Cosine Similarity (AvgCos)

First utilized by Ethayarajh (2019) to identify representation degeneration in
contextual embeddings, this metric approximates spatial uniformity by computing
the average cosine similarity of $N$ randomly sampled pairs of point vectors from
$X$. The absolute value of this average is subtracted from 1:

$$\text{AvgCos} := 1 - \left| \frac{1}{N} \sum_{i=1}^{N} \frac{x_i \cdot y_i}{\|x_i\| \|y_i\|} \right|$$

A score of 0 represents minimal isotropy (meaning vectors frequently point in
similar directions, forming an acute directional cone), and a score of 1 indicates
maximal isotropy.

### Intrinsic Dimensionality (ID) Score

Adapted for evaluating embedding anisotropy by Cai et al. (2021) utilizing the
Maximum Likelihood Estimation method by Levina and Bickel (2004), this score
estimates the true dimension of the underlying manifold from which the point
cloud is sampled. It calculates the intrinsic dimensionality $\text{ID}(X)$ and
divides this value by the ambient embedding dimension $d$ to provide a normalized
ID Score:

$$\text{ID Score} := \frac{\text{ID}(X)}{d}$$

We incorporate this metric specifically to detect whether transformer layers
compress sentences into narrow, low-dimensional language manifolds.

### Maximum Explainable Variance (MEV)

While the IsoScore evaluates the overall distribution of variance across the
representation space, transformer embeddings frequently suffer from severe
anisotropy driven by a single dominant "rogue dimension." To explicitly measure
this phenomenon, we compute the MEV. It isolates the variance captured by the top
principal component (the largest eigenvalue $\lambda_1$) and divides it by the
sum of all eigenvalues:

$$\text{MEV} := \frac{\lambda_1}{\sum_{i=1}^{d} \lambda_i}$$

A high MEV indicates that a single rogue dimension is absorbing a disproportionate
amount of the representational capacity.

We evaluate all four metrics across depth for each of the four combinations:
Gemma-English, Gemma-Hindi, Qwen-English, and Qwen-Hindi. For reporting and LaTeX
inclusion, all figures are saved both as high-resolution PNGs and as vector PDF
files.

## Results

### IsoScore

![IsoScore](results/figures/isoscore_layerwise.png)

Both models exhibit pronounced anisotropy through intermediate layers before
experiencing significant isotropic recovery at the final layer. Gemma starts at
0.058 (English) and 0.028 (Hindi), drops to 0.0030 at penultimate layer 23, and
then surges to 0.198 (English) and 0.220 (Hindi) at output layer 24. Qwen starts
near 0, remains between 0.01 and 0.05 across middle depths, rises to 0.097
(English) and 0.108 (Hindi) at layer 27, and reaches its maximum at 0.116
(English) and 0.125 (Hindi) at output layer 28. English and Hindi trajectories
track each other almost identically throughout.

### Maximum Explainable Variance (MEV)

![MEV](results/figures/mev_layerwise.png)

MEV isolates the variance fraction captured by the leading principal component.
In Gemma, MEV peaks early at layer 1 (0.494 for English, 0.749 for Hindi), settles
between 0.15 and 0.30 through intermediate layers, spikes at penultimate layer 23
($N-1$) to 0.547 (English) and 0.612 (Hindi), and then collapses to 0.047
(English) and 0.037 (Hindi) at final layer 24 ($N$). In Qwen, MEV stays around
0.15 to 0.20 in early-to-mid layers, steadily declines to 0.066 (English) and
0.055 (Hindi) at layer 27, and drops to its minimum of 0.051 (English) and 0.041
(Hindi) at layer 28.

### Average Random Cosine Similarity (AvgCos)

![Average Random Cosine Similarity](results/figures/avg_cosine_similarity_layerwise.png)

Values near 0 indicate that sentences collapse into a narrow directional cone,
where pairwise cosine similarity approaches 1.0. Both models maintain tight
directional alignment throughout their internal layers: Gemma dips to its
minimum at layer 23 (0.0037 for English, 0.0029 for Hindi, representing peak
cone collapse where random sentence cosine similarity reaches 0.9963), while Qwen
sits between 0.01 and 0.08 across layers 1 to 27. At the final output layer, both
models break the cone open: Gemma surges to 0.508 (English) and 0.491 (Hindi),
and Qwen surges to 0.749 (English) and 0.825 (Hindi). The output layer is where
angular dispersion is enforced.

### ID Score

![ID Score](results/figures/id_score_layerwise.png)

Normalized intrinsic dimensionality remains low throughout the network (2% to 6%
of ambient dimension), demonstrating that sentence representations live on a
thin manifold at every depth. Gemma displays slightly higher intrinsic
dimensionality than Qwen through intermediate layers (0.035 to 0.042 vs. 0.022
to 0.028). English and Hindi curves follow the exact same manifold depth
trajectory in both models.

### PCA Compression (3D Point Cloud Progression)

For each model and language combination, we project the unit-normalized sentence
representations ($\mathbf{u} = \mathbf{s} / \|\mathbf{s}\|_2$) onto their top
three principal components across model depth within a fixed shared coordinate
frame ($[-0.35, 0.35]$ across all axes). This directly visualizes the directional
cone and isolates final-layer normalization by deliberately pairing penultimate
layer $N-1$ (highlighted in red) directly alongside final output layer $N$
(highlighted in green).

#### EmbeddingGemma (English)

![Gemma English PCA compression](results/BPCC_hin_Deva_google_embeddinggemma-300m_en/pca_compression.png)

The input layer displays an isotropic distribution with 3D spatial spread of
0.1169 and top-3 variance share of 19.2%. Intermediate layers contract steadily:
spread drops to 0.0766 at layer 6, 0.0472 at layer 12, and 0.0524 at layer 17.
At penultimate layer 23 ($N-1$, red), directional collapse reaches its peak,
compressing into a tight pinpoint knot with a 3D spread of 0.0113. At final
layer 24 ($N$, green), output normalization and contrastive projection burst
this collapsed state back open into an expansive volume (spread surges to 0.1137).

#### EmbeddingGemma (Hindi)

![Gemma Hindi PCA compression](results/BPCC_hin_Deva_google_embeddinggemma-300m_hi/pca_compression.png)

Hindi mirrors the English trajectory: a wide initial distribution at layer 0
(spread 0.1036), narrowing across layers 6 to 17 (spread dropping to 0.0656,
0.0421, and 0.0504). At penultimate layer 23 ($N-1$, red), representations
collapse into a tiny pinpoint knot with a minimal spread of 0.0099. At final
layer 24 ($N$, green), representations burst open into an expansive, spherical
cloud with spread surging to 0.1065.

#### Qwen3-Embedding (English)

![Qwen English PCA compression](results/BPCC_hin_Deva_Qwen_Qwen3-Embedding-0.6B_en/pca_compression.png)

At input layer 0, all sequences share the initial EOS token embedding under
last-token pooling. Across intermediate layers 7 through 20, sentence embeddings
form a moderately dense cluster (spread 0.0713 to 0.0937). Penultimate layer 27
($N-1$, red) tightens to a spread of 0.0500. At final layer 28 ($N$, green), the
collapsed cone is broken, decompressing into an expansive 3D distribution with
spread surging to 0.1592.

#### Qwen3-Embedding (Hindi)

![Qwen Hindi PCA compression](results/BPCC_hin_Deva_Qwen_Qwen3-Embedding-0.6B_hi/pca_compression.png)

Hindi sentence embeddings in Qwen follow the same path: spread starts at 0.0551
at layer 7, progresses through layers 14 to 20 (spread 0.0589 to 0.0734), and
holds at penultimate layer 27 ($N-1$, red, spread 0.0425). At final layer 28
($N$, green), the distribution decompresses into an expansive volume with spread
surging to 0.1566.

## Takeaways

- Both models compress sentence representations into a narrow, low-dimensional
  cone across their middle layers and then reopen the space at the output layer.
  This reflects contrastive fine-tuning objectives, which enforce hyperspherical
  uniformity at the output layer for cosine-similarity retrieval.
- Both architectures exhibit their sharpest geometric transition between
  penultimate layer $N-1$ and final layer $N$, where output normalization
  eliminates dominant rogue axes and restores spherical isotropy.
- English and Hindi representations display remarkably consistent geometry layer
  by layer across both models, confirming that representation space organization
  is largely language-invariant under parallel semantic content.

## Artifacts

- Quantitative metrics: `results/BPCC_hin_Deva_<model>_<lang>/metrics.csv` and
  `metrics.json`
- 3D PCA compression figures:
  `results/BPCC_hin_Deva_<model>_<lang>/pca_compression.png` and
  `pca_compression.pdf`
- Layer-wise metric comparison plots:
  `results/figures/<metric>_layerwise.png` and `<metric>_layerwise.pdf`
  (including `isoscore_layerwise.*`, `mev_layerwise.*`,
  `avg_cosine_similarity_layerwise.*`, `id_score_layerwise.*`)
