# Layer-wise Isotropy of Multilingual Embeddings on BPCC (English vs Hindi)

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
1024). Both models run in full native `bfloat16` precision with default model
architectures.

To evaluate representation isotropy after each transformer layer, we formalize
the token representations as a point cloud. Given an input token sequence of
length $n$, the tokenizer and subsequent embedding layer produce a representation
matrix $X \in \mathbb{R}^{n \times d}$, where $d$ is the embedding dimension.
After processing this sequence through a transformer layer, we obtain an updated
$n \times d$ representation matrix $X$. By treating the $n$ row vectors of this
matrix as a point cloud $X \subseteq \mathbb{R}^d$, we sample up to $K = 10,000$
non-padding token representations uniformly across the corpus using reservoir
sampling (Vitter's Algorithm R). On each sampled token cloud, we compute four
intrinsic geometric metrics:

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

A score near 0 represents minimal isotropy (vectors frequently point in similar
directions, forming a narrow directional cone), while a score near 1 indicates
maximal isotropy.

### Intrinsic Dimensionality (ID) Score

Adapted for evaluating embedding anisotropy by Cai et al. (2021) utilizing the
Maximum Likelihood Estimation method by Levina and Bickel (2004), this score
estimates the true dimension of the underlying manifold from which the point
cloud is sampled. It calculates the intrinsic dimensionality $\text{ID}(X)$ and
divides this value by the ambient embedding dimension $d$ to provide a normalized
ID Score:

$$\text{ID Score} := \frac{\text{ID}(X)}{d}$$

This detects whether transformer layers compress tokens into narrow, low-dimensional
subspaces.

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

Both models are strongly anisotropic once past the input layer. Gemma sits near
0.01 across its depth, drops to 0.0012 at penultimate layer 23, and recovers to
0.033 (English) and 0.044 (Hindi) at final layer 24. Qwen is more extreme: from
layer 3 onward its IsoScore is practically 0.0000, and it rises sharply only at
the final layer (0.136 for English, 0.166 for Hindi). The English and Hindi
curves lie almost on top of each other for both models.

### Maximum Explainable Variance (MEV)

![MEV](results/figures/mev_layerwise.png)

This plot explains Qwen's flat-zero IsoScore. Between layers 3 and 10 a single
principal component holds around 99.99% of the variance, so the whole token cloud
collapses onto one axis. This single-axis dominance gradually relaxes in later
layers and drops to 0.044 (English) and 0.049 (Hindi) at the final layer. Gemma
never collapses this hard: its MEV swings between roughly 0.20 and 0.84 through
the network, peaks at penultimate layer 23 ($N-1$) at 0.718 (English) and 0.758
(Hindi), and falls to 0.186 (English) and 0.152 (Hindi) at final layer 24 ($N$).

### Average Random Cosine Similarity (AvgCos)

![Average Random Cosine Similarity](results/figures/avg_cosine_similarity_layerwise.png)

Values well below 1 mean the tokens point in a shared direction, forming a
narrow cone. Both models tighten through the middle layers: Gemma dips to its
minimum at layer 23 (0.061 for English, 0.059 for Hindi, representing peak cone
formation where cosine similarity reaches ~0.94), while Qwen dips to 0.248
(English) and 0.347 (Hindi) at layer 27 ($N-1$). At the final output layer, both
models break the cone open: Gemma surges to 0.910 (English) and 0.903 (Hindi),
and Qwen surges to 0.773 (English) and 0.830 (Hindi). The output layer is where
directional dispersion is enforced.

### ID Score

![ID Score](results/figures/id_score_layerwise.png)

Intrinsic dimensionality stays low throughout, a few percent of the hidden size,
showing that token representations live on a thin manifold at every depth. The
clearest English-Hindi gap in the study shows up at Qwen's input layer (0.137 for
English vs 0.002 for Hindi), while intermediate and deep layers remain closely
aligned across languages.

### PCA Compression (3D Point Cloud Progression)

For each model and language combination, we project the unit-normalized token
representations onto their top three principal components across model depth within
a fixed shared coordinate frame ($[-0.35, 0.35]$ across all axes). This isolates
the phenomenon of final-layer normalization and contrastive dispersion by
deliberately pairing penultimate layer $N-1$ (highlighted in red) directly
alongside final output layer $N$ (highlighted in green).

#### EmbeddingGemma (English)

![Gemma English PCA compression](results/BPCC_hin_Deva_google_embeddinggemma-300m_en/pca_compression.png)

The input layer displays an isotropic distribution with 3D spatial spread of
0.2359 and top-3 variance share of 21.1%. Intermediate layers contract steadily:
spread drops to 0.2635 at layer 6, 0.1436 at layer 12, and 0.1267 at layer 17.
At penultimate layer 23 ($N-1$, red), directional collapse reaches its peak,
compressing into a tight pinpoint knot with a 3D spread of 0.0546. At final
layer 24 ($N$, green), output normalization and contrastive projection burst
this collapsed state back open into an expansive volume (spread surges to 0.1749).

#### EmbeddingGemma (Hindi)

![Gemma Hindi PCA compression](results/BPCC_hin_Deva_google_embeddinggemma-300m_hi/pca_compression.png)

Hindi mirrors the English trajectory: a wide initial distribution at layer 0
(spread 0.2146), narrowing across layers 6 to 17 (spread dropping to 0.2469,
0.1343, and 0.1246). At penultimate layer 23 ($N-1$, red), representations
collapse into a tiny pinpoint knot with a minimal spread of 0.0609. At final
layer 24 ($N$, green), representations burst open into an expansive, spherical
cloud with spread surging to 0.1850.

#### Qwen3-Embedding (English)

![Qwen English PCA compression](results/BPCC_hin_Deva_Qwen_Qwen3-Embedding-0.6B_en/pca_compression.png)

Qwen displays extreme directional collapse across intermediate layers: spread
drops from 0.2292 at layer 0 down to 0.1681 at layer 7, 0.1712 at layer 14, and
0.1587 at layer 20. Penultimate layer 27 ($N-1$, red) reaches maximum compression
(spread 0.1141). At final layer 28 ($N$, green), the collapsed cone is broken,
decompressing into a full 3D distribution with spread surging to 0.1697.

#### Qwen3-Embedding (Hindi)

![Qwen Hindi PCA compression](results/BPCC_hin_Deva_Qwen_Qwen3-Embedding-0.6B_hi/pca_compression.png)

Hindi token representations in Qwen follow the same trajectory: spread drops
from 0.2451 at layer 0 down to 0.1944 at layer 7, 0.1677 at layer 14, and
reaches peak collapse at penultimate layer 27 ($N-1$, red, spread 0.1158). At
final layer 28 ($N$, green), the distribution decompresses into an expansive
volume with spread surging to 0.1594.

## Takeaways

- Both models compress token representations into a narrow, low-dimensional
  cone across their middle layers and then reopen the space at the output layer.
  This reflects contrastive fine-tuning objectives, which enforce hyperspherical
  uniformity at the output layer for cosine-similarity retrieval.
- Qwen exhibits acute single-axis collapse (MEV > 99.9%) in its early-to-middle
  depths before gradually decompressing, whereas Gemma maintains a more moderate
  variance distribution until an acute spike at penultimate layer 23 ($N-1$).
- In both architectures, the transition from layer $N-1$ to layer $N$ is where the
  directional cone is eliminated and spherical isotropy is restored.
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
