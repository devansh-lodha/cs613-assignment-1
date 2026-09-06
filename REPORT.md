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

## Models and method

We compare EmbeddingGemma-300m (24 layers plus the input embedding, hidden size
768) and Qwen3-Embedding-0.6B (28 layers plus the input embedding, hidden size
1024). Both run in raw mode with no task prompt, and we read the hidden states
after every layer.

At each layer we treat the individual token vectors as a point cloud (we do not
pool them into sentence vectors), sampling up to 10,000 token vectors per layer
using reservoir sampling (Vitter's Algorithm R). On each cloud we compute four
isotropy metrics:

- IsoScore: how evenly variance is spread across all dimensions (Rudman et al.,
  2022). 0 is fully anisotropic, 1 is fully isotropic.
- Average Random Cosine Similarity: one minus the mean absolute cosine of random
  token pairs (Ethayarajh, 2019). Low values mean the tokens share a common
  direction.
- ID Score: intrinsic dimensionality (Levina-Bickel MLE) divided by the hidden
  size.
- SVD Ratio: the fraction of variance held by the single largest principal
  component (Timkey and van Schijndel, 2021). A high value means one dominant
  rogue dimension.

We run this for each of the four combinations: Gemma-English, Gemma-Hindi,
Qwen-English, and Qwen-Hindi. For downstream reporting and LaTeX inclusion, all
figures are saved both as high-resolution PNGs and as vector PDF files.

## Results

### IsoScore

![IsoScore](results/figures/isoscore_layerwise.png)

Both models are strongly anisotropic once past the input layer. Gemma sits near
0.01 across its depth and only recovers a little at the last layer (0.036 for
English, 0.045 for Hindi). Qwen is more extreme: from layer 3 onward its IsoScore
is effectively 0, and it rises only at the final layer (about 0.14 for both
languages). The English and Hindi curves lie almost on top of each other for
both models.

### SVD Ratio

![SVD Ratio](results/figures/svd_ratio_layerwise.png)

This plot explains Qwen's flat-zero IsoScore. Between layers 3 and 27 a single
principal component holds around 99.9% of the variance, so the whole token cloud
collapses onto one axis. The final layer breaks this open and drops the ratio to
about 0.05. Gemma never collapses this hard: its ratio swings between roughly 0.20
and 0.95 through the network and also falls at the final layer (0.15 to 0.18).

### Average Random Cosine Similarity

![Average Random Cosine Similarity](results/figures/avg_cosine_similarity_layerwise.png)

Values well below 1 mean the tokens point in a shared direction, that is, they
form a narrow cone. Both models tighten through the middle layers (Gemma dips to
about 0.06 near layer 23, Qwen stays around 0.50 to 0.60) and then open up
sharply at the final layer (Gemma reaches about 0.90, Qwen reaches about 0.78 to
0.83). The output layer is where directional dispersion is enforced.

### ID Score

![ID Score](results/figures/id_score_layerwise.png)

Intrinsic dimensionality stays low throughout, a few percent of the hidden size,
showing that the representations live on a thin manifold at every depth. The
clearest English-Hindi gap in the whole study shows up here. At Qwen's final layer
English reaches roughly 0.23 to 0.42 while Hindi stays at 0.037, and at the input
layer English is 0.14 against Hindi's 0.002. Hindi tokens occupy a noticeably
lower-dimensional manifold at both ends of the network.

### PCA Compression (3D Point Cloud Progression)

For each model and language combination, we project the token cloud onto its top
three principal components across model depth, deliberately pairing the penultimate
layer ($N-1$) directly alongside the final output layer ($N$) to isolate the
phenomenon of final-layer normalization and contrastive dispersion (layers 0, 6,
12, 17, 23, 24 for Gemma; layers 0, 7, 14, 20, 27, 28 for Qwen).

#### EmbeddingGemma (English)

![Gemma English PCA compression](results/BPCC_hin_Deva_google_embeddinggemma-300m_en/pca_compression.png)

The input layer displays an isotropic point cloud with top-3 variance share of
18.6%. Across intermediate layers (layers 6 to 17), the token cloud contracts
into an elongated manifold dominated by PC 1 (top-3 variance reaches 88.7% at
layer 6). At penultimate layer 23 ($N-1$), representations exhibit severe
directional stretching along PC 1 with magnitudes reaching 60,000 (top-3 variance
share 72.4%). At final layer 24 ($N$), the output normalization and contrastive
projection immediately collapse this dominant axis, dispersing representations
into a bounded spherical cloud with top-3 variance dropping back to 22.0%.

#### EmbeddingGemma (Hindi)

![Gemma Hindi PCA compression](results/BPCC_hin_Deva_google_embeddinggemma-300m_hi/pca_compression.png)

Hindi mirrors the English trajectory closely: an initial isotropic spread at
layer 0 (21.3% top-3 variance), tightening into a dominant axis at layer 6
(89.2% top-3 variance). At penultimate layer 23 ($N-1$), representations reach
77.0% top-3 variance with extreme PC 1 elongation. At final layer 24 ($N$),
normalization redistributes tokens into a dense, spherical cluster, lowering top-3
variance share to 24.8%.

#### Qwen3-Embedding (English)

![Qwen English PCA compression](results/BPCC_hin_Deva_Qwen_Qwen3-Embedding-0.6B_en/pca_compression.png)

Qwen exhibits extreme geometric collapse across intermediate layers 7 through 20,
where top-3 variance share reaches 99.1% to 100.0%, directly reflecting the 99.9%
single-axis SVD dominance observed quantitatively. At penultimate layer 27
($N-1$), representations remain heavily collapsed along a narrow line (top-3
variance 93.7%). At final layer 28 ($N$), the single-axis collapse is broken
open, dropping top-3 variance share to 10.4% and restoring full 3D volume.

#### Qwen3-Embedding (Hindi)

![Qwen Hindi PCA compression](results/BPCC_hin_Deva_Qwen_Qwen3-Embedding-0.6B_hi/pca_compression.png)

Hindi representations in Qwen follow the same path: near-complete collapse across
layers 7 through 20 (97.1% to 99.9% top-3 variance), remaining collapsed at
penultimate layer 27 ($N-1$, 78.5% top-3 variance). At final layer 28 ($N$),
variance suddenly redistributes across all dimensions (9.5% top-3 variance). In
both layer 0 and layer 28, Hindi token vectors cluster into tighter discrete
groupings than English, visually verifying the lower intrinsic dimensionality
(ID score) measured for Hindi.

## Takeaways

- Both models squeeze tokens onto a narrow, low-dimensional cone in their middle
  layers and then reopen the space at the output layer. This reflects contrastive
  fine-tuning objectives, which enforce hyperspherical uniformity at the output
  layer for cosine-similarity retrieval.
- Qwen's mid-network collapse is far more severe than Gemma's. For most of its
  depth a single principal component carries over 99.9% of the variance.
- English and Hindi geometry is broadly aligned across layer depth, with one
  clear structural difference: Hindi consistently occupies a lower-dimensional
  manifold, most visible at Qwen's input and output layers.

## Artifacts

- Quantitative metrics: `results/BPCC_hin_Deva_<model>_<lang>/metrics.csv` and
  `metrics.json`
- 3D PCA compression figures:
  `results/BPCC_hin_Deva_<model>_<lang>/pca_compression.png` and
  `pca_compression.pdf`
- Layer-wise metric comparison plots:
  `results/figures/<metric>_layerwise.png` and `<metric>_layerwise.pdf`
