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

## Models and method

We compare EmbeddingGemma-300m (24 layers plus the input embedding, hidden size
768) and Qwen3-Embedding-0.6B (28 layers plus the input embedding, hidden size
1024). Both run in raw mode with no task prompt, and we read the hidden states
after every layer.

At each layer we compute attention-masked mean-pooled sentence representations
across the 10,000 parallel sentences. For an input sequence with hidden states
$\mathbf{h}_1, \dots, \mathbf{h}_T$ and binary attention mask $M \in \{0, 1\}^T$,
the sentence embedding is $\mathbf{s} = \frac{\sum_i M_i \mathbf{h}_i}{\sum_i M_i}$.
This yields an $(N, d)$ representation matrix at every depth, corresponding to the
sentence vectors evaluated in dense retrieval. On each representation matrix we
compute four intrinsic geometric metrics:

- IsoScore: how evenly variance is spread across all dimensions (Rudman et al.,
  2022). 0 is fully anisotropic, 1 is fully isotropic.
- Average Random Cosine Similarity: one minus the mean absolute cosine of random
  sentence pairs (Ethayarajh, 2019). Values near 0 indicate that sentences
  collapse into a narrow directional cone; values near 1 indicate wide angular
  dispersion.
- ID Score: intrinsic dimensionality (Levina-Bickel MLE) divided by the hidden
  size.
- MEV (Maximum Explained Variance ratio): the fraction of variance captured by
  the single largest principal component ($\lambda_1 / \sum \lambda_k$, formerly
  referred to as SVD ratio). A high value signifies rogue dimension dominance
  along one axis.

We run this for each of the four combinations: Gemma-English, Gemma-Hindi,
Qwen-English, and Qwen-Hindi. For downstream reporting and LaTeX inclusion, all
figures are saved both as high-resolution PNGs and as vector PDF files.

## Results

### IsoScore

![IsoScore](results/figures/isoscore_layerwise.png)

Both models are strongly anisotropic in intermediate layers. Gemma sits below
0.05 across its depth, drops to 0.0033 at penultimate layer 23, and then surges
to 0.196 (English) and 0.214 (Hindi) at final layer 24. Qwen collapses even
harder: from layer 3 onward its IsoScore is practically 0.0000, recovering
slightly in deep layers before jumping to 0.063 (English) and 0.069 (Hindi) at
output layer 28. The final layer in both models marks a significant isotropic
recovery.

### MEV (Maximum Explained Variance Ratio)

![MEV](results/figures/mev_layerwise.png)

MEV tracks the fraction of total variance consumed by the top principal component.
Qwen demonstrates severe single-axis dominance: between layers 3 and 10 a single
axis holds over 99.5% of all variance (reaching 99.9% at layers 3 to 5),
collapsing the entire sentence embedding space onto a line. This ratio gradually
relaxes in later layers and drops to 0.084 (English) and 0.074 (Hindi) at the
output layer. Gemma maintains lower variance concentration through the middle
depths (0.15 to 0.38), spikes at penultimate layer 23 to 0.530 (English) and
0.592 (Hindi), and drops to 0.046 (English) and 0.036 (Hindi) at output layer 24.

### Average Random Cosine Similarity

![Average Random Cosine Similarity](results/figures/avg_cosine_similarity_layerwise.png)

Values near zero mean sentence vectors point in nearly identical directions,
forming an acute directional cone. In Qwen, sentence vectors form an extreme cone
through layers 3 to 15 (values between 0.0003 and 0.005, corresponding to random
sentence cosine similarities above 0.995). Gemma tightens through the middle
layers and reaches peak collapse at layer 23 (0.0038 for English, 0.0030 for Hindi).
At the final layer, both models break the cone: Gemma jumps to 0.506 (English)
and 0.485 (Hindi), while Qwen surges to 0.462 (English) and 0.328 (Hindi). The
output layer is where angular dispersion is enforced for retrieval.

### ID Score

![ID Score](results/figures/id_score_layerwise.png)

Intrinsic dimensionality remains low throughout the network (2% to 6% of ambient
dimension), indicating that sentence embeddings inhabit a low-dimensional
manifold at every depth. Gemma displays slightly higher intrinsic dimensionality
than Qwen through intermediate layers (0.035 to 0.042 vs. 0.022 to 0.030).
English and Hindi exhibit closely aligned manifold dimensions across both models.

### PCA Compression (3D Point Cloud Progression)

For each model and language combination, we project the unit-normalized sentence
representations ($\mathbf{u} = \mathbf{s} / \|\mathbf{s}\|_2$) onto their top
three principal components across model depth within a fixed shared coordinate
frame ($[-0.35, 0.35]$ across all axes). This directly visualizes the directional
cone and isolates the phenomenon of final-layer normalization and contrastive
dispersion by deliberately pairing penultimate layer $N-1$ directly alongside
final output layer $N$ (layers 0, 6, 12, 17, 23, 24 for Gemma; layers 0, 7, 14,
20, 27, 28 for Qwen).

#### EmbeddingGemma (English)

![Gemma English PCA compression](results/BPCC_hin_Deva_google_embeddinggemma-300m_en/pca_compression.png)

The input layer displays an isotropic distribution with 3D spatial spread of
0.1185 and top-3 variance share of 19.6%. Intermediate layers contract steadily:
spread drops to 0.0756 at layer 6, 0.0496 at layer 12, and 0.0536 at layer 17.
At penultimate layer 23 ($N-1$, shown in red), directional collapse reaches its
peak, compressing into a tight pinpoint speck with a 3D spread of 0.0116. At
final layer 24 ($N$, shown in green), output normalization and contrastive
projection immediately explode this collapsed state back open into a wide,
isotropic volume (spread increases tenfold to 0.1171).

#### EmbeddingGemma (Hindi)

![Gemma Hindi PCA compression](results/BPCC_hin_Deva_google_embeddinggemma-300m_hi/pca_compression.png)

Hindi mirrors the English trajectory: a wide initial distribution at layer 0
(spread 0.1038), narrowing across layers 6 to 17 (spread dropping to 0.0641,
0.0441, and 0.0515). At penultimate layer 23 ($N-1$, red), representations
collapse into a tiny pinpoint knot with a minimal spread of 0.0102. At final
layer 24 ($N$, green), representations burst open into an expansive, spherical
cloud with spread surging to 0.1100.

#### Qwen3-Embedding (English)

![Qwen English PCA compression](results/BPCC_hin_Deva_Qwen_Qwen3-Embedding-0.6B_en/pca_compression.png)

Qwen displays extreme directional collapse in early-to-mid layers: at layer 7,
the entire sentence representation space collapses into a needle-like knot with
spread falling to 0.0108. The space gradually loosens through layers 14 to 20
(spread 0.0218 to 0.0451). Penultimate layer 27 ($N-1$, red) remains partially
compressed (spread 0.0540). At final layer 28 ($N$, green), the collapsed cone is
completely broken, decompressing into a full 3D distribution with spread
surging to 0.1527.

#### Qwen3-Embedding (Hindi)

![Qwen Hindi PCA compression](results/BPCC_hin_Deva_Qwen_Qwen3-Embedding-0.6B_hi/pca_compression.png)

Hindi sentence embeddings in Qwen follow the same trajectory: a tight knot at
layer 7 (spread 0.0265), progressing through layers 14 to 20 (spread 0.0396 to
0.0610), and holding at penultimate layer 27 ($N-1$, red, spread 0.0391). At
final layer 28 ($N$, green), the distribution decompresses into an expansive
volume with spread surging to 0.1258.

## Takeaways

- Both models compress sentence representations into a narrow, low-dimensional
  cone across their middle layers and then reopen the space at the output layer.
  This reflects contrastive fine-tuning objectives, which enforce hyperspherical
  uniformity at the output layer for cosine-similarity retrieval.
- Qwen exhibits acute single-axis collapse (MEV > 99.5%) in its early-to-middle
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
