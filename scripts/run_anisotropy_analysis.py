"""Layer-wise isotropy analysis of multilingual sentence embeddings on BPCC (en vs hi).

For each model (EmbeddingGemma, Qwen3-Embedding) and each language (English,
Hindi) drawn from a random parallel subset of BPCC, this script:

1. Collects pooled sentence embeddings after every transformer layer.
2. Computes IsoScore, Average Random Cosine Similarity, ID Score, MEV.
3. Saves the per-layer sentence embeddings and metrics.
4. Renders four metric-vs-depth comparison plots (Gemma vs Qwen, en vs hi) and a
   3D PCA-compression plot per (model, language).
"""

import csv
import gc
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from tabulate import tabulate

from embedding_gemma.config import ModelConfig
from embedding_gemma.data import load_bpcc_parallel
from embedding_gemma.geometry import LayerIsotropyRecord, analyze_layer_clouds
from embedding_gemma.model import EmbeddingGemmaWrapper
from embedding_gemma.utils import set_seed

# ------------------------------------------------------------------ config ----
LANGUAGE_SPLIT = "hin_Deva"
MAX_SAMPLES = 10_000
TOKEN_CAP = 10_000
# Metrics use the full TOKEN_CAP cloud; only a subsample is written to disk to
# keep the saved embeddings from ballooning (Qwen is 4096-dim, saved as float32).
EMBED_SAVE_CAP = 2_000
# Cap sequence length: wiki sentences are short, and this bounds hidden-state
# memory so the 8B model fits a laptop GPU.
MAX_LENGTH = 128
SEED = 42

MODELS: dict[str, str] = {
    "Gemma": "google/embeddinggemma-300m",
    "Qwen": "Qwen/Qwen3-Embedding-0.6B",
}

# Both models are small enough for a comfortable batch size on a laptop GPU.
BATCH_SIZES: dict[str, int] = {"Gemma": 32, "Qwen": 32}

# The four isotropy metrics: (record attribute, human label, higher-is-*).
METRICS: list[tuple[str, str, str]] = [
    ("isoscore", "IsoScore", "higher = more isotropic"),
    (
        "avg_cosine_similarity",
        "Avg Random Cosine Similarity",
        "higher = more isotropic",
    ),
    ("id_score", "ID Score (normalized)", "higher = higher intrinsic dim"),
    ("mev", "MEV", "lower = less rogue dominance"),
]

# Distinct style per (model, language) series for the comparison plots.
SERIES_STYLE: dict[tuple[str, str], dict[str, str]] = {
    ("Gemma", "en"): {"color": "#1f77b4", "linestyle": "-", "marker": "o"},
    ("Gemma", "hi"): {"color": "#1f77b4", "linestyle": "--", "marker": "s"},
    ("Qwen", "en"): {"color": "#d62728", "linestyle": "-", "marker": "^"},
    ("Qwen", "hi"): {"color": "#d62728", "linestyle": "--", "marker": "d"},
}

LANGUAGE_LABELS = {"en": "English", "hi": "Hindi"}


def save_metrics(records: list[LayerIsotropyRecord], output_dir: Path) -> None:
    """Persist per-layer isotropy metrics as JSON and CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dict_records = [r.to_dict() for r in records]

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(dict_records, f, indent=2)

    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict_records[0].keys()))
        writer.writeheader()
        writer.writerows(dict_records)


def save_embeddings(layer_clouds: list[np.ndarray], output_dir: Path) -> None:
    """Save per-layer token point clouds as a compressed float16 npz archive.

    A fixed random subsample (shared across layers) is saved to bound disk usage;
    the full clouds are still used for the metrics themselves.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    n_tokens = layer_clouds[0].shape[0]
    if n_tokens > EMBED_SAVE_CAP:
        rng = np.random.default_rng(SEED)
        keep = rng.choice(n_tokens, size=EMBED_SAVE_CAP, replace=False)
        layer_clouds = [cloud[keep] for cloud in layer_clouds]

    # float32 (not float16): some models overflow float16's range, which would
    # write inf into the saved archive.
    arrays = {
        f"layer_{i:02d}": cloud.astype(np.float32)
        for i, cloud in enumerate(layer_clouds)
    }
    # ty flags **dict unpacking against numpy's allow_pickle:bool stub param.
    np.savez_compressed(output_dir / "embeddings.npz", **arrays)  # ty: ignore[invalid-argument-type]


def print_metric_table(records: list[LayerIsotropyRecord]) -> None:
    """Print a per-layer metric summary table to the console."""
    table = [
        {
            "Layer": r.layer_name,
            "IsoScore": f"{r.isoscore:.4f}",
            "AvgCos": f"{r.avg_cosine_similarity:.4f}",
            "ID Score": f"{r.id_score:.4f}",
            "MEV": f"{r.mev:.4f}",
        }
        for r in records
    ]
    print("\n" + tabulate(table, headers="keys", tablefmt="github"))


def plot_metric_comparison(
    all_records: dict[tuple[str, str], list[LayerIsotropyRecord]],
    attribute: str,
    label: str,
    note: str,
    output_path: Path,
) -> None:
    """Plot one metric vs normalized layer depth across all model/language series.

    Normalized depth (layer index / final layer) aligns models of different
    depth (Gemma has fewer layers than Qwen) on a shared x-axis.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _fig, ax = plt.subplots(figsize=(9, 6), dpi=200)

    for (model_label, lang), records in all_records.items():
        depth = len(records) - 1
        xs = [r.layer_index / depth for r in records] if depth > 0 else [0.0]
        ys = [getattr(r, attribute) for r in records]
        style = SERIES_STYLE[(model_label, lang)]
        ax.plot(
            xs,
            ys,
            linewidth=2,
            markersize=5,
            label=f"{model_label} - {LANGUAGE_LABELS.get(lang, lang)}",
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
        )

    ax.set_title(f"{label} across Layer Depth", fontsize=13, fontweight="bold")
    ax.set_xlabel("Normalized Layer Depth (0 = input, 1 = final)", fontsize=11)
    ax.set_ylabel(f"{label}\n({note})", fontsize=11)
    ax.grid(visible=True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path)
    if output_path.suffix.lower() != ".pdf":
        plt.savefig(output_path.with_suffix(".pdf"))
    plt.close()


def plot_pca_compression(
    layer_clouds: list[np.ndarray],
    model_label: str,
    lang: str,
    output_path: Path,
) -> None:
    """Plot 3D PCA projections of the token cloud at layers across model depth.

    Visualizes how the representation cloud compresses (collapses toward a narrow
    region) as depth increases, explicitly displaying the penultimate layer (N-1)
    alongside the final layer (N) to demonstrate output-layer normalization and
    contrastive dispersion.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_layers = len(layer_clouds)
    # Include depths through penultimate layer (N-1), plus the final layer (N).
    intermediate = np.round(np.linspace(0, num_layers - 2, 5)).astype(int)
    selected = np.append(intermediate, num_layers - 1)

    fig = plt.figure(figsize=(15, 9), dpi=200)

    for panel, layer_idx in enumerate(selected):
        ax = fig.add_subplot(2, 3, panel + 1, projection="3d")
        cloud = layer_clouds[layer_idx].astype(np.float32)

        # Unit-normalize vectors onto the hypersphere to evaluate directional cone
        norms = np.linalg.norm(cloud, axis=1, keepdims=True)
        u = cloud / np.maximum(norms, 1e-12)

        pca = PCA(n_components=3, svd_solver="randomized", random_state=SEED)
        proj = pca.fit_transform(u - u.mean(axis=0))
        var_share = float(pca.explained_variance_ratio_[:3].sum())
        spread = float(np.std(proj, axis=0).mean())

        color = (
            "#d62728"
            if layer_idx == num_layers - 2
            else ("#2ca02c" if layer_idx == num_layers - 1 else "#1f77b4")
        )
        ax.scatter(
            proj[:, 0],
            proj[:, 1],
            proj[:, 2],
            s=5,
            alpha=0.30,
            c=color,
            edgecolors="none",
        )
        ax.set_xlim([-0.35, 0.35])
        ax.set_ylim([-0.35, 0.35])
        ax.set_zlim([-0.35, 0.35])

        if layer_idx == num_layers - 1:
            title = f"Layer {layer_idx} (Final N)"
        elif layer_idx == num_layers - 2:
            title = f"Layer {layer_idx} (N-1)"
        else:
            title = f"Layer {layer_idx}"
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("PC 1", fontsize=8, labelpad=1)
        ax.set_ylabel("PC 2", fontsize=8, labelpad=1)
        ax.set_zlabel("PC 3", fontsize=8, labelpad=1)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=20, azim=-60)
        ax.text2D(
            0.03,
            0.95,
            f"Spread: {spread:.4f}\nTop-3 var: {var_share:.1%}",
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.85},
        )

    plt.suptitle(
        f"{model_label} ({LANGUAGE_LABELS.get(lang, lang)}): "
        "3D PCA Compression of Sentence Embeddings across Depth",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    plt.savefig(output_path)
    if output_path.suffix.lower() != ".pdf":
        plt.savefig(output_path.with_suffix(".pdf"))
    plt.close()


def main() -> None:
    """Run the BPCC layer-wise isotropy analysis for both models and languages."""
    set_seed(SEED)

    output_dir = Path("results")
    figures_dir = output_dir / "figures"

    print("=" * 70)
    print("BPCC Layer-Wise Isotropy Analysis (English vs Hindi)")
    print("=" * 70)

    print(f"\n[Data] Loading BPCC wiki/{LANGUAGE_SPLIT}.tsv ...")
    english, hindi = load_bpcc_parallel(
        language_split=LANGUAGE_SPLIT,
        max_samples=MAX_SAMPLES,
        seed=SEED,
    )
    print(f"     Loaded {len(english)} parallel English-Hindi pairs.")
    texts_by_lang = {"en": english, "hi": hindi}

    all_records: dict[tuple[str, str], list[LayerIsotropyRecord]] = {}

    for model_label, model_id in MODELS.items():
        print("\n" + "-" * 70)
        print(f"Model: {model_label} ({model_id})")
        print("-" * 70)

        config = ModelConfig(
            model_id=model_id,
            task_type="raw",
            batch_size=BATCH_SIZES[model_label],
            max_length=MAX_LENGTH,
        )
        try:
            wrapper = EmbeddingGemmaWrapper(config)
        except Exception as exc:  # noqa: BLE001
            print(f"     [Error] Failed to initialize {model_id}: {exc}")
            continue

        print(f"     Compute device:  {wrapper.device}")
        print(f"     Numerical dtype: {wrapper.dtype}")

        for lang, texts in texts_by_lang.items():
            print(
                f"\n  >> {model_label} / {LANGUAGE_LABELS[lang]}: "
                "collecting sentence embeddings ..."
            )
            layer_clouds = wrapper.collect_layer_sentence_embeddings(texts)
            print(
                f"     Layers: {len(layer_clouds)} | "
                f"sentences/layer: {layer_clouds[0].shape[0]} | "
                f"dim: {layer_clouds[0].shape[1]}"
            )

            print("     Computing isotropy metrics ...")
            records = analyze_layer_clouds(layer_clouds)
            all_records[(model_label, lang)] = records

            safe_model = model_id.replace("/", "_")
            run_dir = output_dir / f"BPCC_{LANGUAGE_SPLIT}_{safe_model}_{lang}"

            save_metrics(records, run_dir)
            save_embeddings(layer_clouds, run_dir)
            plot_pca_compression(
                layer_clouds,
                model_label,
                lang,
                run_dir / "pca_compression.png",
            )
            print_metric_table(records)
            print(f"     Saved metrics, embeddings, PCA plot -> {run_dir}")

            del layer_clouds
            gc.collect()

        del wrapper
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # ------------------------------------------------ comparison figures ----
    if all_records:
        print("\n[Plot] Rendering metric comparison figures ...")
        for attribute, label, note in METRICS:
            plot_metric_comparison(
                all_records,
                attribute,
                label,
                note,
                figures_dir / f"{attribute}_layerwise.png",
            )
        print(f"     Saved 4 metric comparison plots -> {figures_dir}")

    print("\n" + "=" * 70)
    print("Analysis complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
