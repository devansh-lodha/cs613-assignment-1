"""Layer-wise geometric representation and anisotropy analysis on STS-B."""

import csv
import json
from pathlib import Path
import gc
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tabulate import tabulate

from embedding_gemma.config import ModelConfig
from embedding_gemma.data import load_stsb_benchmark, load_bpcc_gold_standard
from embedding_gemma.geometry import (
    LayerGeometryRecord,
    analyze_layer_geometry,
)
from embedding_gemma.model import EmbeddingGemmaWrapper, EmbeddingOutput
from embedding_gemma.utils import set_seed



def save_numerical_results(
    records: list[LayerGeometryRecord],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Persist quantitative geometry records to JSON and CSV formats.

    Args:
        records: List of LayerGeometryRecord objects.
        output_dir: Directory where files will be written.

    Returns:
        Tuple of (json_path, csv_path).

    """
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "stsb_layer_geometry.json"
    csv_path = output_dir / "stsb_layer_geometry.csv"

    dict_records = [r.to_dict() for r in records]

    # Save JSON artifact
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(dict_records, f, indent=2)

    # Save CSV artifact
    fieldnames = list(dict_records[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dict_records)

    return json_path, csv_path


def generate_trajectory_plots(
    records: list[LayerGeometryRecord],
    output_path: Path,
) -> None:
    """Generate and save dual-panel visualization of representation geometry.

    Panel 1: Pairwise Cosine Anisotropy (The Cone Angle) and IsoScore.
    Panel 2: Top Singular Value Variance Share and STS-B Spearman Correlation.

    Args:
        records: List of LayerGeometryRecord across depth.
        output_path: Destination image filepath.

    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    layers = [r.layer_index for r in records]
    anisotropy = [r.cosine_anisotropy for r in records]
    isoscore = [r.isoscore for r in records]
    rogue_ratio = [r.rogue_ratio for r in records]
    spearman = [r.spearman_correlation for r in records]

    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)

    # Panel 1: Directional Cone Formation & IsoScore
    ax1.plot(
        layers,
        anisotropy,
        marker="o",
        linewidth=2,
        color="#d62728",
        label="Cosine Anisotropy (The Cone)",
    )
    ax1.plot(
        layers,
        isoscore,
        marker="s",
        linewidth=2,
        color="#1f77b4",
        label="IsoScore (Isotropy)",
    )
    ax1.set_title(
        "Layer-Wise Cone Formation & Dimensional Variance",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_xlabel("Layer Depth (0 = Input Embeddings)", fontsize=11)
    ax1.set_ylabel("Metric Value", fontsize=11)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(visible=True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower left", framealpha=0.9)

    # Annotate final layer contrastive repair
    final_layer = layers[-1]
    repair_text = (
        f"Contrastive Repair\n"
        f"Anisotropy: {anisotropy[-1]:.2f}\n"
        f"IsoScore: {isoscore[-1]:.2f}"
    )
    ax1.annotate(
        repair_text,
        xy=(final_layer, anisotropy[-1]),
        xytext=(final_layer - 8, 0.25),
        arrowprops={"facecolor": "black", "shrink": 0.08, "width": 1, "headwidth": 6},
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "fc": "#ffffdd", "ec": "#999999"},
    )

    # Panel 2: Rogue Dimension Variance & Semantic Correlation
    ax2.plot(
        layers,
        rogue_ratio,
        marker="^",
        linewidth=2,
        color="#ff7f0e",
        label=r"Top Singular Value Variance Share ($\sigma_1^2 / \sum \sigma^2$)",
    )
    ax2.plot(
        layers,
        spearman,
        marker="d",
        linewidth=2,
        color="#2ca02c",
        label=r"STS-B Spearman Rank Correlation ($\rho$)",
    )
    ax2.set_title(
        "Rogue Dimension Dominance vs Downstream STS-B Performance",
        fontsize=12,
        fontweight="bold",
    )
    ax2.set_xlabel("Layer Depth (0 = Input Embeddings)", fontsize=11)
    ax2.set_ylabel("Metric Value", fontsize=11)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(visible=True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", framealpha=0.9)

    # Annotate peak semantic correlation
    peak_text = f"Peak Performance\nSpearman: {spearman[-1]:.4f}"
    ax2.annotate(
        peak_text,
        xy=(final_layer, spearman[-1]),
        xytext=(final_layer - 7, 0.40),
        arrowprops={"facecolor": "black", "shrink": 0.08, "width": 1, "headwidth": 6},
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "fc": "#e6ffe6", "ec": "#66cc66"},
    )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def generate_3d_geometry_plots(
    output_s1: EmbeddingOutput,
    output_s2: EmbeddingOutput,
    output_path: Path,
    selected_layers: tuple[int, ...] = None,
) -> None:
    if output_s1.layer_hidden_states is None or output_s2.layer_hidden_states is None:
        return

    num_layers = len(output_s1.layer_hidden_states)

    # Dynamically pick layers if not specified or out of bounds
    if selected_layers is None or max(selected_layers) >= num_layers:
        # Pick 6 evenly spaced layers across whatever model depth we have
        selected_layers = tuple(np.linspace(0, num_layer - 1, 6, dtype=int)) if (num_layer := num_layers) else (0, 1, 2, 3, 4, 5)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(18, 10), dpi=300)

    for idx, l_idx in enumerate(selected_layers):
        if l_idx >= num_layers:
            continue
        ax = fig.add_subplot(2, 3, idx + 1, projection="3d")
        title = f"Layer {l_idx} (Total: {num_layers})"

        h = torch.cat(
            [
                output_s1.layer_hidden_states[l_idx],
                output_s2.layer_hidden_states[l_idx],
            ],
            dim=0,
        )
        h_norm = F.normalize(h.float(), p=2, dim=-1)

        mean = h_norm.mean(dim=0, keepdim=True)
        centered = h_norm - mean
        _u, _s, v = torch.pca_lowrank(centered, q=3)
        proj = torch.mm(centered, v[:, :3]).numpy()

        ax.scatter(
            proj[:, 0], proj[:, 1], proj[:, 2],
            alpha=0.35, s=12, c="#1f77b4", edgecolors="none",
        )
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.set_xlim([-0.4, 0.4])
        ax.set_ylim([-0.4, 0.4])
        ax.set_zlim([-0.4, 0.4])
        ax.set_xlabel("PC 1", fontsize=9, labelpad=2)
        ax.set_ylabel("PC 2", fontsize=9, labelpad=2)
        ax.set_zlabel("PC 3", fontsize=9, labelpad=2)
        ax.tick_params(labelsize=7)

        spread = float(np.std(proj, axis=0).mean())
        ax.text2D(
            0.05,
            0.90,
            f"3D Std Spread: {spread:.4f}",
            transform=ax.transAxes,
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.85},
        )

    plt.suptitle(
        "EmbeddingGemma Layer-Wise 3D Representation Geometry Progression",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def generate_cosine_distribution_plots(
    output_s1: EmbeddingOutput,
    output_s2: EmbeddingOutput,
    output_path: Path,
    selected_layers: tuple[int, ...] = (0, 4, 10, 16, 23, 24),
) -> None:
    """Generate multi-panel histograms of pairwise cosine similarities across depth.

    Args:
        output_s1: EmbeddingOutput for sentence 1 corpus.
        output_s2: EmbeddingOutput for sentence 2 corpus.
        output_path: Destination image filepath.
        selected_layers: Tuple of layer indices to display.

    """
    if output_s1.layer_hidden_states is None or output_s2.layer_hidden_states is None:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    layer_titles = [
        "Layer 0 (Embeddings)",
        "Layer 4 (Early Cone)",
        "Layer 10 (Mid Depth)",
        "Layer 16 (Late Depth)",
        "Layer 23 (Peak Cone)",
        "Layer 24 (Contrastive Repair)",
    ]

    _fig, axes = plt.subplots(2, 3, figsize=(16, 8), dpi=300)
    flat_axes = axes.flatten()

    for idx, (l_idx, title) in enumerate(
        zip(selected_layers, layer_titles, strict=False),
    ):
        ax = flat_axes[idx]
        h1 = output_s1.layer_hidden_states[l_idx]
        h2 = output_s2.layer_hidden_states[l_idx]
        h1_norm = F.normalize(h1.float(), p=2, dim=-1)
        h2_norm = F.normalize(h2.float(), p=2, dim=-1)

        sims = (h1_norm * h2_norm).sum(dim=-1).numpy()

        color = "#d62728" if l_idx == 23 else ("#2ca02c" if l_idx == 24 else "#1f77b4")
        ax.hist(
            sims,
            bins=40,
            range=(-0.2, 1.0),
            color=color,
            alpha=0.75,
            edgecolor="black",
            linewidth=0.5,
            density=True,
        )

        mean_sim = float(np.mean(sims))
        ax.axvline(
            mean_sim,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"Mean: {mean_sim:.3f}",
        )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Pairwise Cosine Similarity", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_xlim([-0.2, 1.05])
        ax.grid(visible=True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper left", fontsize=9)

    plt.suptitle(
        "EmbeddingGemma Layer-Wise Pairwise Cosine Similarity Distribution Dynamics",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    """Execute complete layer-wise representation geometry pipeline across multiple models and datasets."""
    set_seed(42)

    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Multi-Model & Multi-Dataset Layer-Wise Representation Analysis")
    print("=" * 70)

    # Define the 2x2 experimental matrix requested by your teammate
    models_to_test = [
    "Qwen/Qwen3-Embedding-8B",
    "google/embeddinggemma-300m",
    "bert-base-multilingual-cased"
    ]
    datasets_to_test = {
        "STS-B": load_stsb_benchmark,
        "BPCC-Human": load_bpcc_gold_standard,
    }

    for data_name, data_loader in datasets_to_test.items():
        print(f"\n[Data] Loading {data_name} dataset...")
        try:
            s1_list, s2_list, gold_scores = data_loader()
        except Exception as e:
            print(f"     [Warning] Could not load {data_name}: {e}. Skipping.")
            continue

        print(f"     Loaded {len(s1_list)} sentence pairs.")

        # Optional: Slice dataset to 1000 pairs during initial testing to save time/compute
        s1_list, s2_list, gold_scores = s1_list[:1000], s2_list[:1000], gold_scores[:1000]

        for model_id in models_to_test:
            print(f"\n----------------------------------------------------------------------")
            print(f"Evaluating Model: {model_id} on Dataset: {data_name}")
            print(f"----------------------------------------------------------------------")

            # Initialize config (ModelConfig is frozen, pass model_id on init)
            config = ModelConfig(model_id=model_id, task_type="raw", batch_size=8)

            try:
                wrapper = EmbeddingGemmaWrapper(config)
            except Exception as e:
                print(f"     [Error] Failed to initialize {model_id}: {e}")
                continue

            print(f"     Compute device:   {wrapper.device}")
            print(f"     Numerical dtype:  {wrapper.dtype}")

            # Extract representations
            print("     Encoding sentence 1 corpus...")
            out1 = wrapper.encode(s1_list, return_hidden_states=True, to_cpu=True)
            print("     Encoding sentence 2 corpus...")
            out2 = wrapper.encode(s2_list, return_hidden_states=True, to_cpu=True)

            # Compute metrics
            print("     Computing layer-wise geometry metrics...")
            records = analyze_layer_geometry(out1, out2, gold_scores)

            # Create distinct subdirectories for results to prevent file overwrites
            safe_model_name = model_id.replace("/", "_")
            run_out_dir = output_dir / f"{data_name}_{safe_model_name}"
            run_out_dir.mkdir(parents=True, exist_ok=True)

            # Display table summary
            table_data = [
                {
                    "Layer": r.layer_name,
                    "Cosine Cone (↓)": f"{r.cosine_anisotropy:.4f}",
                    "STS-B Spearman (↑)": f"{r.spearman_correlation:.4f}",
                    "IsoScore (↑)": f"{r.isoscore:.4f}",
                    "ID Score (↑)": f"{r.id_score:.4f}",
                    "Rogue λ₁ Share (↓)": f"{r.rogue_ratio:.4f}",
                }
                for r in records
            ]
            print("\n" + tabulate(table_data, headers="keys", tablefmt="github"))

            # Save quantitative and qualitative outputs
            json_path, csv_path = save_numerical_results(records, run_out_dir)
            print(f"\nSaved artifacts to: {run_out_dir}")

            # Only generate these specific multi-layer progression plots for Gemma
            # (since BERT has 12 layers instead of 24, avoiding index out of range)
            if "embeddinggemma" in model_id.lower():
                generate_trajectory_plots(records, run_out_dir / "cone_analysis.png")
                generate_3d_geometry_plots(out1, out2, run_out_dir / "cone_3d_progression.png")
                generate_cosine_distribution_plots(out1, out2, run_out_dir / "cosine_distributions.png")
            else:
                print("     [Notice] Skipping 24-layer specific trajectory plots for non-Gemma control model.")

                # Clear VRAM before loading the next model
                del wrapper
                del out1
                del out2
                torch.cuda.empty_cache()


    print("\n" + "=" * 70)
    print("Multi-model analysis execution completed successfully.")
    print("=" * 70)

if __name__ == "__main__":
    main()
