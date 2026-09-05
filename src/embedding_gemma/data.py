"""Dataset loading utilities for semantic similarity and representation benchmarks."""

from collections.abc import Sequence

import numpy as np
from datasets import load_dataset


def load_stsb_benchmark(
    split: str = "test",
) -> tuple[Sequence[str], Sequence[str], np.ndarray]:
    """Load the Semantic Textual Similarity Benchmark (STS-B) dataset split.

    STS-B provides pairs of sentences annotated with human similarity scores
    ranging from 0.0 (unrelated) to 5.0 (semantically identical).

    Args:
        split: Dataset split to load (e.g., "test", "validation", "train").

    Returns:
        A tuple of (sentence1_list, sentence2_list, gold_scores_array).

    Raises:
        RuntimeError: If dataset fails to load or required columns are missing.

    """
    dataset = load_dataset("mteb/stsbenchmark-sts", split=split)
    cols = dataset.column_names
    if "sentence1" not in cols or "sentence2" not in cols:
        msg = "Dataset missing required sentence columns ('sentence1', 'sentence2')"
        raise RuntimeError(msg)

    s1_list: list[str] = [str(s) for s in dataset["sentence1"]]
    s2_list: list[str] = [str(s) for s in dataset["sentence2"]]
    gold_scores: np.ndarray = np.asarray(dataset["score"], dtype=np.float64)

    return s1_list, s2_list, gold_scores

def load_bpcc_gold_standard(split: str = "hin_Deva") -> tuple[Sequence[str], Sequence[str], np.ndarray]:
    """Load the gold-standard human-verified subset of BPCC for a specific language split."""
    # Pass a valid language split like 'hin_Deva' instead of 'train'
    dataset = load_dataset("ai4bharat/BPCC", "daily", split=split)

    # Check your columns or extract source/target texts correctly depending on the dataset structure
    s1_list: list[str] = [str(row.get("en", row.get("source", list(row.values())[0]))) for row in dataset]
    s2_list: list[str] = [str(row.get("hi", row.get("target", list(row.values())[1]))) for row in dataset]

    gold_scores: np.ndarray = np.ones(len(s1_list), dtype=np.float64)

    return s1_list, s2_list, gold_scores
