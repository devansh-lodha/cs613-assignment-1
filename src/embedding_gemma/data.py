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
