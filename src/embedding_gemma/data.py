"""Dataset loading utilities for parallel multilingual representation benchmarks."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

BPCC_REPO_ID: str = "ai4bharat/BPCC"


def load_bpcc_parallel(
    language_split: str = "hin_Deva",
    max_samples: int = 10000,
    seed: int = 42,
) -> tuple[Sequence[str], Sequence[str]]:
    """Load a random parallel English-target subset from a BPCC wiki TSV file.

    BPCC exposes per-language parallel data as tab-separated files under
    ``wiki/<language_split>.tsv``. Each file has a header row with columns
    ``src_lang, tgt_lang, src, tgt`` where ``src`` is the English sentence and
    ``tgt`` is the aligned sentence in the target Indic language. Row ``i`` of
    ``src`` and ``tgt`` are translations of one another (a parallel pair).

    Args:
        language_split: BPCC language code, e.g. ``"hin_Deva"`` for Hindi.
        max_samples: Number of parallel pairs to sample uniformly at random.
        seed: Random seed controlling the reproducible subset selection.

    Returns:
        A tuple ``(english_sentences, target_sentences)`` of equal length, where
        element ``i`` in each list is a translation of the other.

    Raises:
        RuntimeError: If the downloaded file lacks the expected ``src``/``tgt``
            columns.

    """
    local_path = hf_hub_download(
        repo_id=BPCC_REPO_ID,
        repo_type="dataset",
        filename=f"wiki/{language_split}.tsv",
    )

    frame = pd.read_csv(
        local_path,
        sep="\t",
        header=0,
        dtype=str,
        keep_default_na=False,
        on_bad_lines="skip",
        quoting=3,  # csv.QUOTE_NONE: BPCC text contains unbalanced quotes.
    )

    if "src" not in frame.columns or "tgt" not in frame.columns:
        msg = (
            f"BPCC file wiki/{language_split}.tsv missing 'src'/'tgt' columns; "
            f"found {list(frame.columns)}"
        )
        raise RuntimeError(msg)

    # Drop rows with empty text on either side to keep pairs well-formed.
    frame = frame[(frame["src"].str.len() > 0) & (frame["tgt"].str.len() > 0)]

    english: list[str] = frame["src"].astype(str).tolist()
    target: list[str] = frame["tgt"].astype(str).tolist()

    # Uniform random subset without replacement, order preserved for stability.
    rng = np.random.default_rng(seed)
    n_available = len(english)
    n_select = min(max_samples, n_available)
    indices = np.sort(rng.choice(n_available, size=n_select, replace=False))

    english = [english[i] for i in indices]
    target = [target[i] for i in indices]

    return english, target
