"""Verification and pedagogical walkthrough script for EmbeddingGemma inference.

This script demonstrates and validates:
1. Deterministic seeding across random generators and compute backends.
2. Initialization of ModelConfig and hardware-accelerated model wrapper.
3. Batch encoding with masked mean pooling and L2 unit-norm normalization.
4. Layer-wise hidden state extraction across the entire model depth.
5. Semantic similarity evaluation: verifying that semantically equivalent
   sentences (paraphrases) yield higher cosine similarity than unrelated topics.
6. Memory-efficient generator streaming across mini-batch chunks.
"""

import torch

from embedding_gemma.config import ModelConfig
from embedding_gemma.model import EmbeddingGemmaWrapper
from embedding_gemma.utils import set_seed


def main() -> None:
    """Execute end-to-end inference verification and semantic sanity checks."""
    # Step 1: Enforce deterministic environment baseline.
    set_seed(42)

    # Step 2: Initialize configuration in query mode.
    # Query mode applies the prefix 'task: search result | query: {text}'.
    config = ModelConfig(task_type="query")
    wrapper = EmbeddingGemmaWrapper(config)

    print("=" * 60)
    print("EmbeddingGemma Pipeline Verification")
    print("=" * 60)
    print(f"Active compute device:   {wrapper.device}")
    print(f"Working numerical dtype: {wrapper.dtype}")
    print(f"Model identifier:        {config.model_id}")

    # Step 3: Define test sentences representing paraphrase vs unrelated text.
    # Sentence 0 and Sentence 1 are semantic paraphrases.
    # Sentence 2 is an unrelated sentence about quantum computing.
    test_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast auburn canid leaps above an inactive hound.",
        "Quantum computing relies on superposition and entanglement.",
    ]

    # Step 4: Run batched encoding with full layer extraction.
    output = wrapper.encode(test_sentences, return_hidden_states=True, to_cpu=True)

    print("\n--- Output Dimensionality ---")
    print(f"Final embedding tensor shape: {output.embeddings.shape}")
    assert output.embeddings.shape[0] == len(test_sentences)
    assert output.embeddings.shape[1] == wrapper.model.config.hidden_size

    assert output.layer_hidden_states is not None
    total_layers = len(output.layer_hidden_states)
    print(f"Total layer representations extracted: {total_layers}")
    print(f"Layer 0 (embedding) shape:  {output.layer_hidden_states[0].shape}")
    print(
        f"Layer {total_layers - 1} (final) shape:      "
        f"{output.layer_hidden_states[-1].shape}",
    )

    # Step 5: Evaluate semantic similarity via dot product (valid due to L2 norm).
    # Since ||u||_2 = 1 and ||v||_2 = 1, dot(u, v) equals cosine_similarity(u, v).
    sim_similar = torch.dot(output.embeddings[0], output.embeddings[1]).item()
    sim_dissimilar = torch.dot(output.embeddings[0], output.embeddings[2]).item()

    print("\n--- Semantic Similarity Sanity Check ---")
    print(f"Cosine similarity (paraphrase pair [0, 1]): {sim_similar:.4f}")
    print(f"Cosine similarity (unrelated pair  [0, 2]): {sim_dissimilar:.4f}")

    # Paraphrases must exhibit strictly higher similarity than unrelated topics.
    assert sim_similar > sim_dissimilar, (
        f"Expected paraphrase similarity ({sim_similar:.4f}) to exceed "
        f"unrelated similarity ({sim_dissimilar:.4f})"
    )

    # Step 6: Validate streaming generator interface.
    print("\n--- Streaming Interface Verification ---")
    stream_count = 0
    for batch_idx, batch_out in enumerate(
        wrapper.stream_encode(test_sentences, to_cpu=True),
    ):
        batch_size_curr = batch_out.embeddings.shape[0]
        stream_count += batch_size_curr
        print(f"  Batch {batch_idx}: yielded {batch_size_curr} embedding vector(s)")

    assert stream_count == len(test_sentences), (
        f"Streaming count mismatch: expected {len(test_sentences)}, got {stream_count}"
    )

    print("\n" + "=" * 60)
    print("All pipeline assertions passed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
