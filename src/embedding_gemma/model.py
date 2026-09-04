"""Core model wrapper and inference engine for EmbeddingGemma."""

import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from embedding_gemma.config import ModelConfig
from embedding_gemma.device import get_optimal_device, get_optimal_dtype


@dataclass
class EmbeddingOutput:
    """Container holding final pooled embeddings and optional layer states.

    Attributes:
        embeddings: Tensor of shape (batch_size, hidden_dimension) representing the
            pooled (and optionally L2-normalized) representations from the final layer.
        layer_hidden_states: Optional list of tensors, where each element corresponds
            to a model layer (from embedding layer 0 to final layer L) with pooled
            representations of shape (batch_size, hidden_dimension). Used for geometric
            representation trajectory analysis across depth.

    """

    embeddings: torch.Tensor
    layer_hidden_states: list[torch.Tensor] | None = None


class EmbeddingGemmaWrapper:
    """Inference wrapper for Google's EmbeddingGemma representation models.

    This wrapper encapsulates:
    1. Hardware accelerator and numerical precision dispatch (CUDA / MPS / CPU).
    2. Asymmetric task formatting for retrieval queries and corpus documents.
    3. Attention-masked mean pooling over contextual token representations.
    4. Unit hypersphere L2 projection for cosine similarity calculations.
    5. Multi-layer hidden state extraction for geometric representation analysis.
    6. Both batched in-memory encoding and streaming generator evaluation.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Initialize the model wrapper, tokenizer, and underlying neural network.

        Args:
            config: ModelConfig containing hyperparameters, templates, and task mode.

        Raises:
            RuntimeError: If tokenizer or model fails to load from Hugging Face Hub.

        """
        self.config = config
        self.device = get_optimal_device()
        self.dtype = get_optimal_dtype(self.device)

        # Retrieve Hugging Face authentication token if available in the environment.
        # When None, transformers falls back to ~/.cache/huggingface/token.
        token = os.getenv("HF_TOKEN")

        # Initialize tokenizer with right-padding for causal decoder architectures.
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            token=token,
            padding_side="right",
        )
        if tokenizer is None:
            msg = f"Failed to load tokenizer for model ID: {config.model_id}"
            raise RuntimeError(msg)
        self.tokenizer = tokenizer

        # Ensure a valid pad token is registered in the tokenizer vocabulary.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
            pad_str = self.tokenizer.convert_ids_to_tokens(0)
            if isinstance(pad_str, str):
                self.tokenizer.pad_token = pad_str

        # Load transformer weights directly in target precision and transfer to device.
        model = AutoModel.from_pretrained(
            config.model_id,
            dtype=self.dtype,
            attn_implementation=config.attn_implementation,
            token=token,
        )
        if model is None:
            msg = f"Failed to load model for model ID: {config.model_id}"
            raise RuntimeError(msg)

        self.model = model.to(self.device)
        self.model.eval()

        # Synchronize pad token identifier between tokenizer and model configuration.
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def format_text(self, text: str) -> str:
        """Apply task-specific instruction prefixes for the configured task type.

        EmbeddingGemma uses asymmetric prompt structures:
        - Query mode: Formats as 'task: {description} | query: {text}' to guide the
          encoder on the target retrieval objective.
        - Document mode: Formats as 'title: {title} | text: {text}' to incorporate
          title metadata into passage representations.
        - Raw mode: Passes text verbatim without prompt wrapping.

        Args:
            text: Raw input string.

        Returns:
            Formatted prompt string ready for tokenization.

        """
        if self.config.task_type == "query":
            return self.config.query_template.format(
                description=self.config.task_description,
                text=text,
            )
        if self.config.task_type == "document":
            return self.config.document_template.format(
                title=self.config.document_title,
                text=text,
            )
        return text

    def _mean_pool(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute masked mean pooling across token representations.

        Given token representations H in R^(B x L x D) and an attention mask
        M in {0, 1}^(B x L), the pooled representation e_b for sequence b is:
            e_b = sum_{i=1}^L (M_{b,i} * H_{b,i}) / sum_{i=1}^L M_{b,i}

        Computations are accumulated in float32 to prevent numerical underflow or
        loss of precision, then cast back to the target precision.

        Args:
            token_embeddings: Token hidden representations of shape (batch, len, dim).
            attention_mask: Binary mask of shape (batch, len) with 1 for real tokens
                and 0 for padding tokens.

        Returns:
            Mean-pooled tensor of shape (batch, dim).

        """
        token_embeddings_f32 = token_embeddings.to(torch.float32)
        # Expand mask from (B, L) to (B, L, D) to broadcast over hidden dimension.
        mask_expanded = (
            attention_mask.unsqueeze(-1)
            .expand(token_embeddings.size())
            .to(torch.float32)
        )
        # Sum token vectors over the sequence length axis ignoring padding positions.
        sum_embeddings = torch.sum(token_embeddings_f32 * mask_expanded, dim=1)
        # Count non-padded tokens per sequence, clamped to prevent division by zero.
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled = sum_embeddings / sum_mask
        return pooled.to(self.dtype)

    def _normalize_if_enabled(self, tensor: torch.Tensor) -> torch.Tensor:
        """Project vectors onto the unit Euclidean sphere (L2 norm = 1.0) if enabled.

        When embeddings are L2-normalized:
            ||u||_2 = 1 and ||v||_2 = 1 ==> cosine_similarity(u, v) = u . v
        This allows cosine similarity comparisons to be executed via simple matrix
        multiplication (dot products) without subsequent division by norms.

        Args:
            tensor: Input embeddings tensor of shape (batch, dim).

        Returns:
            L2-normalized tensor if config.normalize_embeddings is True, else input.

        """
        if self.config.normalize_embeddings:
            return F.normalize(tensor.float(), p=2, dim=-1).to(self.dtype)
        return tensor

    def _tokenize_batch(self, batch_texts: Sequence[str]) -> dict[str, Any]:
        """Format and tokenize a batch of raw strings into device tensors.

        Args:
            batch_texts: Sequence of input text strings.

        Returns:
            Dictionary containing tokenized 'input_ids' and 'attention_mask' tensors
            transferred to self.device.

        """
        formatted = [self.format_text(t) for t in batch_texts]
        encoded = self.tokenizer(
            formatted,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in encoded.items()}

    def _extract_layer_representations(
        self,
        hidden_states: tuple[torch.Tensor, ...],
        attention_mask: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Extract and mean-pool intermediate representations across all model layers.

        Args:
            hidden_states: Tuple of hidden state tensors from each layer.
            attention_mask: Attention mask of shape (batch, seq_len).

        Returns:
            List of pooled layer tensors of shape (batch, dim), one per model layer.

        """
        layer_outputs: list[torch.Tensor] = []
        for layer_tensor in hidden_states:
            layer_pooled = self._mean_pool(layer_tensor, attention_mask)
            layer_normalized = self._normalize_if_enabled(layer_pooled)
            layer_outputs.append(layer_normalized)
        return layer_outputs

    def _process_batch(
        self,
        batch_texts: Sequence[str],
        *,
        return_hidden_states: bool,
    ) -> EmbeddingOutput:
        """Execute complete inference pipeline for a single batch of text inputs.

        Pipeline stages:
        1. Tokenize batch and transfer tensors to compute device.
        2. Execute transformer forward pass with optional hidden state tracking.
        3. Mean-pool final layer hidden states over valid attention tokens.
        4. Apply L2 normalization if configured.
        5. Optionally mean-pool and normalize intermediate layer representations.

        Args:
            batch_texts: Sequence of text inputs for the current batch.
            return_hidden_states: Whether to extract intermediate layer states.

        Returns:
            EmbeddingOutput containing pooled embeddings and optional layer states.

        """
        inputs = self._tokenize_batch(batch_texts)

        output = self.model(
            **inputs,
            output_hidden_states=return_hidden_states,
        )

        # Pool final hidden states into a single vector per sequence.
        final_pooled = self._mean_pool(
            output.last_hidden_state,
            inputs["attention_mask"],
        )
        final_embeddings = self._normalize_if_enabled(final_pooled)

        # Extract intermediate layer states if requested.
        pooled_layers: list[torch.Tensor] | None = None
        if return_hidden_states and output.hidden_states is not None:
            pooled_layers = self._extract_layer_representations(
                output.hidden_states,
                inputs["attention_mask"],
            )

        return EmbeddingOutput(
            embeddings=final_embeddings,
            layer_hidden_states=pooled_layers,
        )

    def encode(
        self,
        texts: Sequence[str],
        *,
        return_hidden_states: bool = False,
        to_cpu: bool = False,
    ) -> EmbeddingOutput:
        """Encode a collection of texts in mini-batches and return contiguous tensors.

        This method iterates through input texts in increments of `config.batch_size`,
        processes each batch under `torch.inference_mode()` (disabling gradient and view
        tracking overhead), and aggregates results into contiguous tensors.

        Args:
            texts: Sequence of text strings to encode.
            return_hidden_states: Whether to return representations across all layers.
            to_cpu: If True, moves tensors to CPU RAM immediately after batch
                computation to prevent GPU/MPS VRAM exhaustion on large datasets.

        Returns:
            EmbeddingOutput containing full concatenated embeddings and layer states.

        """
        # Handle empty input edge case gracefully.
        if not texts:
            empty_emb = torch.empty(
                (0, self.model.config.hidden_size),
                dtype=self.dtype,
            )
            if not to_cpu:
                empty_emb = empty_emb.to(self.device)
            return EmbeddingOutput(
                embeddings=empty_emb,
                layer_hidden_states=[] if return_hidden_states else None,
            )

        collected_embeddings: list[torch.Tensor] = []
        collected_layers: list[list[torch.Tensor]] | None = (
            [] if return_hidden_states else None
        )

        with torch.inference_mode():
            for i in range(0, len(texts), self.config.batch_size):
                batch = texts[i : i + self.config.batch_size]
                batch_output = self._process_batch(
                    batch,
                    return_hidden_states=return_hidden_states,
                )

                emb = (
                    batch_output.embeddings.cpu() if to_cpu else batch_output.embeddings
                )
                collected_embeddings.append(emb)

                if (
                    return_hidden_states
                    and batch_output.layer_hidden_states is not None
                    and collected_layers is not None
                ):
                    # Initialize layer accumulators on the first batch.
                    if not collected_layers:
                        collected_layers = [
                            [] for _ in range(len(batch_output.layer_hidden_states))
                        ]
                    for layer_idx, layer_tensor in enumerate(
                        batch_output.layer_hidden_states,
                    ):
                        l_ten = layer_tensor.cpu() if to_cpu else layer_tensor
                        collected_layers[layer_idx].append(l_ten)

        # Concatenate collected batch chunks along the batch dimension (dim=0).
        final_embeddings = torch.cat(collected_embeddings, dim=0)
        final_layers = (
            [torch.cat(layer_chunks, dim=0) for layer_chunks in collected_layers]
            if collected_layers is not None
            else None
        )

        return EmbeddingOutput(
            embeddings=final_embeddings,
            layer_hidden_states=final_layers,
        )

    def stream_encode(
        self,
        texts: Sequence[str],
        *,
        return_hidden_states: bool = False,
        to_cpu: bool = True,
    ) -> Iterator[EmbeddingOutput]:
        """Stream encode texts in mini-batches, yielding results as a generator.

        Designed for processing large corpora where storing all embeddings in RAM
        or VRAM simultaneously is prohibitive. Batches are evaluated and yielded
        one by one without global accumulation.

        Args:
            texts: Sequence of text strings to encode.
            return_hidden_states: Whether to extract representations across all layers.
            to_cpu: If True, transfers each yielded batch tensor to CPU RAM.

        Yields:
            EmbeddingOutput for each processed batch.

        """
        with torch.inference_mode():
            for i in range(0, len(texts), self.config.batch_size):
                batch = texts[i : i + self.config.batch_size]
                batch_output = self._process_batch(
                    batch,
                    return_hidden_states=return_hidden_states,
                )

                emb = (
                    batch_output.embeddings.cpu() if to_cpu else batch_output.embeddings
                )
                layers = None
                if (
                    return_hidden_states
                    and batch_output.layer_hidden_states is not None
                ):
                    layers = [
                        l_ten.cpu() if to_cpu else l_ten
                        for l_ten in batch_output.layer_hidden_states
                    ]

                yield EmbeddingOutput(embeddings=emb, layer_hidden_states=layers)
