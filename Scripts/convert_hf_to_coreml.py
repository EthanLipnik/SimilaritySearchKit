#!/usr/bin/env python3
"""
convert_hf_to_coreml.py

Convert HuggingFace sentence-transformer models to CoreML .mlpackage format
optimized for Apple Neural Engine (ANE).

Usage:
    python convert_hf_to_coreml.py --model "BAAI/bge-small-en-v1.5" --validate
    python convert_hf_to_coreml.py --model "intfloat/e5-small-v2" --quantize float16
    python convert_hf_to_coreml.py --model "BAAI/bge-m3" --sequence-length 8192

Requirements:
    pip install torch transformers coremltools numpy

Supported models:
    - BAAI/bge-small-en-v1.5 (384 dim, ~33MB)
    - intfloat/e5-small-v2 (384 dim, ~33MB)
    - BAAI/bge-m3 (1024 dim, ~200MB)
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim)
    - sentence-transformers/multi-qa-MiniLM-L6-cos-v1 (384 dim)
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingModelWrapper(nn.Module):
    """
    Wrapper that adds mean pooling to produce sentence embeddings.

    Most sentence-transformer models output token-level embeddings that need
    to be pooled into a single sentence embedding. This wrapper handles:
    1. Running the base model
    2. Mean pooling with attention mask
    3. L2 normalization (standard for similarity search)
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        # Get transformer outputs
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state  # [batch, seq, hidden]

        # Mean pooling with attention mask
        # Expand attention mask to match embedding dimensions
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        # Sum embeddings where attention mask is 1
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)

        # Count non-padding tokens for averaging
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)

        # Compute mean
        embeddings = sum_embeddings / sum_mask  # [batch, hidden]

        # L2 normalize (standard for similarity search)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Squeeze batch dimension for single-sample inference
        # This matches existing SimilaritySearchKit model output format
        return embeddings.squeeze(0)  # [hidden]


def load_model(model_name: str, cache_dir: str = None):
    """Load HuggingFace model and tokenizer."""
    from transformers import AutoModel, AutoTokenizer

    logging.info(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
    model.eval()

    return model, tokenizer


def trace_model(wrapped_model, sequence_length: int = 512):
    """Trace the wrapped model with example inputs."""
    logging.info(f"Tracing model with sequence length: {sequence_length}")

    # Create dummy inputs matching expected shapes
    # Shape: [batch=1, sequence_length]
    input_ids = torch.zeros(1, sequence_length, dtype=torch.int32)
    attention_mask = torch.ones(1, sequence_length, dtype=torch.int32)

    # Trace the model
    with torch.no_grad():
        traced = torch.jit.trace(
            wrapped_model,
            (input_ids, attention_mask),
            strict=False
        )

    return traced


def convert_to_coreml(
    traced_model,
    sequence_length: int = 512,
    compute_units: str = "all",
    minimum_deployment_target: str = "macOS13"
):
    """Convert traced PyTorch model to CoreML."""
    import coremltools as ct

    logging.info("Converting to CoreML")

    # Define input shapes
    input_shapes = [
        ct.TensorType(
            name="input_ids",
            shape=(1, sequence_length),
            dtype=np.int32
        ),
        ct.TensorType(
            name="attention_mask",
            shape=(1, sequence_length),
            dtype=np.int32
        )
    ]

    # Map compute units string to coremltools enum
    compute_units_map = {
        "all": ct.ComputeUnit.ALL,
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE
    }

    # Map deployment target
    target_map = {
        "macOS13": ct.target.macOS13,
        "macOS14": ct.target.macOS14,
        "iOS16": ct.target.iOS16,
        "iOS17": ct.target.iOS17,
    }

    # Convert with mlprogram format (required for Neural Engine optimization)
    mlmodel = ct.convert(
        traced_model,
        inputs=input_shapes,
        outputs=[ct.TensorType(name="embeddings")],
        convert_to="mlprogram",  # Required for ANE
        compute_units=compute_units_map.get(compute_units, ct.ComputeUnit.ALL),
        minimum_deployment_target=target_map.get(minimum_deployment_target, ct.target.macOS13),
    )

    return mlmodel


def quantize_model(mlmodel, quantization: str):
    """Apply quantization to reduce model size."""
    import coremltools as ct

    if quantization == "none":
        return mlmodel

    elif quantization == "float16":
        # For mlprogram models, float16 is handled during conversion
        # via compute_precision, but we can also post-process
        logging.info("Float16 quantization applied during conversion")
        return mlmodel

    elif quantization == "int8":
        logging.info("Applying int8 weight quantization")
        try:
            from coremltools.optimize.coreml import (
                OpLinearQuantizerConfig,
                OptimizationConfig,
                linear_quantize_weights
            )

            op_config = OpLinearQuantizerConfig(
                mode="linear_symmetric",
                dtype="int8",
                granularity="per_channel"
            )
            config = OptimizationConfig(global_config=op_config)

            return linear_quantize_weights(mlmodel, config)
        except ImportError:
            logging.warning("Int8 quantization requires coremltools >= 7.0, skipping")
            return mlmodel

    else:
        raise ValueError(f"Unknown quantization: {quantization}")


def set_model_metadata(mlmodel, model_name: str, embedding_dim: int):
    """Set CoreML model metadata."""

    # Model-specific info
    model_info = {
        "BAAI/bge-small-en-v1.5": {
            "description": f"BGE Small English v1.5 - Maps sentences to {embedding_dim}-dimensional dense vectors for semantic search.",
            "license": "MIT",
            "author": "BAAI (https://huggingface.co/BAAI/bge-small-en-v1.5)"
        },
        "intfloat/e5-small-v2": {
            "description": f"E5 Small v2 - Maps sentences to {embedding_dim}-dimensional dense vectors optimized for text retrieval.",
            "license": "MIT",
            "author": "Microsoft (https://huggingface.co/intfloat/e5-small-v2)"
        },
        "BAAI/bge-m3": {
            "description": f"BGE M3 - Multi-lingual, multi-granularity model mapping sentences to {embedding_dim}-dimensional vectors.",
            "license": "MIT",
            "author": "BAAI (https://huggingface.co/BAAI/bge-m3)"
        },
        "sentence-transformers/all-MiniLM-L6-v2": {
            "description": f"MiniLM All - General purpose sentence embeddings ({embedding_dim} dimensions).",
            "license": "Apache-2.0",
            "author": "sentence-transformers (https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)"
        },
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1": {
            "description": f"MiniLM MultiQA - Optimized for question-answering retrieval ({embedding_dim} dimensions).",
            "license": "Apache-2.0",
            "author": "sentence-transformers (https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1)"
        }
    }

    info = model_info.get(model_name, {
        "description": f"Sentence embedding model ({embedding_dim}D) converted from {model_name}",
        "license": "See original model",
        "author": f"https://huggingface.co/{model_name}"
    })

    mlmodel.author = info["author"]
    mlmodel.license = info["license"]
    mlmodel.short_description = info["description"]
    mlmodel.version = "1.0"

    # Input/output descriptions
    mlmodel.input_description["input_ids"] = "Tokenized input IDs (BERT WordPiece)"
    mlmodel.input_description["attention_mask"] = "Attention mask (1 for real tokens, 0 for padding)"
    mlmodel.output_description["embeddings"] = f"{embedding_dim}-dimensional L2-normalized sentence embedding"

    return mlmodel


def validate_model(mlpackage_path: str, tokenizer, embedding_dim: int, sequence_length: int):
    """Validate converted model produces correct embeddings."""
    import coremltools as ct

    logging.info("Running validation")

    # Load the converted model
    mlmodel = ct.models.MLModel(mlpackage_path)

    # Test sentences
    test_sentences = [
        "This is a test sentence for validation.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models can process natural language."
    ]

    for sentence in test_sentences:
        # Tokenize
        tokens = tokenizer(
            sentence,
            padding="max_length",
            truncation=True,
            max_length=sequence_length,
            return_tensors="np"
        )

        # Run prediction
        prediction = mlmodel.predict({
            "input_ids": tokens["input_ids"].astype(np.int32),
            "attention_mask": tokens["attention_mask"].astype(np.int32)
        })

        embeddings = prediction["embeddings"]

        # Validate shape
        assert embeddings.shape == (embedding_dim,), \
            f"Expected shape ({embedding_dim},), got {embeddings.shape}"

        # Validate L2 normalized (norm should be ~1.0)
        norm = np.linalg.norm(embeddings)
        assert 0.99 < norm < 1.01, f"Expected L2 norm ~1.0, got {norm}"

    logging.info(f"Validation passed for {len(test_sentences)} test sentences")
    logging.info(f"  - Shape: {embeddings.shape}")
    logging.info(f"  - L2 norm: {norm:.4f}")
    logging.info(f"  - Sample values: [{', '.join(f'{v:.4f}' for v in embeddings[:5])}...]")

    return True


def get_model_config(model_name: str, args):
    """Get model-specific configuration."""

    # Default sequence length based on model
    default_seq_length = {
        "BAAI/bge-m3": 8192,  # BGE-M3 supports longer context
    }

    sequence_length = args.sequence_length
    if sequence_length is None:
        sequence_length = default_seq_length.get(model_name, 512)

    return {
        "sequence_length": sequence_length
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace embedding models to CoreML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic conversion
    python convert_hf_to_coreml.py --model "BAAI/bge-small-en-v1.5" --validate

    # With float16 quantization
    python convert_hf_to_coreml.py --model "intfloat/e5-small-v2" --quantize float16

    # BGE-M3 with longer context
    python convert_hf_to_coreml.py --model "BAAI/bge-m3" --sequence-length 8192

    # Target specific compute units
    python convert_hf_to_coreml.py --model "BAAI/bge-small-en-v1.5" --compute-units cpu_and_ne
        """
    )

    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model name (e.g., 'BAAI/bge-small-en-v1.5')"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .mlpackage path (default: {model_name}.mlpackage)"
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=None,
        help="Max sequence length (default: 512, or 8192 for BGE-M3)"
    )
    parser.add_argument(
        "--quantize",
        choices=["none", "float16", "int8"],
        default="none",
        help="Quantization mode (default: none)"
    )
    parser.add_argument(
        "--compute-units",
        choices=["all", "cpu_only", "cpu_and_gpu", "cpu_and_ne"],
        default="all",
        help="Target compute units (default: all)"
    )
    parser.add_argument(
        "--deployment-target",
        choices=["macOS13", "macOS14", "iOS16", "iOS17"],
        default="macOS13",
        help="Minimum deployment target (default: macOS13)"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation after conversion"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="HuggingFace cache directory"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Determine output path
    if args.output is None:
        model_safe_name = args.model.replace("/", "-")
        args.output = f"{model_safe_name}.mlpackage"

    # Get model-specific config
    config = get_model_config(args.model, args)
    sequence_length = config["sequence_length"]

    logging.info(f"Configuration:")
    logging.info(f"  Model: {args.model}")
    logging.info(f"  Output: {args.output}")
    logging.info(f"  Sequence length: {sequence_length}")
    logging.info(f"  Quantization: {args.quantize}")
    logging.info(f"  Compute units: {args.compute_units}")

    # Load model
    model, tokenizer = load_model(args.model, args.cache_dir)

    # Get embedding dimension
    embedding_dim = model.config.hidden_size
    logging.info(f"Embedding dimension: {embedding_dim}")

    # Wrap model with mean pooling
    logging.info("Creating wrapper with mean pooling and L2 normalization")
    wrapped = EmbeddingModelWrapper(model)
    wrapped.eval()

    # Trace model
    traced = trace_model(wrapped, sequence_length)

    # Convert to CoreML
    mlmodel = convert_to_coreml(
        traced,
        sequence_length=sequence_length,
        compute_units=args.compute_units,
        minimum_deployment_target=args.deployment_target
    )

    # Apply quantization
    mlmodel = quantize_model(mlmodel, args.quantize)

    # Set metadata
    mlmodel = set_model_metadata(mlmodel, args.model, embedding_dim)

    # Save
    logging.info(f"Saving to {args.output}")
    mlmodel.save(args.output)

    # Calculate size
    output_path = Path(args.output)
    if output_path.exists():
        total_size = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file())
        size_mb = total_size / (1024 * 1024)
        logging.info(f"Model size: {size_mb:.1f} MB")

    # Validate
    if args.validate:
        validate_model(args.output, tokenizer, embedding_dim, sequence_length)

    # Summary
    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"{'='*60}")
    print(f"  Model: {args.model}")
    print(f"  Output: {args.output}")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Sequence length: {sequence_length}")
    if output_path.exists():
        print(f"  Size: {size_mb:.1f} MB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
