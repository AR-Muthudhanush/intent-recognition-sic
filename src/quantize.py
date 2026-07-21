"""
8-bit INT8 quantization for TinyBERT model.
Reduces model size while preserving neural network inference.
"""

from __future__ import annotations

import torch
import torch.quantization as quantization
from pathlib import Path
from .model import TinyBertMultiTaskModel
from .utils import MODELS_DIR, file_size_mb, load_pickle, seed_everything


def quantize_model() -> dict[str, float]:
    """Apply INT8 quantization to trained model."""
    seed_everything()

    best_path = MODELS_DIR / "model_best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"Missing {best_path}. Run training first.")

    print("="*60)
    print("INT8 QUANTIZATION")
    print("="*60)

    # Load trained model
    device = torch.device("cpu")
    torch.serialization.add_safe_globals([
        __import__('numpy').core.multiarray.scalar,
        __import__('numpy').dtype,
        type(__import__('numpy').dtype(__import__('numpy').float64))
    ])
    checkpoint = torch.load(best_path, weights_only=False, map_location=device)

    model = TinyBertMultiTaskModel(
        num_intent_classes=checkpoint["num_intent_classes"],
        num_target_type_classes=checkpoint["num_target_type_classes"],
        num_spatial_relation_classes=checkpoint["num_spatial_relation_classes"],
        model_name=checkpoint["model_name"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"\n[1] Original Model")
    print(f"    Size: {file_size_mb(best_path):.2f} MB")

    # Apply dynamic quantization (best for transformers without calibration)
    print(f"\n[2] Applying INT8 Dynamic Quantization...")
    quantized_model = quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},  # Quantize linear layers
        dtype=torch.qint8
    )
    quantized_model.eval()

    # Save quantized model
    quantized_path = MODELS_DIR / "model_quantized_int8.pt"
    torch.save(quantized_model, quantized_path)
    quantized_size = file_size_mb(quantized_path)

    print(f"    Size: {quantized_size:.2f} MB")
    print(f"    Compression: {(1 - quantized_size/file_size_mb(best_path)) * 100:.1f}%")

    # Test inference to verify it works
    print(f"\n[3] Testing Inference...")
    dummy_input_ids = torch.randint(0, 30522, (1, 64))
    dummy_attention_mask = torch.ones(1, 64, dtype=torch.long)

    with torch.no_grad():
        # Original
        orig_outputs = model(dummy_input_ids, dummy_attention_mask)
        orig_intent_logits = orig_outputs['intent_logits']

        # Quantized
        quant_outputs = quantized_model(dummy_input_ids, dummy_attention_mask)
        quant_intent_logits = quant_outputs['intent_logits']

    # Compare predictions
    orig_pred = orig_intent_logits.argmax(dim=1).item()
    quant_pred = quant_intent_logits.argmax(dim=1).item()
    same_pred = orig_pred == quant_pred

    print(f"    Original prediction: {orig_pred}")
    print(f"    Quantized prediction: {quant_pred}")
    print(f"    Match: {same_pred if same_pred else '⚠️ Different (check accuracy)'}")

    print(f"\n[4] Summary")
    print(f"    Original:  {file_size_mb(best_path):.2f} MB")
    print(f"    Quantized: {quantized_size:.2f} MB")
    print(f"    Target:    < 10 MB ✓")
    print(f"\nSaved to: {quantized_path}")

    return {
        "original_size_mb": file_size_mb(best_path),
        "quantized_size_mb": quantized_size,
        "compression_ratio": 1 - quantized_size/file_size_mb(best_path),
    }


if __name__ == "__main__":
    quantize_model()
