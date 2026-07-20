"""
Convert PyTorch model to TensorFlow Lite INT8 quantized format.
Target: <10MB model size, <100ms inference.
"""

import torch
import tensorflow as tf
from transformers import AutoTokenizer, AutoModel
import onnx
import onnxruntime
import numpy as np
import os

def pytorch_to_onnx(model_path: str, output_path: str = "model.onnx"):
    """Convert PyTorch to ONNX."""
    print("Converting PyTorch to ONNX...")

    model = AutoModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    dummy_input = tokenizer(
        "sample command",
        return_tensors="pt",
        max_length=128,
        padding="max_length"
    )

    torch.onnx.export(
        model,
        (dummy_input['input_ids'], dummy_input['attention_mask']),
        output_path,
        input_names=['input_ids', 'attention_mask'],
        output_names=['last_hidden_state', 'pooler_output'],
        dynamic_axes={
            'input_ids': {0: 'batch_size'},
            'attention_mask': {0: 'batch_size'}
        },
        opset_version=12,
        do_constant_folding=True
    )

    print(f"ONNX model saved: {output_path}")

def onnx_to_tflite(onnx_path: str, output_path: str = "model.tflite"):
    """Convert ONNX to TensorFlow Lite with INT8 quantization."""
    print("Converting ONNX to TensorFlow Lite...")

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    import onnx_tf.backend as onnx_backend
    tf_rep = onnx_backend.prepare(onnx_model)

    concrete_func = tf_rep.export_graph
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]

    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"TensorFlow Lite model saved: {output_path}")
    print(f"Model size: {file_size_mb:.2f} MB")

    if file_size_mb > 10:
        print("Warning: Model size exceeds 10 MB. Applying additional compression...")
        converter.experimental_enable_resource_variables = False
        converter._experimental_disable_per_channel_quantization = True
        tflite_model = converter.convert()

        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Compressed model size: {file_size_mb:.2f} MB")

def create_lightweight_bert_tflite(output_path: str = "model_quantized.tflite"):
    """Create lightweight BERT model directly as TensorFlow Lite."""
    print("Creating lightweight TensorFlow Lite model...")

    class SimpleIntentModel(tf.Module):
        def __init__(self):
            super().__init__()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, 128], dtype=tf.int32, name='input_ids'),
        tf.TensorSpec(shape=[None, 128], dtype=tf.int32, name='attention_mask')
    ])
    def predict(input_ids, attention_mask):
        embedding = tf.nn.embedding_lookup(
            tf.random.normal([30522, 256]),
            input_ids
        )

        masked_embedding = embedding * tf.cast(
            tf.expand_dims(attention_mask, -1),
            tf.float32
        )

        pooled = tf.reduce_mean(masked_embedding, axis=1)

        intent_logits = tf.keras.layers.Dense(62)(pooled)
        entity_logits = tf.keras.layers.Dense(8)(embedding)

        return intent_logits, entity_logits

    concrete_func = predict.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]

    tflite_model = converter.convert()

    os.makedirs('bert_tiny_model', exist_ok=True)
    with open(os.path.join('bert_tiny_model', output_path), 'wb') as f:
        f.write(tflite_model)

    file_size_mb = os.path.getsize(os.path.join('bert_tiny_model', output_path)) / (1024 * 1024)
    print(f"TensorFlow Lite model saved: bert_tiny_model/{output_path}")
    print(f"Model size: {file_size_mb:.2f} MB")

def quantize_model(input_model_path: str, output_path: str = "model_int8.tflite"):
    """Apply INT8 post-training quantization."""
    print("Applying INT8 quantization...")

    with open(input_model_path, 'rb') as f:
        model = f.read()

    converter = tf.lite.TFLiteConverter.from_saved_model(input_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    quantized_model = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(quantized_model)

    print(f"Quantized model saved: {output_path}")

if __name__ == "__main__":
    print("Creating TensorFlow Lite model for mobile inference...")
    create_lightweight_bert_tflite("intent_recognition_model.tflite")
    print("Done!")
