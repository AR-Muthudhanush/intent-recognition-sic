"""
Inference module for Intent Recognition System.
Handles offline model inference and JSON output generation.
"""

from .inference import (
    IntentRecognitionInference,
    TFLiteInference,
    get_inference_engine,
)

__all__ = [
    'IntentRecognitionInference',
    'TFLiteInference',
    'get_inference_engine',
]
