"""
Inference module for Intent Recognition System.
Handles offline model inference and JSON output generation.
"""

import torch
import json
from transformers import AutoTokenizer
from typing import Dict, Any, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'training'))
from model import MultitaskIntentModel, ConfigManager

class IntentRecognitionInference:
    """Offline inference engine for intent recognition."""

    def __init__(self, model_path: str = "bert_tiny_model", device: str = "cpu"):
        """Initialize model and tokenizer."""
        self.device = torch.device(device)
        self.model_path = model_path

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = MultitaskIntentModel(
            model_name=model_path,
            num_intents=len(ConfigManager.INTENTS),
            num_entities=len(ConfigManager.ENTITY_TYPES)
        ).to(self.device)

        model_weights = f"{model_path}/pytorch_model.bin"
        if Path(model_weights).exists():
            self.model.load_state_dict(torch.load(model_weights, map_location=self.device))
        else:
            print(f"Warning: Model weights not found at {model_weights}")

        self.model.eval()
        print(f"Model loaded from {model_path} on device {device}")

    def extract_target_info(self, command: str) -> Dict[str, Optional[str]]:
        """Extract target object and related information from command."""
        target_info = {
            "type": None,
            "attribute": None,
            "label": None,
            "index": None,
            "relation": None,
            "reference": None
        }

        command_lower = command.lower()

        for target in ConfigManager.TARGETS:
            if target.replace('_', ' ') in command_lower:
                target_info["type"] = target
                break

        spatial_keywords = {
            "top": "top", "bottom": "bottom", "left": "left", "right": "right",
            "center": "center", "above": "above", "below": "below",
            "first": "first", "second": "second", "third": "third",
            "last": "last", "previous": "previous", "next": "next"
        }

        for keyword, relation in spatial_keywords.items():
            if keyword in command_lower:
                target_info["relation"] = relation
                break

        attributes = ["red", "blue", "green", "large", "small", "disabled", "primary"]
        for attr in attributes:
            if attr in command_lower:
                target_info["attribute"] = attr
                break

        return target_info

    def predict(self, command: str) -> Dict[str, Any]:
        """
        Predict intent for given command.
        Returns structured JSON output.
        """
        encoding = self.tokenizer(
            command,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            intent_logits, entity_logits = self.model(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask']
            )

        intent_probs = torch.softmax(intent_logits, dim=-1)
        intent_id = intent_probs.argmax(dim=-1).item()
        intent_confidence = intent_probs[0, intent_id].item()
        intent_name = ConfigManager.get_intent_label(intent_id)

        target_info = self.extract_target_info(command)

        output = {
            "intent": intent_name,
            "confidence": round(intent_confidence, 4),
            "target": {
                "type": target_info.get("type"),
                "attribute": target_info.get("attribute"),
                "label": target_info.get("label"),
                "index": target_info.get("index"),
                "relation": target_info.get("relation"),
                "reference": target_info.get("reference")
            }
        }

        return output

    def predict_batch(self, commands: list) -> list:
        """Predict intents for multiple commands."""
        results = []
        for command in commands:
            results.append(self.predict(command))
        return results

class TFLiteInference:
    """TensorFlow Lite inference for mobile deployment."""

    def __init__(self, model_path: str = "bert_tiny_model/intent_recognition_model.tflite"):
        """Initialize TensorFlow Lite interpreter."""
        try:
            import tensorflow as tf
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print(f"TFLite model loaded from {model_path}")
        except ImportError:
            print("TensorFlow not available. Using PyTorch inference instead.")
            self.interpreter = None

    def predict(self, command: str) -> Dict[str, Any]:
        """Predict using TensorFlow Lite model."""
        if not self.interpreter:
            return {"error": "TensorFlow Lite model not available"}

        try:
            import tensorflow as tf
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")
            encoding = tokenizer(
                command,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='np'
            )

            for i, input_detail in enumerate(self.input_details):
                if input_detail['name'] == 'input_ids':
                    self.interpreter.set_tensor(
                        input_detail['index'],
                        encoding['input_ids'].astype(np.int32)
                    )
                elif input_detail['name'] == 'attention_mask':
                    self.interpreter.set_tensor(
                        input_detail['index'],
                        encoding['attention_mask'].astype(np.int32)
                    )

            self.interpreter.invoke()

            output_data = []
            for output_detail in self.output_details:
                output_data.append(self.interpreter.get_tensor(output_detail['index']))

            intent_logits = output_data[0]
            intent_id = intent_logits.argmax(axis=-1)[0]
            intent_confidence = float(torch.softmax(torch.from_numpy(intent_logits[0]), dim=0)[intent_id])

            return {
                "intent": ConfigManager.get_intent_label(int(intent_id)),
                "confidence": round(intent_confidence, 4),
                "target": {
                    "type": None,
                    "attribute": None,
                    "label": None,
                    "index": None,
                    "relation": None,
                    "reference": None
                }
            }
        except Exception as e:
            return {"error": str(e)}

def get_inference_engine(model_type: str = "pytorch", **kwargs):
    """Factory function to get inference engine."""
    if model_type == "pytorch":
        return IntentRecognitionInference(**kwargs)
    elif model_type == "tflite":
        return TFLiteInference(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

if __name__ == "__main__":
    import numpy as np

    engine = IntentRecognitionInference()

    test_commands = [
        "click the button",
        "scroll down",
        "delete the message",
        "search for item",
        "open the menu",
    ]

    print("\n" + "="*60)
    print("Testing Intent Recognition Inference")
    print("="*60)

    for command in test_commands:
        result = engine.predict(command)
        print(f"\nCommand: {command}")
        print(f"Output: {json.dumps(result, indent=2)}")
