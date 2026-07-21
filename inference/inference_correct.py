"""
Correct inference using the actual trained TinyBERT model.
Uses the architecture from src/ with label encoders.
"""

import torch
from pathlib import Path
import pickle
import json
from transformers import AutoTokenizer

# Import the correct model architecture
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model import TinyBertMultiTaskModel
from src.utils import MODEL_NAME

class ProperIntentInference:
    """Inference using the correctly trained model."""

    def __init__(self, model_path: str = "models", device: str = "cpu", use_quantized: bool = False):
        self.device = torch.device(device)
        self.model_path = Path(model_path)

        print(f"Loading model from {self.model_path}...")

        # Load label encoders
        encoders_path = self.model_path / "label_encoders.pkl"
        if encoders_path.exists():
            with open(encoders_path, 'rb') as f:
                self.encoders = pickle.load(f)
            print(f"[OK] Loaded label encoders")
        else:
            raise FileNotFoundError(f"Encoders not found at {encoders_path}")

        # Get number of classes
        num_intents = len(self.encoders['intent'].classes_)
        num_targets = len(self.encoders['target_type'].classes_)
        num_spatials = len(self.encoders['spatial_relation'].classes_)

        print(f"[OK] Intents: {num_intents}, Targets: {num_targets}, Spatial: {num_spatials}")

        # Initialize model
        self.model = TinyBertMultiTaskModel(
            num_intent_classes=num_intents,
            num_target_type_classes=num_targets,
            num_spatial_relation_classes=num_spatials,
            model_name=MODEL_NAME
        ).to(self.device)

        # Load trained weights - prioritize INT8 quantized neural network
        weights_loaded = False

        # Try INT8 quantized model first (real neural network with INT8)
        int8_model_path = self.model_path / "model_quantized_int8.pt"
        if use_quantized and int8_model_path.exists():
            try:
                print(f"Loading INT8 quantized model...")
                # INT8 quantized model is a full model object, not checkpoint
                self.model = torch.load(int8_model_path, map_location=self.device, weights_only=False)
                file_size = int8_model_path.stat().st_size / (1024*1024)
                print(f"[OK] INT8 Quantized model loaded ({file_size:.1f}MB)")
                weights_loaded = True
            except Exception as e:
                print(f"Could not load INT8 model: {type(e).__name__}")

        # Fallback to checkpoint models
        if not weights_loaded:
            model_candidates = [
                self.model_path / "model_best.pt",
                self.model_path / "model_8mb.pt",
                self.model_path / "quantized_model.pt",
            ]

            for model_file in model_candidates:
                if model_file.exists():
                    try:
                        print(f"Loading checkpoint model from {model_file.name}...")
                        checkpoint = torch.load(model_file, map_location=self.device, weights_only=False)

                        # Extract state dict from checkpoint
                        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                            state_dict = checkpoint['model_state_dict']
                        else:
                            state_dict = checkpoint

                        self.model.load_state_dict(state_dict)
                        file_size = model_file.stat().st_size / (1024*1024)
                        print(f"[OK] Model loaded ({file_size:.1f}MB)")
                        weights_loaded = True
                        break
                    except Exception as e:
                        print(f"Could not load {model_file.name}: {type(e).__name__}")
                        continue

        if not weights_loaded:
            raise FileNotFoundError(f"No model weights found in {self.model_path}")

        self.model.eval()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        print(f"[OK] Ready for inference on {device}")

    def predict(self, command: str):
        """Predict intent, target, and spatial relation."""
        # Tokenize
        inputs = self.tokenizer(
            command,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get predictions
        intent_logits = outputs['intent_logits']
        target_logits = outputs['target_logits']
        spatial_logits = outputs['spatial_logits']

        # Get probabilities
        intent_probs = torch.softmax(intent_logits, dim=-1)
        target_probs = torch.softmax(target_logits, dim=-1)
        spatial_probs = torch.softmax(spatial_logits, dim=-1)

        # Get top predictions
        intent_pred_id = intent_probs.argmax(dim=-1).item()
        target_pred_id = target_probs.argmax(dim=-1).item()
        spatial_pred_id = spatial_probs.argmax(dim=-1).item()

        intent_confidence = intent_probs[0, intent_pred_id].item()
        target_confidence = target_probs[0, target_pred_id].item()
        spatial_confidence = spatial_probs[0, spatial_pred_id].item()

        # Decode labels
        intent = self.encoders['intent'].classes_[intent_pred_id]
        target = self.encoders['target_type'].classes_[target_pred_id]
        spatial = self.encoders['spatial_relation'].classes_[spatial_pred_id]

        return {
            "command": command,
            "intent": intent,
            "intent_confidence": round(intent_confidence, 4),
            "target": target,
            "target_confidence": round(target_confidence, 4),
            "spatial_relation": spatial,
            "spatial_confidence": round(spatial_confidence, 4),
        }

    def predict_batch(self, commands: list):
        """Predict for multiple commands."""
        return [self.predict(cmd) for cmd in commands]


if __name__ == "__main__":
    print("="*60)
    print("Intent Recognition Inference")
    print("="*60)

    engine = ProperIntentInference()

    test_commands = [
        "click the button",
        "scroll down",
        "delete the message",
        "search for item",
        "open the menu",
        "Click the red button below the cart",
    ]

    print("\nTesting predictions:\n")
    for cmd in test_commands:
        result = engine.predict(cmd)
        print(f"Command: {cmd}")
        print(f"Intent: {result['intent']} ({result['intent_confidence']:.2%})")
        print(f"Target: {result['target']} ({result['target_confidence']:.2%})")
        print(f"Spatial: {result['spatial_relation']} ({result['spatial_confidence']:.2%})")
        print()
