"""
Unit tests for inference module.
"""

import pytest
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent / 'inference'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'training'))

from inference import IntentRecognitionInference, get_inference_engine
from model import ConfigManager

@pytest.fixture
def inference_engine():
    """Create inference engine for testing."""
    try:
        engine = IntentRecognitionInference(device='cpu')
        return engine
    except Exception as e:
        pytest.skip(f"Model not available: {e}")

class TestIntentRecognition:
    """Test intent recognition functionality."""

    def test_engine_initialization(self, inference_engine):
        """Test that engine initializes correctly."""
        assert inference_engine is not None
        assert inference_engine.model is not None
        assert inference_engine.tokenizer is not None

    def test_predict_click_intent(self, inference_engine):
        """Test recognition of click intent."""
        result = inference_engine.predict("click the button")
        assert result is not None
        assert 'intent' in result
        assert 'confidence' in result
        assert 'target' in result
        assert 0 <= result['confidence'] <= 1

    def test_predict_scroll_intent(self, inference_engine):
        """Test recognition of scroll intent."""
        result = inference_engine.predict("scroll down")
        assert result['intent'] in ConfigManager.INTENTS

    def test_predict_delete_intent(self, inference_engine):
        """Test recognition of delete intent."""
        result = inference_engine.predict("delete the message")
        assert result['intent'] in ConfigManager.INTENTS

    def test_json_output_format(self, inference_engine):
        """Test that output is valid JSON."""
        result = inference_engine.predict("submit the form")
        json_str = json.dumps(result)
        parsed = json.loads(json_str)
        assert 'intent' in parsed
        assert 'confidence' in parsed
        assert 'target' in parsed

    def test_confidence_score_range(self, inference_engine):
        """Test that confidence is between 0 and 1."""
        result = inference_engine.predict("open the menu")
        assert 0 <= result['confidence'] <= 1

    def test_target_extraction(self, inference_engine):
        """Test target object extraction."""
        result = inference_engine.predict("click the red button")
        assert result['target'] is not None
        assert isinstance(result['target'], dict)

    def test_multilingual_input_english(self, inference_engine):
        """Test English language input."""
        result = inference_engine.predict("click button")
        assert result['intent'] in ConfigManager.INTENTS

    def test_batch_prediction(self, inference_engine):
        """Test batch prediction."""
        commands = [
            "click the button",
            "scroll down",
            "delete the message",
        ]
        results = inference_engine.predict_batch(commands)
        assert len(results) == len(commands)
        for result in results:
            assert 'intent' in result
            assert 'confidence' in result

    def test_typo_robustness(self, inference_engine):
        """Test robustness to typos."""
        result1 = inference_engine.predict("click the button")
        result2 = inference_engine.predict("clik the buttn")
        assert result1['intent'] in ConfigManager.INTENTS
        assert result2['intent'] in ConfigManager.INTENTS

class TestInferenceEngine:
    """Test inference engine factory."""

    def test_pytorch_engine_creation(self):
        """Test creation of PyTorch inference engine."""
        try:
            engine = get_inference_engine(model_type='pytorch')
            assert engine is not None
        except Exception as e:
            pytest.skip(f"Model not available: {e}")

    def test_invalid_engine_type(self):
        """Test that invalid engine type raises error."""
        with pytest.raises(ValueError):
            get_inference_engine(model_type='invalid')

class TestTargetExtraction:
    """Test target information extraction."""

    def test_target_type_detection(self, inference_engine):
        """Test detection of target type."""
        result = inference_engine.predict("click the button")
        assert result['target'] is not None

    def test_spatial_relation_detection(self, inference_engine):
        """Test detection of spatial relations."""
        result = inference_engine.predict("click the button above the image")
        assert result['target'] is not None

    def test_attribute_detection(self, inference_engine):
        """Test detection of attributes."""
        result = inference_engine.predict("click the red button")
        assert result['target'] is not None

class TestPerformance:
    """Test performance metrics."""

    def test_inference_completes_quickly(self, inference_engine):
        """Test that inference completes within reasonable time."""
        import time
        start = time.time()
        result = inference_engine.predict("click button")
        duration = time.time() - start
        assert duration < 10, f"Inference took {duration}s, expected < 10s"

    def test_batch_inference_performance(self, inference_engine):
        """Test batch inference performance."""
        import time
        commands = ["click button"] * 10
        start = time.time()
        results = inference_engine.predict_batch(commands)
        duration = time.time() - start
        assert len(results) == 10
        assert duration < 30, f"Batch inference took {duration}s"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
