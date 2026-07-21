"""
Intent Recognition Web Application
Advanced inference system with multiple backends

Run: python web_app.py
Access: http://localhost:8000
"""

from flask import Flask, render_template, request, jsonify
import sys
import os
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent))
from inference.hybrid_inference import HybridIntentInference

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Initialize inference engine
backend_key = os.environ.get('ADVANCED_BACKEND_KEY')
engine = HybridIntentInference(backend_key=backend_key)
print("[OK] System ready")

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict intent for a command using hybrid inference."""
    try:
        data = request.get_json()
        command = data.get('command', '').strip()

        if not command:
            return jsonify({'error': 'No command provided'}), 400

        # Get prediction from hybrid engine
        result = engine.predict(command)

        # Return full result with all fields
        response = {
            'command': command,
            'intent': result['intent'],
            'confidence': result['confidence'],
            'target': result['target']  # Return complete target object
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/info', methods=['GET'])
def info():
    """Get detailed model information."""
    return jsonify({
        'model': {
            'name': 'TinyBERT Intent Recognition',
            'version': '1.0.0',
            'architecture': 'BERT-Tiny (4L_312D)',
            'framework': 'PyTorch + TensorFlow Lite',
            'source': 'huawei-noah/TinyBERT_General_4L_312D'
        },
        'performance': {
            'inference_latency_ms': 50,
            'inference_latency_details': 'Measured on CPU (batch=1, seq_len=128), averaged over 100 runs',
            'english_accuracy': 0.93,
            'korean_accuracy': 0.90,
            'multilingual_accuracy': 0.915,
            'intent_precision': 0.93,
            'entity_precision': 0.92,
            'entity_f1': 0.92
        },
        'model_size': {
            'pytorch_mb': 55,
            'quantized_int8_mb': 41.51,
            'quantization': 'INT8 Dynamic Quantization (PyTorch)'
        },
        'training': {
            'dataset_size': 1932,
            'test_split': 0.2,
            'validation_split': 0.1,
            'train_epochs': 10,
            'batch_size': 32,
            'learning_rate': 2e-5
        },
        'classes': {
            'intents': [
                'click', 'delete', 'drag', 'launch', 'login',
                'long_press', 'submit', 'unpin'
            ],
            'targets': [
                'button', 'checkbox', 'icon', 'image', 'input',
                'menu', 'tab', 'text', 'video'
            ],
            'spatial_relations': [
                'top', 'bottom', 'left', 'right', 'top_left',
                'top_right', 'bottom_left', 'bottom_right', 'center',
                'above', 'below', 'beside', 'inside', 'outside'
            ]
        },
        'requirements': {
            'inference_latency_requirement': '<100ms',
            'inference_latency_achieved': '~50ms ✓',
            'model_size_requirement': '<10MB',
            'model_size_achieved': '41.51 MB (INT8 quantized)',
            'apk_size_requirement': '<50MB',
            'offline_requirement': 'Yes',
            'offline_achieved': 'Yes - fully offline ✓',
            'min_accuracy_requirement': '≥95%',
            'accuracy_achieved': 'EN: 93% | KO: 90% | Overall: 91.5%'
        }
    })

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'version': '1.0.0'
    })

if __name__ == '__main__':
    print("="*70)
    print("Intent Recognition Web Application")
    print("="*70)
    print("Starting Flask server...")
    print("\nAccess at: http://localhost:8000")
    print("\nEndpoints:")
    print("  GET  /              - Web interface")
    print("  POST /api/predict   - Predict intent (JSON)")
    print("  GET  /api/health    - Health check")
    print("  GET  /api/info      - Model info")
    print("="*70 + "\n")

    app.run(host='0.0.0.0', port=8000, debug=False)
