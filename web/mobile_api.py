"""
Flask API backend for Intent Recognition mobile app.
Serves predictions from the trained TinyBERT model.
Run: python mobile_api.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from inference.inference_correct import ProperIntentInference

app = Flask(__name__)
CORS(app)  # Enable CORS for mobile app

# Initialize inference engine
print("Initializing inference engine...")
engine = ProperIntentInference()
print("[OK] Inference engine ready")

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict intent for a command."""
    try:
        data = request.get_json()
        command = data.get('command', '').strip()

        if not command:
            return jsonify({'error': 'No command provided'}), 400

        # Get prediction
        result = engine.predict(command)

        # Format response for mobile app
        response = {
            'intent': result['intent'],
            'confidence': result['intent_confidence'],
            'target': {
                'type': result['target'] if result['target'] != 'none' else None,
                'attribute': None,
                'label': None,
                'index': None,
                'relation': result['spatial_relation'] if result['spatial_relation'] != 'none' else None,
                'reference': None,
            }
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model': 'TinyBERT Intent Recognition',
        'version': '1.0.0'
    })

@app.route('/api/info', methods=['GET'])
def info():
    """Get model information."""
    return jsonify({
        'model': 'TinyBERT Intent Recognition',
        'version': '1.0.0',
        'intents': 8,
        'targets': 9,
        'spatial_relations': 6,
        'inference_latency_ms': 50,
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Intent Recognition API Server")
    print("="*60)
    print("Listening on: http://localhost:5000")
    print("Endpoints:")
    print("  POST /api/predict")
    print("  GET /api/health")
    print("  GET /api/info")
    print("\nFor mobile app use: http://localhost:5000/api")
    print("="*60 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=False)
