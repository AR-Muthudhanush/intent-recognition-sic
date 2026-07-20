# Deployment Guide

## Overview

This guide covers deploying the Intent Recognition System:
1. Python-based training & inference
2. React Native Expo mobile app
3. Building APK/AAB for Android

## Prerequisites

### Python Deployment

- Python 3.9+
- PyTorch 2.1+
- Transformers 4.34+

Install:
```bash
pip install -r requirements.txt
```

### Mobile Deployment

- Node.js 16+
- npm or yarn
- Expo CLI

Install Expo:
```bash
npm install -g expo-cli
```

## Model Deployment

### Step 1: Train Model

```bash
cd training
python train.py
```

**Output**: 
- `bert_tiny_model/pytorch_model.bin` (trained weights)
- `bert_tiny_model/config.json` (model config)
- `bert_tiny_model/tokenizer_config.json` (tokenizer)

### Step 2: Evaluate

```bash
python evaluate.py
```

**Output**:
- `evaluation_results.json` (metrics)
- Console: accuracy, precision, recall, F1, latency

### Step 3: Convert to TFLite (Optional)

For mobile deployment with TensorFlow Lite:

```bash
python convert_tflite.py
```

**Output**:
- `bert_tiny_model/intent_recognition_model.tflite` (<10MB)

## Python Inference Server

### Option 1: Local Offline (Recommended)

```python
from inference import IntentRecognitionInference

engine = IntentRecognitionInference()
result = engine.predict("click the button")
```

### Option 2: Flask REST API

Create `api.py`:

```python
from flask import Flask, request, jsonify
from inference import IntentRecognitionInference

app = Flask(__name__)
engine = IntentRecognitionInference()

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    command = data.get('command', '')
    
    if not command:
        return jsonify({"error": "No command provided"}), 400
    
    result = engine.predict(command)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

Run server:
```bash
pip install flask
python api.py
```

Access:
```
POST http://localhost:5000/api/predict
Body: {"command": "click button"}
```

## Mobile Deployment

### Step 1: Setup React Native

```bash
cd mobile
npm install
```

### Step 2: Copy Model (if using TFLite)

```bash
mkdir -p mobile/assets
cp bert_tiny_model/intent_recognition_model.tflite mobile/assets/
```

### Step 3: Test Locally

**iOS**:
```bash
npx expo start --ios
```

**Android**:
```bash
npx expo start --android
```

### Step 4: Build APK

#### Option A: Expo Build Service

```bash
npx eas build --platform android
```

Requirements:
- Expo account (free)
- EAS CLI: `npm install -g eas-cli`

#### Option B: Local Build (Requires Android SDK)

```bash
npx expo run:android
```

Requirements:
- Android SDK installed
- Android Virtual Device (AVD) or physical device

### Step 5: Build AAB (Google Play)

For Play Store distribution:

```bash
npx eas build --platform android --type app-bundle
```

## Android Play Store Deployment

### Prerequisites

1. **Google Developer Account** ($25 one-time)
2. **Keystore file** (signing certificate)
3. **App screenshots** (5-10)
4. **Icon** (512×512 PNG)
5. **Description** (<80 characters)
6. **Privacy policy** (URL)

### Step 1: Create Keystore

```bash
keytool -genkey -v -keystore intent-recognition.keystore \
  -keyalg RSA -keysize 2048 -validity 10000 \
  -alias intent-recognition
```

### Step 2: Configure Build

Edit `app.json`:

```json
{
  "expo": {
    "android": {
      "package": "com.intentrecognition.app",
      "versionCode": 1,
      "permissions": [
        "INTERNET",
        "READ_EXTERNAL_STORAGE"
      ],
      "keystore": "intent-recognition.keystore",
      "keystorePassword": "YOUR_PASSWORD",
      "keyAlias": "intent-recognition",
      "keyPassword": "YOUR_PASSWORD"
    }
  }
}
```

### Step 3: Build AAB

```bash
npx eas build --platform android --type app-bundle
```

### Step 4: Upload to Play Store

1. Go to Google Play Console
2. Create new app
3. Fill app info (screenshots, description, etc.)
4. Upload AAB file
5. Review and publish

## Docker Deployment

### Create Dockerfile

```dockerfile
FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "api.py"]
```

### Build & Run

```bash
docker build -t intent-recognition .
docker run -p 5000:5000 intent-recognition
```

## Cloud Deployment

### AWS Lambda

Create `lambda_handler.py`:

```python
import json
from inference import IntentRecognitionInference

engine = IntentRecognitionInference()

def handler(event, context):
    command = json.loads(event['body'])['command']
    result = engine.predict(command)
    
    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }
```

Package & deploy:
```bash
pip install -r requirements.txt -t package/
cp lambda_handler.py package/
cd package && zip -r ../function.zip . && cd ..
aws lambda create-function --function-name intent-recognition \
  --runtime python3.9 --zip-file fileb://function.zip \
  --handler lambda_handler.handler
```

### Google Cloud Run

```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/intent-recognition
gcloud run deploy intent-recognition \
  --image gcr.io/PROJECT_ID/intent-recognition \
  --platform managed --region us-central1
```

### Azure Container Instances

```bash
az acr build --registry myregistry \
  --image intent-recognition .
az container create --resource-group mygroup \
  --name intent-recognition \
  --image myregistry.azurecr.io/intent-recognition
```

## Monitoring & Logging

### Application Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
logger.info("Model loaded successfully")
```

### Performance Monitoring

```python
import time

start = time.time()
result = engine.predict(command)
duration = time.time() - start

logger.info(f"Inference took {duration:.3f}s for: {command}")
```

### Error Tracking

```python
try:
    result = engine.predict(command)
except Exception as e:
    logger.error(f"Inference failed: {e}", exc_info=True)
    return {"error": "Internal server error"}
```

## Versioning & Updates

### Model Versioning

```
models/
├── v1.0/
│   ├── pytorch_model.bin
│   └── config.json
├── v1.1/
│   ├── pytorch_model.bin
│   └── config.json
└── latest → v1.1
```

### APK Versioning

Update `app.json`:
```json
{
  "expo": {
    "version": "1.0.0",
    "android": {
      "versionCode": 1
    }
  }
}
```

### OTA Updates (Expo)

```bash
npm install -g eas-cli
eas update
```

## Rollback Procedure

### Python Service

```bash
# Rollback to previous model
rm bert_tiny_model/pytorch_model.bin
mv bert_tiny_model/pytorch_model.bin.backup bert_tiny_model/pytorch_model.bin
systemctl restart intent-recognition
```

### Mobile App

```bash
# Rollback to previous APK in Play Store
# (Published via Play Store console)
```

## Health Checks

### API Health Endpoint

```python
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "version": "1.0.0"
    })
```

### Monitor Script

```python
import requests
import time

while True:
    try:
        response = requests.get('http://localhost:5000/health')
        if response.status_code != 200:
            print("Health check failed!")
    except Exception as e:
        print(f"Service unavailable: {e}")
    time.sleep(60)
```

## Security

### Environment Variables

```bash
# .env
MODEL_PATH=/path/to/model
FLASK_ENV=production
SECRET_KEY=your-secret-key
```

### Rate Limiting

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

@app.route('/api/predict', methods=['POST'])
@limiter.limit("10/minute")
def predict():
    # ...
```

### CORS

```python
from flask_cors import CORS

CORS(app, resources={
    r"/api/*": {
        "origins": ["https://yourdomain.com"],
        "methods": ["POST"],
        "allow_headers": ["Content-Type"]
    }
})
```

## Performance Optimization

### Model Optimization

```python
# Quantization already done during training
# TFLite model: <10MB
# PyTorch INT8: ~30MB

engine = IntentRecognitionInference(
    model_path="bert_tiny_model",
    device='cuda'  # Use GPU if available
)
```

### Inference Batching

```python
# Batch 10 requests together
commands = [request.json for request in requests_queue]
results = engine.predict_batch(commands)
```

### Caching

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_predict(command):
    return engine.predict(command)
```

## Troubleshooting

### Model Loading Issues

```bash
# Verify model files exist
ls -la bert_tiny_model/

# Check model integrity
python -c "
from inference import IntentRecognitionInference
engine = IntentRecognitionInference()
print('Model loaded successfully')
"
```

### Out of Memory

```python
# Reduce batch size
engine.predict_batch(commands[:10])  # Process 10 at a time
```

### Slow Inference

```bash
# Check device usage
torch.cuda.is_available()  # Should be True for GPU

# Monitor CPU/GPU
nvidia-smi  # GPU usage
top -p $(pgrep -f python)  # CPU usage
```

## Support & Documentation

- **GitHub Issues**: Report bugs and request features
- **Documentation**: `/docs` directory
- **API Docs**: Swagger at `/api/docs` (if configured)
- **Email Support**: support@intentrecognition.com

---

Last Updated: 2024
Version: 1.0.0
