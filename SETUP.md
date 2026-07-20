# Quick Setup Guide

## Environment Setup

### Option 1: Full Setup (Training + Inference)

```bash
# Create virtual environment
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Time**: 5-10 minutes  
**Includes**: PyTorch, Transformers, TensorFlow, evaluation tools

### Option 2: Minimal Setup (Inference Only)

```bash
# Create virtual environment
python -m venv venv

# Activate
venv\Scripts\activate

# Install minimal dependencies
pip install -r requirements-minimal.txt
```

**Time**: 2-3 minutes  
**Includes**: PyTorch, Transformers only (no training)

### Option 3: Mobile Only

```bash
cd mobile
npm install
```

**Time**: 3-5 minutes  
**Includes**: React Native + Expo

---

## Verification

### Test Python Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### Test Dataset

```bash
python -c "
import csv
with open('dataset/dataset.csv') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    print(f'Dataset: {len(rows)} samples loaded')
"
```

### Test Inference

```bash
python -c "
from inference import IntentRecognitionInference
engine = IntentRecognitionInference()
result = engine.predict('click the button')
print('Inference works!')
print(f'Intent: {result[\"intent\"]}')
print(f'Confidence: {result[\"confidence\"]}')
"
```

### Test Mobile

```bash
cd mobile
npm install
npx expo start
# Scan QR code with Expo app or press 'a' for Android simulator
```

---

## Troubleshooting

### PyTorch Installation Issues

**Problem**: "No matching distribution found for torch==X.X.X"

**Solution**: Use flexible versions
```bash
pip install torch transformers --upgrade
```

**Or** use CPU-only version (smaller):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Memory Issues

**Problem**: "RuntimeError: CUDA out of memory" or slow installation

**Solution**: Increase virtual memory or use CPU
```bash
# Use CPU during training
export CUDA_VISIBLE_DEVICES=""
python training/train.py
```

### Missing Model Files

**Problem**: "FileNotFoundError: Model not found"

**Solution**: Ensure model exists
```bash
ls -la bert_tiny_model/
```

If missing, train the model:
```bash
python training/train.py
```

### Mobile NPM Issues

**Problem**: "npm ERR! code ERESOLVE"

**Solution**: Use legacy dependency resolver
```bash
cd mobile
npm install --legacy-peer-deps
```

---

## Quick Start

### 1. Dataset Only (View)
```bash
head -10 dataset/dataset.csv
wc -l dataset/dataset.csv  # Check row count
```

### 2. Inference (No Training)
```bash
python inference/inference.py
```

### 3. Mobile App
```bash
cd mobile
npm install
npx expo start
```

### 4. Training (Full)
```bash
cd training
python train.py        # Train model
python evaluate.py     # Evaluate metrics
```

### 5. Testing
```bash
pytest tests/ -v       # Run all tests
pytest tests/test_inference.py -v   # Inference tests only
pytest tests/test_dataset.py -v     # Dataset tests only
```

---

## System Requirements

### Minimum
- Python 3.9+
- 4GB RAM
- 5GB disk space

### Recommended for Training
- Python 3.10+
- 16GB+ RAM
- GPU (NVIDIA with CUDA)
- 10GB+ disk space

### Mobile
- Node.js 16+
- npm or yarn
- Android SDK (for APK build)
- Expo account (free, for publishing)

---

## Installation Logs

Save your installation logs for troubleshooting:

```bash
# Full log
pip install -r requirements.txt 2>&1 | tee install.log

# Just errors
pip install -r requirements.txt 2>&1 | grep -i error
```

---

## Next Steps

1. **Dataset**: View at `dataset/dataset.csv`
2. **Documentation**: Read `README.md` and `docs/ARCHITECTURE.md`
3. **Inference**: Run `python inference/inference.py`
4. **Mobile**: Launch `npx expo start` from `mobile/`
5. **Training**: Run `python training/train.py` (if GPU available)

---

## Support

- Check **README.md** for overview
- Read **docs/** for detailed guides
- Review **tests/** for examples
- See **DELIVERABLES.md** for project summary

---

**Last Updated**: July 20, 2024
