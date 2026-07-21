# Codebase Structure Guide

## Organization Overview

The codebase has been restructured for better maintainability and clarity.

### Root Level

**Configuration & Documentation:**
- `README.md` — Main project documentation with images
- `SETUP.md` — Setup and installation guide
- `DELIVERABLES.md` — Project deliverables
- `STRUCTURE.md` — This file
- `requirements.txt` — Python dependencies
- `requirements-minimal.txt` — Minimal dependencies
- `pytest.ini` — Pytest configuration

### Key Directories

#### `/src` — Core Library
Main source code for the intent recognition system.

```
src/
├── __init__.py
├── model.py                # BERT-Tiny architecture
├── dataset.py              # Dataset utilities
├── train.py                # Training pipeline
├── evaluate.py             # Evaluation metrics
├── quantize.py             # Model quantization
├── compact_model.py        # Compact deployment model
└── utils.py                # Utility functions
```

#### `/scripts` — Utility Scripts
Standalone scripts for data generation, model compression, and inference.

```
scripts/
├── generate_dataset.py           # Generate training dataset
├── create_balanced_dataset.py    # Create balanced dataset
├── create_small_dataset.py       # Create small test dataset
├── compress_model.py             # Compress and quantize model
├── predict_command.py            # CLI inference tool
└── main.py                       # Main entry point
```

#### `/training` — Training Workflows
Training-specific implementations and evaluation.

```
training/
├── train.py                # Full training pipeline
├── train_quick.py          # Quick training (fewer epochs)
├── evaluate.py             # Training evaluation
├── model.py                # Alternative model implementations
└── convert_tflite.py       # TFLite conversion
```

#### `/inference` — Inference Engines
Production inference implementations.

```
inference/
├── __init__.py
├── inference.py            # Core inference engine
├── inference_correct.py    # Corrected inference
└── hybrid_inference.py     # Hybrid inference approach
```

#### `/web` — Web Application
Flask web server and API endpoints.

```
web/
├── web_app.py              # Flask application
├── mobile_api.py           # Mobile API endpoints
└── templates/              # HTML templates
    └── index.html
```

#### `/mobile` — React Native App
Mobile application (Expo/React Native).

```
mobile/
├── App.tsx                 # Main app component
├── app.json                # Expo configuration
├── package.json            # Dependencies
├── screens/                # App screens
├── services/               # Services (including InferenceService)
└── tsconfig.json           # TypeScript configuration
```

#### `/models` — Model Artifacts
Trained and quantized models.

```
models/
├── model_best.pt           # Best trained checkpoint
└── model_quantized_int8.pt # Quantized for mobile
```

#### `/data` — Raw Data
Original datasets before processing.

```
data/
└── parallel_en_ko_ui_intent_10k.csv
```

#### `/dataset` — Processed Datasets
Cleaned and prepared datasets.

```
dataset/
└── dataset_balanced_1932.csv
```

#### `/reports` — Evaluation Reports
Performance metrics and analysis outputs.

```
reports/
├── confusion_matrix_intent.png      # Intent classification matrix
├── confusion_matrix_spatial.png     # Spatial relation matrix
├── confusion_matrix_target.png      # Target type matrix
└── failure_report.csv               # Failed predictions
```

#### `/outputs` — Application Outputs
Screenshots and UI demonstrations.

```
outputs/
├── intent2.png                      # UI screenshot 1
├── intent3.png                      # UI screenshot 2
└── intent_classiffication 1.png    # Classification demo
```

#### `/tests` — Unit Tests
Test suite for the system.

```
tests/
├── test_inference.py       # Inference tests
└── test_dataset.py         # Dataset tests
```

#### `/docs` — Documentation
Detailed documentation.

```
docs/
├── ARCHITECTURE.md         # System architecture
├── DATASET.md              # Dataset format and generation
├── DEPLOYMENT.md           # Deployment guide
├── INFERENCE.md            # Inference pipeline
└── documentation.md        # Comprehensive documentation
```

#### `/bert_tiny_model` — BERT Artifacts
Tokenizer and configuration files.

```
bert_tiny_model/
├── tokenizer.json
└── tokenizer_config.json
```

## Key Changes

### Reorganization (Latest Update)

1. **Scripts Consolidated**
   - All utility scripts moved from root to `/scripts`
   - Includes data generation, compression, and CLI tools

2. **Web App Organized**
   - Flask app moved to `/web/web_app.py`
   - API moved to `/web/mobile_api.py`
   - Templates organized in `/web/templates`

3. **README Enhanced**
   - Added project structure documentation
   - Integrated UI screenshots and confusion matrices
   - Organized usage examples
   - Added image references for GitHub display

4. **Git Cleanup**
   - Large model files removed from history
   - `.gitignore` expanded with comprehensive rules
   - Unnecessary files excluded

## Running Commands

### After Reorganization

```bash
# Generate dataset
python scripts/generate_dataset.py

# Training
python training/train.py

# Inference
python scripts/predict_command.py "your command"

# Web app
python web/web_app.py

# Evaluation
python training/evaluate.py
```

## File Naming Conventions

- **Python files**: `snake_case.py`
- **TypeScript files**: `PascalCase.ts` / `camelCase.ts`
- **Directories**: `lowercase` or `snake_case`
- **Test files**: `test_*.py` in `/tests`

## Image References in README

All images are displayed in the README and GitHub will automatically render them:

- **UI Screenshots**: Located in `/outputs/`
- **Metrics**: Located in `/reports/`
- Markdown references use relative paths: `![alt](path/to/image.png)`

## Notes

- All __pycache__ directories are in `.gitignore`
- Virtual environment (`venv/`) is in `.gitignore`
- Node modules (`mobile/node_modules/`) are in `.gitignore`
- Large model files are in `.gitignore`
- All changes have been pushed to the remote repository
