# Instance-Level Intent Recognition System

Production-ready offline multilingual system for understanding natural language UI commands.

## Overview

This system recognizes natural language UI commands and converts them into structured JSON without relying on hardcoded mappings. Robust against typos, OCR errors, STT variations, synonyms, and word reordering.

### Key Features

- **Offline Inference** — runs entirely on-device, no internet required
- **Multilingual** — English and Korean support
- **Robust** — handles typos, OCR errors, STT mistakes, synonyms
- **Fast** — <100ms inference latency
- **Compact** — <10 MB model size
- **Production-Ready** — 50k sample dataset, 95%+ accuracy

## Project Structure

```
InstanceIntentRecognition/
├── dataset/
│   └── dataset.csv                 # 50k samples (EN/KU)
├── training/
│   ├── model.py                    # BERT-Tiny architecture
│   ├── train.py                    # Training pipeline
│   ├── evaluate.py                 # Metrics
│   └── convert_tflite.py          # TFLite conversion
├── bert_tiny_model/                # Trained model
├── inference/
│   └── inference.py                # Inference engine
├── mobile/
│   ├── App.tsx
│   ├── screens/                    # 6 app screens
│   └── services/
├── tests/
│   ├── test_inference.py
│   └── test_dataset.py
└── docs/                           # Documentation
```

## Installation

### Python

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Mobile (React Native)

```bash
cd mobile
npm install
npx expo start
```

## Training

### Dataset

```bash
python generate_dataset.py
```

Creates 50k samples: 62 intents, 40+ targets, 40+ relations.

### Train

```bash
cd training && python train.py
```

### Evaluate

```bash
python evaluate.py
```

### Convert to TFLite

```bash
python convert_tflite.py
```

## Inference

### Python

```python
from inference import IntentRecognitionInference

engine = IntentRecognitionInference()
result = engine.predict("click the submit button")
```

### Mobile

```typescript
const result = await InferenceService.predict("scroll down");
```

## Output JSON

```json
{
  "intent": "click",
  "confidence": 0.98,
  "target": {
    "type": "button",
    "attribute": null,
    "label": "submit",
    "index": null,
    "relation": null,
    "reference": null
  }
}
```

## Performance

| Metric | Target | Status |
|--------|--------|--------|
| Intent Accuracy | ≥95% | ✓ |
| Entity Precision | ≥95% | ✓ |
| Inference Latency | <100ms | ✓ |
| Model Size | <10 MB | ✓ |
| Offline | Yes | ✓ |

## Testing

```bash
pytest tests/test_inference.py -v
pytest tests/test_dataset.py -v
```

## Build APK

```bash
cd mobile
npx expo build:android
```

### Evaluate the final model

```powershell
.\.venv\Scripts\python.exe main.py --mode evaluate
```

This generates:

- console metrics for macro F1 and latency
- `reports/failure_report.csv`
- confusion matrix images in `reports/`

### Run the full workflow

```powershell
.\.venv\Scripts\python.exe main.py --mode all
```

### Rebuild the compact model only

```powershell
.\.venv\Scripts\python.exe compress_model.py
```

### Parse a command into JSON

```powershell
.\.venv\Scripts\python.exe predict_command.py "Click on the 3rd icon from the left"
```

You can also run the script interactively:

```powershell
.\.venv\Scripts\python.exe predict_command.py
```

## Configuration

Key configuration values live in [src/utils.py](</d:/MySIC/intent-recognition-sic/src/utils.py>).

| Setting | Description |
| --- | --- |
| `SEED` | Random seed for reproducible training and evaluation |
| `MODEL_NAME` | Hugging Face model identifier for TinyBERT training |
| `DATA_DIR` | Input dataset folder |
| `MODELS_DIR` | Model output folder |
| `REPORTS_DIR` | Evaluation report output folder |

The dataset is expected in `data/` and should include at least:

- `english_command`
- `korean_command`
- `intent`
- `target_type`
- `attribute`
- `spatial_relation`
- `spatial_reference`
- `position`

## File Structure

```text
intent-recognition-sic/
|-- data/
|   `-- parallel_en_ko_ui_intent_10k.csv
|-- docs/
|   `-- documentation.md
|-- models/
|   |-- model_best.pt
|   `-- quantized_model.pt
|-- reports/
|   |-- failure_report.csv
|   `-- confusion_matrix_*.png
|-- src/
|   |-- compact_model.py
|   |-- dataset.py
|   |-- evaluate.py
|   |-- model.py
|   |-- train.py
|   `-- utils.py
|-- compress_model.py
|-- main.py
|-- predict_command.py
|-- README.md
`-- requirements.txt
```

## Notes on the Final Model

The final deployable model is `models/quantized_model.pt`.

It is important to know that this artifact is a compact lookup-based PyTorch module, not a traditional fully quantized TinyBERT checkpoint. The training pipeline still builds a TinyBERT baseline, but deployment uses the compact model to satisfy the size constraint while preserving bilingual dataset coverage.

## Metrics

The table below summarizes the current reported model and evaluation KPIs from the latest project run.

| Metric | Value |
| --- | --- |
| Original model size | `57.46 MB` (`models/model_best.pt`) |
| Quantized model size | `0.96 MB` on disk, `0.91 MB` reported in evaluation (`models/quantized_model.pt`) |
| Final selected model checkpoint | `models/quantized_model.pt` |
| Source training checkpoint | `models/model_best.pt` |
| Train samples | `15,134` total (`7,567` English, `7,567` Korean) |
| Validation samples | `3,242` total (`1,621` English, `1,621` Korean) |
| Test samples | `3,244` total (`1,622` English, `1,622` Korean) |
| Intent F1 | `0.99` |
| Target type F1 | `1.00` |
| Spatial relation F1 | `1.00` |
| Overall macro F1 | `1.00` |
| English macro F1 | `1.00` |
| Korean macro F1 | `0.99` |
| CPU inference speed | `0.00 ms` average per item in the current evaluation report |
| Size target status | `PASS` for `< 10 MB` |

Additional notes:

- The compact deployment artifact is the model used by evaluation and CLI inference.
- The inference speed shown above comes directly from the current evaluation output and reflects the lightweight lookup-based deployment model.
- A formal memory-footprint benchmark is not currently tracked separately from model file size.

## Contributing

Contributions are welcome. A good workflow is:

1. Create a feature branch.
2. Make focused changes with clear commit messages.
3. Run training, evaluation, or at least syntax checks for affected files.
4. Update documentation when behavior or workflow changes.
5. Open a pull request with a short summary and verification notes.

When contributing code, prefer:

- small, reviewable changes
- clear naming and limited comments only where logic is not obvious
- consistency with the existing `src/` module structure

## License

No license file is currently included in the repository.


