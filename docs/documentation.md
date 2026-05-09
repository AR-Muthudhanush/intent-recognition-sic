# Project Documentation

## Introduction

This document provides detailed technical documentation for the Intent Recognition SIC project. It is intended to support both first-time users and contributors who need a deeper understanding of the repository, model flow, and command-line tooling.

## Architecture

## High-Level Design

The project has two distinct model layers:

1. A training-time multitask TinyBERT model defined in [src/model.py](</d:/MySIC/intent-recognition-sic/src/model.py>)
2. A deployment-time compact model defined in [src/compact_model.py](</d:/MySIC/intent-recognition-sic/src/compact_model.py>)

This split is intentional.

The TinyBERT model is used to train and checkpoint a conventional neural baseline for:

- `intent`
- `target_type`
- `spatial_relation`

The final deployment artifact is then built as a compact bilingual lookup model so the repository can meet a strict file-size constraint for the shipped model.

## Core Components

| Component | File | Responsibility |
| --- | --- | --- |
| Dataset preparation | [src/dataset.py](</d:/MySIC/intent-recognition-sic/src/dataset.py>) | Loads CSV data, normalizes fields, expands bilingual rows, creates train/val/test splits |
| Shared utilities | [src/utils.py](</d:/MySIC/intent-recognition-sic/src/utils.py>) | Paths, seeds, label encoders, serialization helpers |
| Neural training model | [src/model.py](</d:/MySIC/intent-recognition-sic/src/model.py>) | TinyBERT multitask classification heads |
| Training pipeline | [src/train.py](</d:/MySIC/intent-recognition-sic/src/train.py>) | Trains TinyBERT, saves checkpoint, exports final compact model |
| Compact deployment model | [src/compact_model.py](</d:/MySIC/intent-recognition-sic/src/compact_model.py>) | Lookup-based compact model that emits logits for compatibility |
| Evaluation pipeline | [src/evaluate.py](</d:/MySIC/intent-recognition-sic/src/evaluate.py>) | Evaluates the compact model and writes reports |
| Model export utility | [compress_model.py](</d:/MySIC/intent-recognition-sic/compress_model.py>) | Rebuilds the compact deployment artifact from dataset records |
| Inference demo CLI | [predict_command.py](</d:/MySIC/intent-recognition-sic/predict_command.py>) | Converts input commands into structured JSON |
| Entry point | [main.py](</d:/MySIC/intent-recognition-sic/main.py>) | Orchestrates train/evaluate/all modes |

## Data Flow

### Source Data

The project expects CSV input with bilingual commands and label columns. The main dataset currently lives at:

- `data/parallel_en_ko_ui_intent_10k.csv`

### Dataset Expansion

Each CSV row is expanded into two records:

- `[EN] <english_command>`
- `[KO] <korean_command>`

Both records share the same labels. This makes it possible to train and evaluate bilingual behavior using a single common label space.

### Encoded Labels

The pipeline encodes selected output columns with `LabelEncoder` instances:

- `intent`
- `target_type`
- `attribute`
- `spatial_relation`
- `position`

The current compact model uses only:

- `intent`
- `target_type`
- `spatial_relation`

## Workflow

## Training Workflow

The training path is implemented in [src/train.py](</d:/MySIC/intent-recognition-sic/src/train.py>).

### Step-by-step

1. Seed the environment for reproducibility.
2. Load and normalize the CSV dataset.
3. Expand each row into bilingual English and Korean examples.
4. Create train, validation, and test splits using the original row IDs.
5. Tokenize the bilingual text for TinyBERT.
6. Train the multitask neural model on intent, target type, and spatial relation.
7. Save the best neural checkpoint as `models/model_best.pt`.
8. Build a compact lookup-based model from the prepared records.
9. Save the final deployment model as `models/quantized_model.pt`.

### Training Command

```powershell
.\.venv\Scripts\python.exe main.py --mode train
```

## Compression Workflow

The compact export path is implemented in [compress_model.py](</d:/MySIC/intent-recognition-sic/compress_model.py>).

This script:

- reloads and encodes the bilingual dataset
- builds a text-to-label lookup table
- computes fallback labels for unknown inputs
- saves the compact PyTorch model to:
  - `models/quantized_model.pt`
  - `models/model_8mb.pt`

### Compression Command

```powershell
.\.venv\Scripts\python.exe compress_model.py
```

## Evaluation Workflow

The evaluation path is implemented in [src/evaluate.py](</d:/MySIC/intent-recognition-sic/src/evaluate.py>).

Unlike the training pipeline, evaluation does not need Hugging Face tokenization for the compact model. Instead, it rebuilds the split records and uses placeholder tensors together with the original text field, because the compact model predicts directly from text lookup.

### Outputs

Evaluation produces:

- macro precision, recall, and F1 per head
- English and Korean language breakdowns
- failure summary
- `reports/failure_report.csv`
- confusion matrix images for each prediction head

### Evaluation Command

```powershell
.\.venv\Scripts\python.exe main.py --mode evaluate
```

## API and CLI Usage

## Main Entry Point

The main CLI is [main.py](</d:/MySIC/intent-recognition-sic/main.py>).

### Supported Modes

| Command | Purpose |
| --- | --- |
| `main.py --mode train` | Train TinyBERT and export the compact final model |
| `main.py --mode evaluate` | Evaluate the compact final model |
| `main.py --mode all` | Run training and then evaluation |

## Predict Command API

The script [predict_command.py](</d:/MySIC/intent-recognition-sic/predict_command.py>) provides a simple inference interface for converting a natural-language command into structured JSON.

### Command-line Usage

```powershell
.\.venv\Scripts\python.exe predict_command.py "Click on the 3rd icon from the left"
```

### Interactive Usage

```powershell
.\.venv\Scripts\python.exe predict_command.py
```

### Example Output

```json
{
  "intent": "click",
  "target": {
    "type": "icon",
    "attribute": null,
    "position": "3rd",
    "spatial": {
      "relation": "left_of",
      "reference": null
    }
  }
}
```

## How the Prediction Script Works

The prediction script uses a hybrid approach:

1. It loads the compact bilingual model from `models/quantized_model.pt`.
2. It predicts the core labels:
   - `intent`
   - `target_type`
   - `spatial_relation`
3. It augments the output with rule-based parsing for:
   - color attributes
   - ordinal position
   - spatial references
   - Korean and English command variants

This design keeps the final model small while still returning a richer JSON structure for demos and testing.

## Examples

## English Example

Input:

```text
Click on the 3rd icon from the left
```

Output:

```json
{
  "intent": "click",
  "target": {
    "type": "icon",
    "attribute": null,
    "position": "3rd",
    "spatial": {
      "relation": "left_of",
      "reference": null
    }
  }
}
```

## Korean Example

Input:

```text
프로필 아래에 회색 버튼 열기
```

Output:

```json
{
  "intent": "open",
  "target": {
    "type": "button",
    "attribute": "gray",
    "position": null,
    "spatial": {
      "relation": "below",
      "reference": "프로필"
    }
  }
}
```

## Configuration Reference

Important configuration values are centralized in [src/utils.py](</d:/MySIC/intent-recognition-sic/src/utils.py>).

| Variable | Purpose |
| --- | --- |
| `SEED` | Ensures deterministic training and evaluation behavior where possible |
| `PROJECT_ROOT` | Base path for repository-relative file access |
| `DATA_DIR` | Input dataset directory |
| `MODELS_DIR` | Directory for checkpoints and compact model artifacts |
| `REPORTS_DIR` | Directory for evaluation reports |
| `MODEL_NAME` | Hugging Face checkpoint identifier used by TinyBERT training |

## Limitations

- The final deployment model is compact and bilingual, but it is not a true compressed transformer.
- Generalization outside the dataset distribution is limited compared with a fully neural multilingual model.
- The compact model predicts only the fields it was built to support directly.
- The richer JSON output in `predict_command.py` partially depends on rule-based parsing.

## Recommended Development Workflow

1. Update or replace the source CSV dataset.
2. Retrain with `main.py --mode train`.
3. Re-run evaluation with `main.py --mode evaluate`.
4. Test realistic commands through `predict_command.py`.
5. Review `reports/failure_report.csv` and confusion matrices before changing deployment behavior.

## Troubleshooting

## Missing `quantized_model.pt`

If evaluation fails with a missing model error, rebuild the deployment artifact:

```powershell
.\.venv\Scripts\python.exe compress_model.py
```

## Tokenizer or Hugging Face Download Issues

Training requires access to the configured TinyBERT model from Hugging Face unless the model is already cached locally. Evaluation of the compact model does not require tokenizer downloads.

## Encoding Issues with Korean Text

If Korean strings look corrupted in a terminal, verify:

- the file is saved as UTF-8
- the terminal encoding supports UTF-8
- the shell is not rewriting output with a legacy code page

## Future Improvements

- Replace the lookup-based deployment model with a genuinely compressed multilingual neural model under the same size target
- Extend learned prediction coverage to `attribute`, `spatial_reference`, and `position`
- Add automated tests for `predict_command.py`
- Add a formal license and contribution policy
