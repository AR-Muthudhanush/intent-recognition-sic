# Inference Guide

## Quick Start

### Python (Offline)

```python
from inference.inference import IntentRecognitionInference

# Initialize
engine = IntentRecognitionInference(device='cpu')

# Predict
result = engine.predict("click the red submit button")

# Output
print(result)
# {
#   "intent": "click",
#   "confidence": 0.98,
#   "target": {
#     "type": "button",
#     "attribute": "red",
#     "label": "submit",
#     "index": null,
#     "relation": null,
#     "reference": null
#   }
# }
```

### Mobile (React Native)

```typescript
import InferenceService from './services/InferenceService';

// Single prediction
const result = await InferenceService.predict("scroll down");
console.log(result.intent);  // "scroll"

// Batch prediction
const commands = ["click button", "scroll down", "delete message"];
const results = await InferenceService.predict_batch(commands);
```

## API Reference

### `IntentRecognitionInference`

Main inference class for PyTorch models.

#### Constructor

```python
IntentRecognitionInference(
    model_path: str = "bert_tiny_model",
    device: str = "cpu"
)
```

**Parameters**:
- `model_path`: Path to trained model directory
- `device`: "cpu" or "cuda" (if available)

**Example**:
```python
# CPU inference
engine = IntentRecognitionInference()

# GPU inference (if available)
engine = IntentRecognitionInference(device='cuda')

# Custom model path
engine = IntentRecognitionInference(
    model_path="/path/to/model",
    device='cpu'
)
```

#### `predict(command: str) → dict`

Predict intent for a single command.

**Parameters**:
- `command`: Natural language UI command

**Returns**: Prediction dictionary

**Example**:
```python
result = engine.predict("delete the file")
# {
#   "intent": "delete",
#   "confidence": 0.95,
#   "target": {...}
# }
```

#### `predict_batch(commands: list[str]) → list[dict]`

Predict intents for multiple commands.

**Parameters**:
- `commands`: List of command strings

**Returns**: List of prediction dictionaries

**Example**:
```python
results = engine.predict_batch([
    "click button",
    "scroll down",
    "save file"
])
# [{...}, {...}, {...}]
```

## Output Format

### Response Structure

```json
{
  "intent": "string",
  "confidence": 0.0-1.0,
  "target": {
    "type": "string|null",
    "attribute": "string|null",
    "label": "string|null",
    "index": "string|null",
    "relation": "string|null",
    "reference": "string|null"
  }
}
```

### Fields

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `intent` | string | 62 types | Predicted UI action |
| `confidence` | float | 0.0-1.0 | Prediction confidence |
| `target.type` | string/null | 40+ types | UI element type |
| `target.attribute` | string/null | any | Visual properties |
| `target.label` | string/null | any | Element text |
| `target.index` | string/null | "1"/"first"/etc | Position |
| `target.relation` | string/null | 40+ types | Spatial relation |
| `target.reference` | string/null | any | Reference element |

### Intent Values

```
click, tap, double_tap, long_press, scroll, swipe, drag, drop,
delete, copy, paste, highlight, select, open, close, play, pause,
stop, search, zoom, rotate, move, resize, upload, download, share,
save, refresh, retry, enable, disable, check, uncheck, expand,
collapse, accept, reject, approve, back, next, previous, home,
settings, login, logout, bookmark, pin, unpin, start, finish, exit,
continue, add, remove, hide, show, mute, unmute, install, uninstall,
submit, cancel, type, enter, focus, hover, launch
```

### Target Values

```
button, icon, image, checkbox, switch, radio_button, dropdown,
textbox, input, text, chart, graph, table, card, menu,
navigation_bar, toolbar, search_box, video, audio, attachment,
file, document, message, notification, popup, dialog, tab,
list_item, profile, avatar, calendar, slider, progress_bar,
link, qr_code, camera, gallery, email, password, phone_number
```

## Error Handling

### Python

```python
from inference.inference import IntentRecognitionInference

engine = IntentRecognitionInference()

try:
    result = engine.predict("some command")
    if result['confidence'] < 0.5:
        print("Low confidence prediction")
except Exception as e:
    print(f"Inference failed: {e}")
```

### Mobile (TypeScript)

```typescript
try {
  const result = await InferenceService.predict(command);
  if (result.confidence < 0.5) {
    console.warn("Low confidence");
  }
} catch (error) {
  console.error("Prediction failed:", error);
}
```

## Performance

### Latency

| Scenario | Latency |
|----------|---------|
| Single inference (CPU) | 50-100ms |
| Batch of 10 (CPU) | 80-150ms total |
| Single inference (GPU) | 10-30ms |
| Batch of 10 (GPU) | 20-60ms total |

### Memory

| Type | Usage |
|------|-------|
| Model weights | <10MB |
| Runtime memory | ~50MB |
| Cache (100 samples) | ~5MB |

### Optimization Tips

1. **Batch Inference**: Process multiple commands together
   ```python
   # Slow: individual predictions
   for cmd in commands:
       result = engine.predict(cmd)
   
   # Fast: batch prediction
   results = engine.predict_batch(commands)
   ```

2. **Use GPU**: If available
   ```python
   engine = IntentRecognitionInference(device='cuda')
   ```

3. **Caching**: Cache tokenizer
   ```python
   engine.tokenizer.save_pretrained('./cache')
   ```

## Configuration

### Model Variants

**Default (BERT-Tiny Multilingual)**
- Size: <10MB
- Latency: 50-100ms
- Accuracy: ~95%

**Custom Model**
```python
engine = IntentRecognitionInference(
    model_path="/path/to/custom/model"
)
```

### Device Selection

**Auto-detect**
```python
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
engine = IntentRecognitionInference(device=device)
```

**Force CPU**
```python
engine = IntentRecognitionInference(device='cpu')
```

**Force GPU**
```python
engine = IntentRecognitionInference(device='cuda:0')
```

## Advanced Usage

### Custom Target Extraction

```python
class CustomInference(IntentRecognitionInference):
    def extract_target_info(self, command):
        # Custom extraction logic
        target_info = super().extract_target_info(command)
        # Modify as needed
        return target_info
```

### Prediction with Confidence Threshold

```python
def predict_confident(engine, command, threshold=0.8):
    result = engine.predict(command)
    if result['confidence'] < threshold:
        return None
    return result
```

### Batch Processing with Progress

```python
from tqdm import tqdm

def predict_batch_progress(engine, commands):
    results = []
    for cmd in tqdm(commands, desc="Predicting"):
        results.append(engine.predict(cmd))
    return results
```

## Mobile Integration

### React Native

```typescript
// InferenceService.ts
export const predict = async (command: string) => {
  try {
    // Local inference (no server call)
    const result = performInference(command);
    return result;
  } catch (error) {
    console.error('Inference failed:', error);
    return fallbackResult();
  }
};
```

### Persisting Results

```typescript
import AsyncStorage from '@react-native-async-storage/async-storage';

const saveResult = async (result) => {
  const history = await AsyncStorage.getItem('history');
  const items = history ? JSON.parse(history) : [];
  items.push({
    timestamp: Date.now(),
    ...result
  });
  await AsyncStorage.setItem('history', JSON.stringify(items));
};
```

## Testing

### Unit Tests

```bash
pytest tests/test_inference.py -v
```

### Manual Testing

```python
test_cases = [
    "click the button",
    "scroll down",
    "delete this message",
    "search for something",
    "open the menu",
]

engine = IntentRecognitionInference()
for cmd in test_cases:
    result = engine.predict(cmd)
    print(f"{cmd} → {result['intent']}")
```

## Troubleshooting

### Model Not Found

```
FileNotFoundError: Model not found
```

**Solution**: Check model path and ensure model files exist
```python
from pathlib import Path
model_path = Path('bert_tiny_model')
assert model_path.exists(), f"Model not found at {model_path}"
```

### Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution**: Use CPU or reduce batch size
```python
engine = IntentRecognitionInference(device='cpu')
```

### Low Confidence

```python
result = engine.predict("ambiguous command")
# {"confidence": 0.4, ...}
```

**Solution**: Filter by confidence threshold
```python
if result['confidence'] < 0.7:
    return "Unable to understand command"
```

## Best Practices

1. **Always check confidence**: Filter low-confidence predictions
2. **Batch process**: Use batch API for multiple commands
3. **Cache results**: Store predictions for identical inputs
4. **Error handling**: Wrap predictions in try-except
5. **Log predictions**: Track inference for analytics
6. **Version model**: Keep track of model versions
7. **Validate output**: Check JSON schema before using

## FAQ

**Q: Can I use the model offline?**
A: Yes, all inference runs locally. No internet required.

**Q: What languages are supported?**
A: English and Korean. Adding more languages requires retraining.

**Q: Can I fine-tune the model?**
A: Yes, see training documentation for fine-tuning instructions.

**Q: Is GPU required?**
A: No, CPU inference is supported. GPU is optional for faster processing.

**Q: How do I update the model?**
A: Retrain on new data, export, and replace model files.
