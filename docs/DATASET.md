# Dataset Documentation

## Overview

The Intent Recognition Dataset contains **50,000 diverse samples** of natural language UI commands with multilingual support (English and Korean).

**Location**: `dataset/dataset.csv`

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 50,000 |
| Intents Covered | 62 |
| Target Objects | 40+ |
| Spatial Relations | 40+ |
| Languages | 2 (English, Korean) |
| Average Command Length | 15-20 words |
| Encoding | UTF-8 |
| Format | CSV (comma-separated) |

## CSV Columns

### 1. `command` (string)

The natural language UI command in English or Korean.

**Examples**:
- "click the submit button"
- "scroll down to the bottom"
- "제출 버튼을 클릭해" (Korean)
- "delete this message quickly"
- "clck the red buton" (typo)

**Characteristics**:
- Conversational language
- Typos and OCR errors
- STT variations
- Multiple phrasings for same action
- 5-100+ characters

### 2. `intent` (string)

The classified UI action intent.

**62 Supported Intents**:
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

### 3. `target` (string)

The UI element type being acted upon.

**40+ Supported Targets**:
```
button, icon, image, checkbox, switch, radio_button, dropdown, textbox, 
input, text, chart, graph, table, card, menu, navigation_bar, toolbar, 
search_box, video, audio, attachment, file, document, message, 
notification, popup, dialog, tab, list_item, profile, avatar, calendar, 
slider, progress_bar, link, qr_code, camera, gallery, email, password, 
phone_number
```

**Examples**:
- "button"
- "input"
- "navigation_bar"
- "" (empty when not specified)

### 4. `attribute` (string)

Visual or state attributes of the target element.

**Common Attributes**:
```
red, blue, green, yellow, large, small, disabled, active, highlighted, 
focused, expanded, collapsed, primary, secondary, danger, warning, 
success, info, dark, light, bold, italic, underlined, strikethrough
```

**Examples**:
- "red" (red button)
- "large" (large icon)
- "disabled" (disabled input)
- "" (empty if no attribute)

### 5. `label` (string)

The text label displayed on the UI element.

**Common Labels**:
```
submit, cancel, ok, save, delete, edit, add, remove, search, filter, 
sort, export, import, share, download, upload, refresh, retry, back, 
next, previous, home, settings, profile, logout, login, register, 
password, email, phone, address, message, notification, alert, confirm, 
warning, error, success, info, loading, empty
```

**Examples**:
- "submit" (submit button)
- "search" (search box)
- "cancel" (cancel dialog)
- "" (empty if no visible label)

### 6. `index` (string)

Position or ordinal index of the element.

**Values**:
```
"0", "1", "2", ..., "99" (numeric indices)
"first", "second", "third", "last", "second_last"
"first_row", "second_row", "first_column", "last_column"
"" (empty if position not specified)
```

**Examples**:
- "first" (first button)
- "3" (fourth item)
- "last" (last element)

### 7. `relation` (string)

Spatial relationship to a reference element.

**40+ Supported Relations**:
```
top, bottom, left, right, top_left, top_right, bottom_left, 
bottom_right, center, center_left, center_right, upper_center, 
lower_center, above, below, beside, adjacent_to, next_to, inside, 
outside, between, before, after, behind, in_front_of, beneath, over, 
under, nearest, farthest, first, second, third, last, second_last, 
first_row, second_row, first_column, last_column, top_edge, 
bottom_edge, left_edge, right_edge
```

**Examples**:
- "top" (at top of screen)
- "above" (above another element)
- "beside" (beside a reference)
- "bottom_right" (bottom-right corner)
- "" (empty if no spatial relation)

### 8. `reference` (string)

Reference element for relative positioning.

**Common References**:
```
cart icon, menu button, search bar, profile icon, settings button, 
back button, submit button, cancel button, save button, delete button, 
edit button, add button, remove button, refresh button, home button, 
sidebar menu, top navigation, footer, header, main content, dialog box, 
popup, modal, card list, image gallery, video player, audio player, 
map view, calendar view, table view, chart view
```

**Examples**:
- "cart icon"
- "search bar"
- "settings button"
- "" (empty if no reference)

## Data Quality Metrics

### Coverage

- **Intent Distribution**: Balanced (no single intent >5% of dataset)
- **Target Coverage**: All 40+ targets represented
- **Spatial Relation Coverage**: All 40+ relations present
- **Language Mix**: ~50% English, ~50% Korean

### Variations

| Type | Coverage | Examples |
|------|----------|----------|
| Typos | ~30% of samples | "clik" → "click" |
| OCR Errors | ~25% of samples | "c1ick" → "click" |
| STT Errors | ~25% of samples | "klik" → "click" |
| Punctuation Variants | ~20% of samples | "click!" vs "click" |
| Capitalization | ~15% of samples | "CLICK" vs "click" |
| Word Reordering | ~15% of samples | "button the click" |
| Filler Words | ~10% of samples | "just click the button" |
| Passive Voice | ~8% of samples | "the button is clicked" |
| Abbreviations | ~5% of samples | "msg" → "message" |
| Synonyms | ~10% of samples | "tap" → "click" |

### Duplicate Rate

- Intentional variations: ~20%
- Accidental duplicates: <2%
- Unique commands: >80%

## Dataset Generation

### Process

1. **Intent Sampling**: Select intent distribution
2. **Command Generation**: Create variations
   - Base command
   - Typos (random character substitution)
   - OCR errors (character confusion)
   - STT errors (phonetic similarity)
   - Word reordering
   - Filler words
   - Passive voice
3. **Entity Assignment**: Extract/assign targets, attributes, relations
4. **Multilingual Support**: Translate/adapt to Korean
5. **Validation**: Check for empty fields, duplicates
6. **Shuffle & Export**: Random order, save to CSV

### Code

See `generate_dataset.py`:
- `introduce_typo()`: Realistic typos
- `introduce_ocr_error()`: Character confusion
- `introduce_stt_error()`: Phonetic errors
- `generate_command_variations()`: Multiple phrasings
- `generate_bilingual_commands()`: EN/KU variants

## Usage Examples

### Load Dataset

**Python (pandas)**:
```python
import pandas as pd

df = pd.read_csv('dataset/dataset.csv')
print(df.shape)  # (50000, 8)
print(df.head())
```

**Python (csv)**:
```python
import csv

with open('dataset/dataset.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row['command'], row['intent'])
```

### Filtering

**Get all 'click' intents**:
```python
clicks = df[df['intent'] == 'click']
```

**Get all 'button' targets**:
```python
buttons = df[df['target'] == 'button']
```

**Get commands with attributes**:
```python
with_attrs = df[df['attribute'] != '']
```

### Statistics

**Intent distribution**:
```python
df['intent'].value_counts()
```

**Target types**:
```python
df['target'].value_counts()
```

**Spatial relations**:
```python
df['relation'].value_counts()
```

## Validation

### Unit Tests

Run dataset validation:
```bash
pytest tests/test_dataset.py -v
```

Tests verify:
- File exists and has 50k rows
- All required columns present
- No empty commands/intents
- Intent coverage ≥30
- Target coverage ≥20
- Relation coverage ≥15
- Balanced distribution
- Language mix present
- Low duplicate rate

## Licensing & Usage

- **License**: Open source (MIT)
- **Attribution**: Optional
- **Commercial Use**: Permitted
- **Modifications**: Permitted
- **Redistribution**: Permitted

## Contact & Support

For dataset issues:
- Report bugs: GitHub Issues
- Request features: GitHub Discussions
- Send feedback: [email]

## Changelog

### v1.0.0 (Initial Release)
- 50,000 samples
- 62 intents
- 40+ targets
- 40+ spatial relations
- English + Korean
- Typo, OCR, STT variations
