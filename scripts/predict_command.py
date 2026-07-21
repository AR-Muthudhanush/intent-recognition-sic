from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from src.compact_model import CompactIntentModel
from src.utils import MODELS_DIR, load_pickle


COLOR_WORDS = {
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "pink",
    "gray",
    "grey",
    "black",
    "white",
}

KOREAN_COLOR_WORDS = {
    "빨간색": "red",
    "주황색": "orange",
    "노란색": "yellow",
    "녹색": "green",
    "초록색": "green",
    "파란색": "blue",
    "보라색": "purple",
    "분홍색": "pink",
    "회색": "gray",
    "검정색": "black",
    "흰색": "white",
}

TARGET_TYPES = {
    "button",
    "text",
    "image",
    "icon",
    "checkbox",
    "check box",
    "link",
    "menu",
    "tab",
    "switch",
}

KOREAN_TARGET_TYPES = {
    "버튼": "button",
    "텍스트": "text",
    "글자": "text",
    "이미지": "image",
    "아이콘": "icon",
    "체크박스": "checkbox",
    "체크 박스": "checkbox",
    "링크": "link",
    "메뉴": "menu",
    "탭": "tab",
    "스위치": "switch",
}

SPATIAL_PATTERNS = [
    ("left_of", r"\b(?:to the left of|left of)\s+(.+)$"),
    ("right_of", r"\b(?:to the right of|right of)\s+(.+)$"),
    ("above", r"\b(?:above|on top of|over)\s+(.+)$"),
    ("below", r"\b(?:below|under|underneath|beneath)\s+(.+)$"),
    ("next_to", r"\b(?:next to|beside)\s+(.+)$"),
]

POSITION_PATTERNS = [
    ("1st", r"\b(?:1st|first)\b"),
    ("2nd", r"\b(?:2nd|second)\b"),
    ("3rd", r"\b(?:3rd|third)\b"),
    ("last", r"\b(?:last|final)\b"),
]

KOREAN_POSITION_PATTERNS = [
    ("1st", r"\b(?:1번째|첫 번째|첫번째)\b"),
    ("2nd", r"\b(?:2번째|두 번째|두번째)\b"),
    ("3rd", r"\b(?:3번째|세 번째|세번째)\b"),
    ("last", r"\b(?:마지막)\b"),
]

ACTION_ALIASES = {
    "press": "click",
    "tap": "click",
    "touch": "click",
    "hit": "click",
    "select": "click",
    "open": "open",
    "launch": "open",
    "scroll": "scroll",
    "copy": "copy",
    "duplicate": "copy",
    "paste": "paste",
    "insert": "paste",
    "delete": "delete",
    "remove": "delete",
    "long-press": "long-press",
    "hold": "long-press",
}


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def normalize_lookup_text(text: str) -> str:
    return " ".join(text.strip().split())


def is_korean(text: str) -> bool:
    return bool(re.search(r"[\uac00-\ud7a3]", text))


def load_model():
    model_path = MODELS_DIR / "quantized_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")
    torch.serialization.add_safe_globals([CompactIntentModel])
    model = torch.load(model_path, weights_only=False, map_location="cpu")
    model.eval()
    return model


def predict_labels(model: CompactIntentModel, text: str, encoders: dict) -> tuple[str, str, str]:
    with torch.no_grad():
        outputs = model(
            input_ids=torch.zeros((1, 1), dtype=torch.long),
            attention_mask=torch.ones((1, 1), dtype=torch.long),
            text=[text],
        )

    intent_idx = int(outputs["intent_logits"].argmax(dim=1).item())
    target_idx = int(outputs["target_logits"].argmax(dim=1).item())
    spatial_idx = int(outputs["spatial_logits"].argmax(dim=1).item())
    intent = str(encoders["intent"].inverse_transform([intent_idx])[0])
    target_type = str(encoders["target_type"].inverse_transform([target_idx])[0])
    spatial_relation = str(encoders["spatial_relation"].inverse_transform([spatial_idx])[0])
    return intent, target_type, spatial_relation


def extract_attribute(text: str) -> str | None:
    if is_korean(text):
        for token, label in KOREAN_COLOR_WORDS.items():
            if token in text:
                return label
        return None
    tokens = normalize_text(text).replace("-", " ").split()
    for token in tokens:
        if token in COLOR_WORDS:
            return "gray" if token == "grey" else token
    return None


def extract_target_type(text: str) -> str | None:
    if is_korean(text):
        for token, label in sorted(KOREAN_TARGET_TYPES.items(), key=lambda item: len(item[0]), reverse=True):
            if token in text:
                return label
        return None
    normalized = normalize_text(text)
    for target_type in sorted(TARGET_TYPES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(target_type)}\b", normalized):
            return "checkbox" if target_type == "check box" else target_type
    return None


def extract_position(text: str) -> str | None:
    if is_korean(text):
        for label, pattern in KOREAN_POSITION_PATTERNS:
            if re.search(pattern, text):
                return label
        return None
    normalized = normalize_text(text)
    for label, pattern in POSITION_PATTERNS:
        if re.search(pattern, normalized):
            return label
    return None


def extract_spatial_reference(text: str) -> tuple[str | None, str | None]:
    if is_korean(text):
        patterns = [
            ("left_of", r"(.+?)\s+(?:왼쪽에|좌측에|왼편에)"),
            ("right_of", r"(.+?)\s+(?:오른쪽에|우측에|오른편에)"),
            ("above", r"(.+?)\s+(?:위에|상단에)"),
            ("below", r"(.+?)\s+(?:아래에|하단에)"),
            ("next_to", r"(.+?)\s+(?:옆에)"),
        ]
        for relation, pattern in patterns:
            match = re.search(pattern, text)
            if match:
                reference = match.group(1).strip()
                return relation, reference or None
        return None, None
    normalized = normalize_text(text)
    for relation, pattern in SPATIAL_PATTERNS:
        match = re.search(pattern, normalized)
        if match:
            reference = match.group(1).strip(" .,!?:;")
            return relation, reference or None
    if re.search(r"\bfrom the left\b", normalized):
        return "left_of", None
    if re.search(r"\bfrom the right\b", normalized):
        return "right_of", None
    if re.search(r"\bfrom the top\b", normalized):
        return "above", None
    if re.search(r"\bfrom the bottom\b", normalized):
        return "below", None
    return None, None


def fallback_intent(text: str) -> str:
    if is_korean(text):
        korean_aliases = [
            ("길게 누르기", "long-press"),
            ("길게", "long-press"),
            ("열기", "open"),
            ("실행", "open"),
            ("스크롤", "scroll"),
            ("복사", "copy"),
            ("붙여넣기", "paste"),
            ("삭제", "delete"),
            ("탭", "click"),
            ("클릭", "click"),
            ("누르기", "click"),
        ]
        for token, label in korean_aliases:
            if token in text:
                return label
        return "click"
    normalized = normalize_text(text)
    for token, label in ACTION_ALIASES.items():
        if re.search(rf"\b{re.escape(token)}\b", normalized):
            return label
    return "click"


def build_response(text: str, model: CompactIntentModel, encoders: dict) -> dict:
    korean = is_korean(text)
    normalized = normalize_text(text)
    lookup_text = normalize_lookup_text(text)
    model_input = f"[KO] {lookup_text}" if korean else f"[EN] {normalized}"
    intent, target_type, spatial_relation = predict_labels(model, model_input, encoders)

    parse_text = text if korean else normalized
    parsed_relation, reference = extract_spatial_reference(parse_text)
    attribute = extract_attribute(parse_text)
    explicit_target_type = extract_target_type(parse_text)
    position = extract_position(parse_text)

    # When the compact model does not have an exact text match, fall back to
    # lightweight parsing so the CLI still returns a usable structured payload.
    if model_input not in model.label_lookup:
        intent = fallback_intent(normalized)
        if explicit_target_type is not None:
            target_type = explicit_target_type
        if parsed_relation is not None:
            spatial_relation = parsed_relation

    response = {
        "intent": intent,
        "target": {
            "type": explicit_target_type or target_type,
            "attribute": attribute,
            "position": position,
            "spatial": {
                "relation": None if spatial_relation == "none" else spatial_relation,
                "reference": reference,
            },
        },
    }
    return response


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict structured UI intent JSON from text.")
    parser.add_argument("text", nargs="*", help="Command text to parse.")
    args = parser.parse_args()

    if args.text:
        text = " ".join(args.text)
    else:
        text = input("Enter command: ").strip()

    if not text:
        raise SystemExit("No input provided.")

    encoders = load_pickle(MODELS_DIR / "label_encoders.pkl")
    model = load_model()
    result = build_response(text, model, encoders)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
