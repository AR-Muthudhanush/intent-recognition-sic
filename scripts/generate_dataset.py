#!/usr/bin/env python3
"""
Generate comprehensive static dataset for Intent Recognition System.
Covers all intents, targets, spatial relations with variations.
Output: dataset/dataset.csv (40k-60k samples)
"""

import csv
import random
from itertools import product
from typing import List, Tuple

random.seed(42)

INTENTS = [
    "click", "tap", "double_tap", "long_press", "scroll", "swipe",
    "drag", "drop", "delete", "copy", "paste", "highlight", "select",
    "open", "close", "play", "pause", "stop", "search", "zoom",
    "rotate", "move", "resize", "upload", "download", "share", "save",
    "refresh", "retry", "enable", "disable", "check", "uncheck",
    "expand", "collapse", "accept", "reject", "approve", "back",
    "next", "previous", "home", "settings", "login", "logout",
    "bookmark", "pin", "unpin", "start", "finish", "exit", "continue",
    "add", "remove", "hide", "show", "mute", "unmute", "install",
    "uninstall", "submit", "cancel", "type", "enter", "focus",
    "hover", "launch"
]

TARGETS = [
    "button", "icon", "image", "checkbox", "switch", "radio_button",
    "dropdown", "textbox", "input", "text", "chart", "graph", "table",
    "card", "menu", "navigation_bar", "toolbar", "search_box", "video",
    "audio", "attachment", "file", "document", "message", "notification",
    "popup", "dialog", "tab", "list_item", "profile", "avatar",
    "calendar", "slider", "progress_bar", "link", "qr_code", "camera",
    "gallery", "email", "password", "phone_number"
]

SPATIAL_RELATIONS = [
    "top", "bottom", "left", "right", "top_left", "top_right",
    "bottom_left", "bottom_right", "center", "center_left", "center_right",
    "upper_center", "lower_center", "above", "below", "beside",
    "adjacent_to", "next_to", "inside", "outside", "between",
    "before", "after", "behind", "in_front_of", "beneath", "over",
    "under", "nearest", "farthest", "first", "second", "third",
    "last", "second_last", "first_row", "second_row", "first_column",
    "last_column", "top_edge", "bottom_edge", "left_edge", "right_edge"
]

ATTRIBUTES = [
    "red", "blue", "green", "yellow", "large", "small", "disabled",
    "active", "highlighted", "focused", "expanded", "collapsed",
    "primary", "secondary", "danger", "warning", "success", "info",
    "dark", "light", "bold", "italic", "underlined", "strikethrough",
    ""
]

LABELS = [
    "submit", "cancel", "ok", "save", "delete", "edit", "add",
    "remove", "search", "filter", "sort", "export", "import",
    "share", "download", "upload", "refresh", "retry", "back",
    "next", "previous", "home", "settings", "profile", "logout",
    "login", "register", "password", "email", "phone", "address",
    "message", "notification", "alert", "confirm", "warning",
    "error", "success", "info", "loading", "empty", ""
]

REFERENCES = [
    "cart icon", "menu button", "search bar", "profile icon",
    "settings button", "back button", "submit button", "cancel button",
    "save button", "delete button", "edit button", "add button",
    "remove button", "refresh button", "home button", "filter button",
    "sort button", "export button", "import button", "share button",
    "download button", "upload button", "email field", "password field",
    "search field", "address field", "phone field", "message field",
    "notification panel", "sidebar menu", "top navigation", "footer",
    "header", "main content", "dialog box", "popup", "modal",
    "card list", "image gallery", "video player", "audio player",
    "map view", "calendar view", "table view", "chart view", ""
]

TYPO_VARIATIONS = {
    "click": ["clik", "clck", "clkck", "c1ick"],
    "tap": ["tapp", "tap", "tpa", "t4p"],
    "scroll": ["scrol", "scrlol", "scrolll", "scro11"],
    "delete": ["delte", "delet", "deleet", "de1ete"],
    "submit": ["subm1t", "summit", "submitt", "submi7"],
    "button": ["buton", "buttn", "butt0n", "buttton"],
    "input": ["inpu7", "imput", "inpu", "inp0t"],
}

OCR_VARIATIONS = {
    "click": ["c1ick", "c|ick", "click"],
    "search": ["5earch", "searcn", "se@rch"],
    "close": ["c1o5e", "clo5e", "c|ose"],
}

STT_VARIATIONS = {
    "click": ["quick", "klik", "clique"],
    "scroll": ["skull", "scrolled"],
    "delete": ["de leaf", "uh leaf"],
}

def introduce_typo(word: str) -> str:
    """Introduce realistic typos."""
    if word.lower() in TYPO_VARIATIONS:
        return random.choice(TYPO_VARIATIONS[word.lower()])
    if random.random() < 0.3:
        chars = list(word)
        idx = random.randint(0, len(chars) - 1)
        chars[idx] = random.choice('abcdefghijklmnopqrstuvwxyz0123456789')
        return ''.join(chars)
    return word

def introduce_ocr_error(word: str) -> str:
    """Simulate OCR errors."""
    if word.lower() in OCR_VARIATIONS:
        return random.choice(OCR_VARIATIONS[word.lower()])
    if random.random() < 0.2:
        replacements = {'l': '1', 'o': '0', 's': '5', 'e': '3', 'a': '@'}
        chars = list(word)
        idx = random.randint(0, len(chars) - 1)
        if chars[idx].lower() in replacements:
            chars[idx] = replacements[chars[idx].lower()]
        return ''.join(chars)
    return word

def introduce_stt_error(word: str) -> str:
    """Simulate STT errors."""
    if word.lower() in STT_VARIATIONS:
        return random.choice(STT_VARIATIONS[word.lower()])
    return word

def generate_command_variations(intent: str, target: str, attribute: str,
                               label: str, spatial: str, reference: str) -> List[str]:
    """Generate natural language command variations."""
    variations = []

    base_commands = [
        f"{intent} {target}",
        f"{intent} the {target}",
        f"please {intent} {target}",
        f"can you {intent} {target}",
        f"{intent} on {target}",
    ]

    if reference:
        base_commands.extend([
            f"{intent} {target} {spatial} {reference}",
            f"{intent} the {target} that is {spatial} the {reference}",
            f"please {intent} the {target} {spatial} to the {reference}",
        ])

    if attribute:
        base_commands.extend([
            f"{intent} the {attribute} {target}",
            f"{intent} {attribute} {target}",
        ])

    if label:
        base_commands.extend([
            f"{intent} {label}",
            f"{intent} the {label}",
            f"please {intent} the {label} {target}",
        ])

    for cmd in base_commands:
        variations.append(cmd)

        if random.random() < 0.4:
            words = cmd.split()
            typo_cmd = ' '.join(introduce_typo(w) for w in words)
            if typo_cmd != cmd:
                variations.append(typo_cmd)

        if random.random() < 0.3:
            words = cmd.split()
            ocr_cmd = ' '.join(introduce_ocr_error(w) for w in words)
            if ocr_cmd != cmd:
                variations.append(ocr_cmd)

        if random.random() < 0.3:
            words = cmd.split()
            stt_cmd = ' '.join(introduce_stt_error(w) for w in words)
            if stt_cmd != cmd:
                variations.append(stt_cmd)

        if random.random() < 0.3:
            words = cmd.split()
            random.shuffle(words[1:])
            reordered = ' '.join(words)
            if reordered != cmd:
                variations.append(reordered)

        if random.random() < 0.2:
            filler_words = ["actually", "i think", "you know", "like", "maybe", "just"]
            filler = random.choice(filler_words)
            variations.append(f"{filler} {cmd}")

    return list(set(variations[:random.randint(3, 8)]))

def generate_bilingual_commands(intent: str, target: str, attribute: str,
                               label: str, spatial: str, reference: str) -> Tuple[str, str]:
    """Generate English and Korean commands."""
    en_variations = generate_command_variations(intent, target, attribute, label, spatial, reference)
    en_cmd = random.choice(en_variations) if en_variations else f"{intent} {target}"

    ko_commands = {
        "click": ["클릭", "클릭해", "탭하세요"],
        "scroll": ["스크롤", "스크롤하세요", "위로"],
        "delete": ["삭제", "삭제해", "지우세요"],
        "submit": ["제출", "전송", "확인하세요"],
        "search": ["검색", "찾기", "검색해"],
    }

    ko_base = ko_commands.get(intent, [intent])
    ko_cmd = random.choice(ko_base)
    if target:
        target_ko_map = {
            "button": "버튼", "icon": "아이콘", "input": "입력", "checkbox": "체크박스"
        }
        ko_target = target_ko_map.get(target, target)
        ko_cmd = f"{ko_cmd} {ko_target}"

    return en_cmd, ko_cmd

def generate_dataset(num_samples: int = 50000) -> List[dict]:
    """Generate complete dataset."""
    data = []
    sample_count = 0

    intents_sample = random.sample(INTENTS, min(35, len(INTENTS)))
    targets_sample = random.sample(TARGETS, min(30, len(TARGETS)))
    spatial_sample = random.sample(SPATIAL_RELATIONS, min(30, len(SPATIAL_RELATIONS)))

    for intent in intents_sample:
        for target in targets_sample:
            for spatial in spatial_sample:
                for _ in range(random.randint(4, 8)):
                    if sample_count >= num_samples:
                        break

                    attribute = random.choice(ATTRIBUTES) if random.random() < 0.6 else ""
                    label = random.choice(LABELS) if random.random() < 0.5 else ""
                    reference = random.choice(REFERENCES) if random.random() < 0.4 else ""

                    en_cmd, ko_cmd = generate_bilingual_commands(
                        intent, target, attribute, label, spatial, reference
                    )

                    command = random.choice([en_cmd, ko_cmd])
                    index = str(random.randint(0, 99)) if random.random() < 0.7 else ""

                    data.append({
                        "command": command,
                        "intent": intent,
                        "target": target,
                        "attribute": attribute or "",
                        "label": label or "",
                        "index": index,
                        "relation": spatial,
                        "reference": reference or ""
                    })

                    sample_count += 1

                if sample_count >= num_samples:
                    break
            if sample_count >= num_samples:
                break
        if sample_count >= num_samples:
            break

    random.shuffle(data)
    return data[:num_samples]

def save_dataset(data: List[dict], filepath: str = "dataset/dataset.csv"):
    """Save dataset to CSV."""
    fieldnames = ["command", "intent", "target", "attribute", "label", "index", "relation", "reference"]

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    print(f"Dataset saved: {filepath}")
    print(f"Total samples: {len(data)}")

if __name__ == "__main__":
    print("Generating dataset...")
    dataset = generate_dataset(50000)
    save_dataset(dataset)
    print("Done!")
