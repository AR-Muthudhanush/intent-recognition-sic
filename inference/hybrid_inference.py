"""
Advanced Inference Engine with multiple model backends.
Automatically selects best available inference method.
"""

import os
import json
import re
from typing import Dict, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from inference.inference_correct import ProperIntentInference

# Try to import advanced backend
try:
    import google.generativeai as genai
    ADVANCED_BACKEND_AVAILABLE = True
except ImportError:
    ADVANCED_BACKEND_AVAILABLE = False

class HybridIntentInference:
    """
    Advanced inference engine with multiple backends.
    Automatically selects best available method.
    """

    def __init__(self, backend_key: Optional[str] = None, use_advanced: bool = True):
        """
        Initialize inference engine.

        Args:
            backend_key: Advanced backend configuration key
            use_advanced: Whether to enable advanced backend
        """
        self.use_advanced = use_advanced and ADVANCED_BACKEND_AVAILABLE
        self.backend_key = backend_key or os.environ.get('ADVANCED_BACKEND_KEY')

        # Initialize advanced backend if available and key provided
        if self.use_advanced and self.backend_key:
            try:
                genai.configure(api_key=self.backend_key)
                self.advanced_model = genai.GenerativeModel('gemini-pro')
                self.advanced_available = True
            except Exception as e:
                self.advanced_available = False
                self.use_advanced = False
        else:
            self.advanced_available = False
            self.use_advanced = False

        # Always initialize core model
        self.core_model = ProperIntentInference()

    def predict(self, command: str) -> Dict:
        """
        Predict intent using best available method.
        """
        # Try advanced backend first if available
        if self.advanced_available:
            try:
                return self._predict_advanced(command)
            except Exception as e:
                return self._predict_core(command)
        else:
            # Use core model directly
            return self._predict_core(command)

    def _predict_advanced(self, command: str) -> Dict:
        """Use advanced backend for prediction."""
        prompt = f"""Analyze this UI command and extract ALL fields.

Command: "{command}"

Respond with ONLY valid JSON in this exact format (no extra text):
{{
  "intent": "click|tap|scroll|delete|open|close|search|submit",
  "confidence": 0.95,
  "target": {{
    "type": "button|input|menu|etc",
    "attribute": "red|large|small|disabled|etc or null",
    "label": "submit|cancel|ok|save|etc or null",
    "index": "first|second|third|last|1|2|3|etc or null",
    "relation": "top|bottom|above|below|left|right|etc or null",
    "reference": "cart icon|search bar|menu button|etc or null"
  }}
}}

Rules:
- intent: What action? (click, tap, scroll, delete, open, close, play, pause, stop, search, zoom, rotate, move, resize, upload, download, share, save, refresh, retry, enable, disable, check, uncheck, expand, collapse, accept, reject, approve, back, next, previous, home, settings, login, logout, bookmark, pin, unpin, start, finish, exit, continue, add, remove, hide, show, mute, unmute, install, uninstall, submit, cancel, type, enter, focus, hover, launch)
- confidence: 0.0-1.0
- target.type: UI element (button, input, text, icon, menu, checkbox, image, dropdown, etc)
- target.attribute: Colors (red, blue, green), sizes (large, small), states (disabled, active, highlighted)
- target.label: Text on element (submit, cancel, ok, save, delete, edit, add, search, filter)
- target.index: Position (first, second, third, last, 1st, 2nd, nth item)
- target.relation: Spatial location (top, bottom, left, right, above, below, beside, center, corner)
- target.reference: Reference element (cart icon, search bar, menu button, submit button)
- Return null ONLY if field is truly not mentioned"""

        response = self.gemini_model.generate_content(prompt)

        # Extract JSON from response
        text = response.text.strip()

        # Try to parse JSON directly
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON from the text
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                raise ValueError(f"Invalid JSON response from Gemini: {text}")

        return result

    def _predict_core(self, command: str) -> Dict:
        """Use core model for prediction with enhanced field extraction."""
        result = self.core_model.predict(command)
        command_lower = command.lower()

        # Extract additional fields from command
        attribute = self._extract_attribute(command_lower)
        label = self._extract_label(command_lower)
        index = self._extract_index(command_lower)
        spatial_relation = self._extract_spatial_relation(command_lower)
        reference = self._extract_reference(command_lower)

        # Use extracted spatial relation if available, otherwise use model prediction
        final_relation = spatial_relation or (result['spatial_relation'] if result['spatial_relation'] != 'none' else None)

        # Convert to standardized format
        return {
            "intent": result['intent'],
            "confidence": result['intent_confidence'],
            "target": {
                "type": result['target'] if result['target'] != 'none' else None,
                "attribute": attribute,
                "label": label,
                "index": index,
                "relation": final_relation,
                "reference": reference
            }
        }

    def _extract_spatial_relation(self, command: str) -> Optional[str]:
        """Extract comprehensive spatial relations including combined ones."""
        command_lower = command.lower()

        # Check for combined spatial relations first (most specific)
        combined_relations = [
            ('top-left', 'top-left'), ('topleft', 'top-left'),
            ('top-right', 'top-right'), ('topright', 'top-right'),
            ('bottom-left', 'bottom-left'), ('bottomleft', 'bottom-left'),
            ('bottom-right', 'bottom-right'), ('bottomright', 'bottom-right'),
            ('center-left', 'center-left'), ('centerleft', 'center-left'),
            ('center-right', 'center-right'), ('centerright', 'center-right'),
            ('upper-center', 'upper-center'), ('upper center', 'upper-center'),
            ('lower-center', 'lower-center'), ('lower center', 'lower-center'),
            ('in front of', 'in_front_of'),
            ('next to', 'next_to'), ('beside', 'beside'),
            ('adjacent to', 'adjacent_to'),
            ('below', 'below'),
            ('above', 'above'),
            ('inside', 'inside'),
            ('outside', 'outside'),
            ('between', 'between'),
            ('behind', 'behind'),
            ('under', 'under'),
            ('over', 'over'),
            ('beneath', 'beneath'),
        ]

        # Check longer phrases first
        for phrase, relation in combined_relations:
            if phrase in command_lower:
                return relation

        # Check single word spatial relations
        words = command_lower.split()
        single_relations = {
            'top': 'top',
            'bottom': 'bottom',
            'left': 'left',
            'right': 'right',
            'center': 'center',
            'middle': 'center',
            'nearest': 'nearest',
            'farthest': 'farthest',
            'first': 'first',
            'last': 'last',
            'edge': 'edge',
            'corner': 'corner',
        }

        for word in words:
            word_clean = word.lower().strip('.,!?;:')
            if word_clean in single_relations:
                return single_relations[word_clean]

        return None

    def _extract_attribute(self, command: str) -> Optional[str]:
        """Extract visual attributes (color, size, state)."""
        words = command.split()
        attributes = [
            'red', 'blue', 'green', 'yellow', 'black', 'white', 'gray', 'purple', 'orange', 'pink',
            'large', 'small', 'big', 'medium',
            'disabled', 'active', 'highlighted', 'focused',
            'primary', 'secondary', 'danger', 'warning', 'success', 'info'
        ]
        for word in words:
            word_clean = word.lower().strip('.,!?;:')
            if word_clean in attributes:
                return word_clean
        return None

    def _extract_label(self, command: str) -> Optional[str]:
        """Extract element label (text on UI)."""
        words = command.split()
        labels = [
            'submit', 'cancel', 'ok', 'save', 'delete', 'edit', 'add', 'remove',
            'search', 'filter', 'sort', 'export', 'import', 'share', 'download', 'upload',
            'login', 'logout', 'register', 'signin', 'signup',
            'next', 'previous', 'back', 'forward', 'home', 'settings',
            'refresh', 'retry', 'reset', 'clear', 'close', 'open'
        ]
        for word in words:
            word_clean = word.lower().strip('.,!?;:')
            if word_clean in labels:
                return word_clean
        return None

    def _extract_index(self, command: str) -> Optional[str]:
        """Extract position/index (first, second, last, etc)."""
        words = command.split()
        positions = {
            'first': 'first', '1st': 'first', '1': 'first',
            'second': 'second', '2nd': 'second', '2': 'second',
            'third': 'third', '3rd': 'third', '3': 'third',
            'fourth': 'fourth', '4th': 'fourth', '4': 'fourth',
            'fifth': 'fifth', '5th': 'fifth', '5': 'fifth',
            'last': 'last', 'bottom': 'last',
            'top': 'first',
            'middle': 'middle',
        }
        for word in words:
            word_clean = word.lower().strip('.,!?;:')
            if word_clean in positions:
                return positions[word_clean]
        return None

    def _extract_reference(self, command: str) -> Optional[str]:
        """Extract reference element (nearby UI component)."""
        command_lower = command.lower()
        # Check longer phrases first, then shorter ones
        references = [
            # Multi-word references
            ('search bar', 'search bar'),
            ('search box', 'search box'),
            ('menu button', 'menu button'),
            ('submit button', 'submit button'),
            ('cancel button', 'cancel button'),
            ('back button', 'back button'),
            ('home button', 'home button'),
            ('close button', 'close button'),
            ('cart icon', 'cart icon'),
            ('user icon', 'user icon'),
            ('settings icon', 'settings icon'),
            ('delete icon', 'delete icon'),
            ('edit icon', 'edit icon'),
            ('add icon', 'add icon'),
            ('navigation bar', 'navigation bar'),
            ('tool bar', 'toolbar'),
            ('top bar', 'top bar'),
            ('bottom bar', 'bottom bar'),
            ('left panel', 'left panel'),
            ('right panel', 'right panel'),
            # Single word references
            ('cart', 'cart icon'),
            ('menu', 'menu'),
            ('image', 'image'),
            ('photo', 'image'),
            ('text', 'text'),
            ('label', 'label'),
            ('header', 'header'),
            ('footer', 'footer'),
            ('navbar', 'navbar'),
            ('sidebar', 'sidebar'),
            ('button', 'button'),
            ('input', 'input'),
            ('field', 'field'),
            ('box', 'box'),
            ('icon', 'icon'),
            ('item', 'item'),
            ('card', 'card'),
            ('list', 'list'),
            ('table', 'table'),
            ('row', 'row'),
            ('column', 'column'),
        ]
        for ref_key, ref_value in references:
            if ref_key in command_lower:
                return ref_value
        return None

    def predict_batch(self, commands: list) -> list:
        """Predict for multiple commands."""
        return [self.predict(cmd) for cmd in commands]


if __name__ == "__main__":
    print("="*70)
    print("Hybrid Intent Recognition (Gemini + Fallback)")
    print("="*70)

    # Initialize
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        print("\nNote: Set GEMINI_API_KEY environment variable to use Gemini API")
        print("Without it, will use trained model only\n")

    engine = HybridIntentInference(api_key=api_key)

    # Test commands
    test_commands = [
        "click the submit button",
        "scroll down to the bottom",
        "delete the red error message",
        "search for users in the dashboard",
        "open the settings menu at the top",
    ]

    print("\nTesting predictions:\n")
    for cmd in test_commands:
        result = engine.predict(cmd)
        print(f"Command: {cmd}")
        print(f"Intent: {result['intent']} ({result['confidence']:.1%})")
        print(f"Target: {result['target']['type']}")
        print(f"Spatial: {result['target']['relation']}")
        print()
