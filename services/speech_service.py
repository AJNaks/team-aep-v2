"""
Speech Service Abstraction Layer
---------------------------------
PoC: Uses browser Web Speech API (client-side JS) + Claude for translation
Production: Swap to BhashiniSpeechService when API access is available
"""

import os
import anthropic
from typing import Optional


class SpeechService:
    """Base speech/language service. Translation via Claude for PoC."""

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        if source_lang == target_lang:
            return text
        lang_map = {
            "hi": "Hindi", "en": "English", "ta": "Tamil", "te": "Telugu",
            "kn": "Kannada", "ml": "Malayalam", "bn": "Bengali", "mr": "Marathi",
            "gu": "Gujarati", "pa": "Punjabi", "or": "Odia", "as": "Assamese"
        }
        src_name = lang_map.get(source_lang, source_lang)
        tgt_name = lang_map.get(target_lang, target_lang)
        response = self.client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": f"Translate the following {src_name} text to {tgt_name}. "
                           f"Return ONLY the translation, nothing else.\n\n{text}"
            }]
        )
        return response.content[0].text.strip()

    def detect_language(self, text: str) -> str:
        response = self.client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{
                "role": "user",
                "content": f"Reply with ONLY the ISO 639-1 language code for this text "
                           f"(e.g. hi, en, ta, kn):\n\n{text}"
            }]
        )
        return response.content[0].text.strip().lower()


class BhashiniSpeechService(SpeechService):
    """Production Bhashini implementation. Swap in when ULCA access is available."""

    def __init__(self):
        super().__init__()
        self.user_id = os.getenv("BHASHINI_USER_ID")
        self.ulca_api_key = os.getenv("BHASHINI_ULCA_API_KEY")
        self.inference_key = os.getenv("BHASHINI_INFERENCE_KEY")

    # TODO: Implement transcribe(), translate(), synthesize() with Bhashini pipeline calls
