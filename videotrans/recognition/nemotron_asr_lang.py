"""Map app language codes to Nemotron target_lang prompt keys."""
from typing import Optional

_LOCALE_MAP = {
    "en": "en-US",
    "en-us": "en-US",
    "en-gb": "en-GB",
    "zh": "zh-CN",
    "zh-cn": "zh-CN",
    "zh-tw": "zh-CN",
    "es": "es-ES",
    "de": "de-DE",
    "fr": "fr-FR",
    "it": "it-IT",
    "pt": "pt-BR",
    "ja": "ja-JP",
    "ko": "ko-KR",
    "ru": "ru-RU",
    "ar": "ar-AR",
    "hi": "hi-IN",
    "vi": "vi-VN",
    "uk": "uk-UA",
    "nl": "nl-NL",
    "tr": "tr-TR",
}


def nemotron_target_lang(detect_language: Optional[str]) -> str:
    if not detect_language:
        return "auto"
    raw = str(detect_language).strip().lower().replace("_", "-")
    if not raw or raw in ("auto", "-"):
        return "auto"
    if raw in _LOCALE_MAP:
        return _LOCALE_MAP[raw]
    base = raw.split("-")[0]
    return _LOCALE_MAP.get(base, "auto")
