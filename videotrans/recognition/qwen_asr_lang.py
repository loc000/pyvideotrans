"""Map app recognition language codes to Qwen3-ASR language names."""
from typing import Optional

from videotrans.translator import get_language_qwen

# Aliases not covered by LANG_CODE[..][9] (qwen-asr name)
_QWEN_ASR_EXTRA = {
    "yue": "Cantonese",
    "cantonese": "Cantonese",
    "zho": "Chinese",
    "eng": "English",
    "jp": "Japanese",
    "kor": "Korean",
    "fil": "Filipino",
    "tl": "Filipino",
}


def qwen_asr_language_name(detect_language: Optional[str]) -> Optional[str]:
    """
    Return a Qwen3-ASR language name, or None for automatic detection.
    Uses translator.get_language_qwen() when possible; extra map for edge aliases.
    """
    if not detect_language:
        return None
    raw = str(detect_language).strip()
    if not raw or raw.lower() in ("auto", "-"):
        return None

    low = raw.lower().replace("_", "-")
    if low in _QWEN_ASR_EXTRA:
        return _QWEN_ASR_EXTRA[low]

    name = get_language_qwen(low)
    if name:
        return name

    try:
        from qwen_asr.inference.utils import SUPPORTED_LANGUAGES, normalize_language_name

        norm = normalize_language_name(raw)
        if norm in SUPPORTED_LANGUAGES:
            return norm
    except (ValueError, ImportError):
        pass

    return get_language_qwen(low.split("-")[0])
