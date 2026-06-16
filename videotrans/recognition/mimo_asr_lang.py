"""Map app recognition language codes to MiMo-V2.5-ASR audio_tag values."""
from typing import Optional


def mimo_asr_audio_tag(detect_language: Optional[str]) -> str:
    """
    Return audio_tag for MimoAudio.asr_sft(), or '' for automatic detection.
    Supports zh / en / auto per Xiaomi MiMo-V2.5-ASR docs.
    """
    if not detect_language:
        return ""
    raw = str(detect_language).strip().lower().replace("_", "-")
    if not raw or raw in ("auto", "-"):
        return ""
    if raw.startswith("zh") or raw in ("yue", "cantonese", "zho"):
        return "<chinese>"
    if raw.startswith("en"):
        return "<english>"
    return ""
