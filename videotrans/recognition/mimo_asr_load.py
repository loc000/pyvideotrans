"""Shared MiMo-V2.5-ASR model load and transcribe helpers."""
import sys
from pathlib import Path
from typing import List, Optional

from videotrans.configure.config import ROOT_DIR, logger, params
from videotrans.recognition._constants import DEFAULT_MIMO_ASR_MODEL, MIMO_ASR_CHANNEL_ID
from videotrans.recognition.model_assets import local_dir_for, mimo_tokenizer_dir
from videotrans.recognition.mimo_asr_lang import mimo_asr_audio_tag


def get_mimo_inference_path() -> Path:
    """Local clone of XiaomiMiMo/MiMo-V2.5-ASR inference repo (contains src.mimo_audio)."""
    custom = (params.get("mimo_asr_repo_path") or "").strip()
    if custom:
        return Path(custom)
    return Path(f"{ROOT_DIR}/models/MiMo-V2.5-ASR-inference")


def ensure_mimo_inference_on_path() -> Path:
    repo = get_mimo_inference_path()
    if not repo.is_dir():
        from videotrans.configure.config import tr
        from videotrans.configure.excepts import SpeechToTextError

        raise SpeechToTextError(
            tr(
                "MiMo-ASR inference code not found at {path}. "
                "Clone: git clone https://github.com/XiaomiMiMo/MiMo-V2.5-ASR.git {path} "
                "then install its requirements.txt (Linux/CUDA recommended)."
            ).format(path=repo)
        )
    repo_str = str(repo.resolve())
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    return repo


def check_mimo_import() -> None:
    ensure_mimo_inference_on_path()
    try:
        from src.mimo_audio.mimo_audio import MimoAudio  # noqa: F401
    except ImportError as e:
        from videotrans.configure.config import tr
        from videotrans.configure.excepts import SpeechToTextError

        raise SpeechToTextError(
            tr(
                "Cannot import MimoAudio from MiMo inference repo: {err}. "
                "Ensure {path} is a full clone of XiaomiMiMo/MiMo-V2.5-ASR with dependencies installed."
            ).format(err=e, path=get_mimo_inference_path())
        ) from e


def load_mimo_asr(
    model_name: str,
    *,
    is_cuda: bool = False,
):
    check_mimo_import()
    from src.mimo_audio.mimo_audio import MimoAudio

    model_name = model_name or DEFAULT_MIMO_ASR_MODEL
    model_path = str(local_dir_for(MIMO_ASR_CHANNEL_ID, model_name))
    tokenizer_path = str(mimo_tokenizer_dir())
    logger.debug(
        f"[MiMo ASR] loading model={model_path} tokenizer={tokenizer_path} cuda={is_cuda}"
    )
    return MimoAudio(model_path, tokenizer_path)


def transcribe_files(
    model,
    paths: List[str],
    *,
    detect_language: Optional[str] = None,
) -> List[str]:
    if not paths:
        return []
    audio_tag = mimo_asr_audio_tag(detect_language)
    out: List[str] = []
    for path in paths:
        if audio_tag:
            text = model.asr_sft(path, audio_tag=audio_tag)
        else:
            text = model.asr_sft(path)
        out.append((text or "").strip() if text else "")
    return out
