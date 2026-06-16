"""Unified model download paths and asset resolution for recognition channels."""
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, List, Optional

from videotrans.configure.config import ROOT_DIR, defaulelang
from videotrans.configure.contants import FASTER_MODELS_DICT
from videotrans.util import tools


class ExecutionMode(str, Enum):
    SUBPROCESS = "subprocess"
    INLINE = "inline"
    EXTERNAL_BINARY = "external_binary"
    API = "api"
    NONE = "none"


@dataclass(frozen=True)
class ModelAsset:
    backend: str  # hf | ms
    model_id: str
    repo_id: Optional[str] = None
    local_dir: Optional[str] = None


def local_dir_for(recogn_type: int, model_name: str) -> Path:
    from videotrans import recognition

    model_name = model_name or ""
    if recogn_type in (recognition.FASTER_WHISPER, recognition.OPENAI_WHISPER):
        repo_id = (
            FASTER_MODELS_DICT.get(model_name, model_name)
            if recogn_type == recognition.FASTER_WHISPER
            else model_name
        )
        return Path(f"{ROOT_DIR}/models/models--{repo_id.replace('/', '--')}")
    if recogn_type == recognition.HUGGINGFACE_ASR:
        return Path(f"{ROOT_DIR}/models/models--{model_name.replace('/', '--')}")
    if recogn_type == recognition.QWENASR:
        return Path(f"{ROOT_DIR}/models/models--Qwen--Qwen3-ASR-{model_name}")
    if recogn_type == recognition.MIMO_ASR:
        from videotrans.recognition._constants import DEFAULT_MIMO_ASR_MODEL

        repo = model_name or DEFAULT_MIMO_ASR_MODEL
        return Path(f"{ROOT_DIR}/models/models--{repo.replace('/', '--')}")
    if recogn_type == recognition.NEMOTRON_ASR:
        from videotrans.recognition._constants import NEMOTRON_ASR_MODEL

        repo = model_name or NEMOTRON_ASR_MODEL
        return Path(f"{ROOT_DIR}/models/models--{repo.replace('/', '--')}")
    return Path(f"{ROOT_DIR}/models")


def mimo_tokenizer_dir() -> Path:
    from videotrans.recognition._constants import MIMO_AUDIO_TOKENIZER_REPO

    return Path(
        f"{ROOT_DIR}/models/models--"
        f"{MIMO_AUDIO_TOKENIZER_REPO.replace('/', '--')}"
    )


def resolve_funasr_model_name(model_name: str, detect_language: str) -> str:
    """Same alias rules as FunasrRecogn._exec (before download)."""
    model_name = model_name or "paraformer-zh"
    lang = (detect_language or "auto")[:2].lower()
    if model_name == "paraformer-zh" and lang not in ("zh", "en"):
        return (
            "FunAudioLLM/Fun-ASR-MLT-Nano-2512"
            if lang not in ("zh", "en", "ja", "yu")
            else "FunAudioLLM/Fun-ASR-Nano-2512"
        )
    if model_name == "SenseVoiceSmall":
        return "iic/SenseVoiceSmall"
    if model_name == "Fun-ASR-Nano-2512":
        if lang not in ("zh", "en", "ja", "yu"):
            return "FunAudioLLM/Fun-ASR-MLT-Nano-2512"
        return "FunAudioLLM/Fun-ASR-Nano-2512"
    if model_name != "paraformer-zh":
        return "FunAudioLLM/Fun-ASR-MLT-Nano-2512"
    return model_name


def resolve_assets(
    recogn_type: int,
    model_name: str,
    *,
    detect_language: Optional[str] = None,
) -> List[ModelAsset]:
    from videotrans import recognition

    model_name = model_name or ""
    detect_language = detect_language or "auto"
    assets: List[ModelAsset] = []

    if recogn_type in (recognition.FASTER_WHISPER, recognition.OPENAI_WHISPER):
        repo_id = (
            FASTER_MODELS_DICT.get(model_name, model_name)
            if recogn_type == recognition.FASTER_WHISPER
            else model_name
        )
        d = str(local_dir_for(recogn_type, model_name))
        assets.append(ModelAsset("hf", model_name, repo_id, d))
    elif recogn_type == recognition.HUGGINGFACE_ASR:
        d = str(local_dir_for(recogn_type, model_name))
        assets.append(ModelAsset("hf", model_name, model_name, d))
    elif recogn_type == recognition.QWENASR:
        d = str(local_dir_for(recogn_type, model_name))
        repo = f"Qwen/Qwen3-ASR-{model_name}"
        if defaulelang == "zh":
            assets.append(ModelAsset("ms", repo, repo, d))
        else:
            assets.append(
                ModelAsset("hf", f"Qwen3-ASR-{model_name}", repo, d)
            )
    elif recogn_type == recognition.MIMO_ASR:
        from videotrans.recognition._constants import (
            DEFAULT_MIMO_ASR_MODEL,
            MIMO_AUDIO_TOKENIZER_REPO,
        )

        asr_repo = model_name or DEFAULT_MIMO_ASR_MODEL
        d_asr = str(local_dir_for(recognition.MIMO_ASR, asr_repo))
        assets.append(ModelAsset("hf", asr_repo, asr_repo, d_asr))
        d_tok = str(mimo_tokenizer_dir())
        assets.append(
            ModelAsset("hf", MIMO_AUDIO_TOKENIZER_REPO, MIMO_AUDIO_TOKENIZER_REPO, d_tok)
        )
    elif recogn_type == recognition.NEMOTRON_ASR:
        from videotrans.recognition._constants import NEMOTRON_ASR_MODEL

        asr_repo = model_name or NEMOTRON_ASR_MODEL
        d_asr = str(local_dir_for(recognition.NEMOTRON_ASR, asr_repo))
        assets.append(ModelAsset("hf", asr_repo, asr_repo, d_asr))
    elif recogn_type == recognition.FUNASR_CN:
        resolved = resolve_funasr_model_name(model_name, detect_language)
        assets.append(
            ModelAsset("ms", "damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch")
        )
        if resolved == "paraformer-zh":
            for mid in (
                "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                "damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                "damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
                "damo/speech_campplus_sv_zh-cn_16k-common",
            ):
                assets.append(ModelAsset("ms", mid))
        else:
            assets.append(ModelAsset("ms", resolved))
    return assets


def ensure_assets(
    recogn_type: int,
    model_name: str,
    *,
    detect_language: Optional[str] = None,
    callback: Optional[Callable] = None,
) -> None:
    from videotrans import recognition

    if recogn_type == recognition.QWENASR:
        recognition.check_qwen_asr_installed()
    if recogn_type == recognition.MIMO_ASR:
        recognition.check_mimo_asr_installed()
    if recogn_type == recognition.NEMOTRON_ASR:
        recognition.check_nemotron_asr_installed()

    for asset in resolve_assets(
        recogn_type, model_name, detect_language=detect_language
    ):
        if asset.backend == "ms":
            tools.check_and_down_ms(
                model_id=asset.model_id,
                callback=callback,
                local_dir=asset.local_dir,
            )
        elif asset.backend == "hf":
            local_dir = asset.local_dir or str(
                local_dir_for(recogn_type, model_name)
            )
            repo_id = asset.repo_id or asset.model_id
            tools.check_and_down_hf(
                asset.model_id,
                repo_id,
                local_dir,
                callback=callback,
            )


def execution_mode(recogn_type: int, *, live: bool = False) -> ExecutionMode:
    from videotrans import recognition

    if live:
        if recogn_type in (
            recognition.QWENASR,
            recognition.FASTER_WHISPER,
            recognition.OPENAI_WHISPER,
        ):
            return ExecutionMode.INLINE
        if recogn_type in (
            recognition.OPENAI_API,
            recognition.GEMINI_SPEECH,
            recognition.QWEN3ASR,
            recognition.Deepgram,
            recognition.WHISPERX_API,
            recognition.OPENROUTER_ASR,
        ):
            return ExecutionMode.API
        if recogn_type == recognition.FUNASR_CN:
            return ExecutionMode.SUBPROCESS
    if recogn_type in (recognition.Whisper_CPP, recognition.Faster_Whisper_XXL):
        return ExecutionMode.EXTERNAL_BINARY
    if recogn_type in (
        recognition.OPENAI_API,
        recognition.GEMINI_SPEECH,
        recognition.QWEN3ASR,
        recognition.Deepgram,
        recognition.OPENROUTER_ASR,
    ):
        return ExecutionMode.API
    return ExecutionMode.SUBPROCESS
