"""Shared Qwen3-ASR model load and transcribe helpers."""
from pathlib import Path
from typing import List, Union

from videotrans.configure.config import logger
from videotrans.recognition.model_assets import local_dir_for
from videotrans.recognition import QWENASR, check_qwen_asr_installed
from videotrans.recognition.qwen_asr_lang import qwen_asr_language_name


def load_qwen_asr(
    model_name: str,
    *,
    is_cuda: bool = False,
    device_index: int = 0,
):
    check_qwen_asr_installed()
    import torch
    from qwen_asr import Qwen3ASRModel

    if is_cuda:
        device_map = f"cuda:{device_index}"
        dtype = torch.float16
    else:
        device_map = "cpu"
        dtype = torch.float32
    model_dir = str(local_dir_for(QWENASR, model_name))
    logger.debug(f"[Qwen ASR] loading from {model_dir} on {device_map}")
    return Qwen3ASRModel.from_pretrained(
        model_dir,
        dtype=dtype,
        device_map=device_map,
        attn_implementation=None,
        max_inference_batch_size=8,
        max_new_tokens=2048,
    )


def transcribe_files(
    model,
    paths: List[str],
    *,
    language: str,
    return_time_stamps: bool = False,
) -> List[str]:
    if not paths:
        return []
    results = model.transcribe(
        audio=paths,
        language=[language] * len(paths),
        return_time_stamps=return_time_stamps,
    )
    return [(r.text or "").strip() if r else "" for r in results]
