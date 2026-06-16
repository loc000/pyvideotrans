"""NVIDIA Nemotron 3.5 streaming ASR load and live session helpers."""
from typing import List, Optional, Tuple

import numpy as np

from videotrans.configure.config import logger, settings
from videotrans.recognition._constants import (
    DEFAULT_NEMOTRON_ATT_CONTEXT,
    NEMOTRON_ASR_MODEL,
)
from videotrans.recognition.nemotron_asr_lang import nemotron_target_lang


def check_nemotron_import() -> None:
    try:
        import nemo.collections.asr  # noqa: F401
    except ImportError as e:
        from videotrans.configure.config import tr
        from videotrans.configure.excepts import SpeechToTextError

        raise SpeechToTextError(
            tr(
                "Nemotron ASR requires NVIDIA NeMo. Install with: "
                "pip install Cython packaging && "
                "pip install git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]"
            )
        ) from e


def _extract_transcriptions(hyps) -> List[str]:
    from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

    if not hyps:
        return []
    if isinstance(hyps[0], Hypothesis):
        return [(h.text or "").strip() for h in hyps]
    return [str(h).strip() for h in hyps]


def load_nemotron_asr(
    model_name: str = "",
    *,
    is_cuda: bool = False,
    detect_language: Optional[str] = None,
    att_context_size: Optional[List[int]] = None,
):
    check_nemotron_import()
    import torch
    import nemo.collections.asr as nemo_asr

    repo = model_name or NEMOTRON_ASR_MODEL
    device = torch.device("cuda:0" if is_cuda and torch.cuda.is_available() else "cpu")
    logger.debug(f"[Nemotron ASR] loading {repo} on {device}")
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=repo)
    ctx = att_context_size or settings.get(
        "nemotron_att_context_size", DEFAULT_NEMOTRON_ATT_CONTEXT
    )
    if hasattr(model.encoder, "set_default_att_context_size"):
        model.encoder.set_default_att_context_size(att_context_size=list(ctx))
    target_lang = nemotron_target_lang(detect_language)
    if hasattr(model, "set_inference_prompt"):
        model.set_inference_prompt(target_lang)
    if hasattr(model, "decoding") and hasattr(model.decoding, "set_strip_lang_tags"):
        model.decoding.set_strip_lang_tags(True)
    model = model.to(device=device, dtype=torch.float32)
    model.eval()
    return model


class NemotronStreamSession:
    """Cache-aware streaming session for live microphone input."""

    def __init__(
        self,
        model_name: str = "",
        *,
        is_cuda: bool = False,
        detect_language: Optional[str] = None,
        att_context_size: Optional[List[int]] = None,
    ):
        import torch
        from nemo.collections.asr.parts.utils.streaming_utils import (
            CacheAwareStreamingAudioBuffer,
        )

        self.model = load_nemotron_asr(
            model_name,
            is_cuda=is_cuda,
            detect_language=detect_language,
            att_context_size=att_context_size,
        )
        self.device = next(self.model.parameters()).device
        self.compute_dtype = torch.float32
        self.buffer = CacheAwareStreamingAudioBuffer(
            model=self.model,
            online_normalization=False,
            pad_and_drop_preencoded=False,
        )
        self._reset_stream_state()
        self.step_num = 0

    def _reset_stream_state(self) -> None:
        (
            self.cache_last_channel,
            self.cache_last_time,
            self.cache_last_channel_len,
        ) = self.model.encoder.get_initial_cache_state(batch_size=1)
        self.previous_hypotheses = None
        self.pred_out_stream = None
        self.step_num = 0
        self._last_text = ""

    def reset(self) -> None:
        self.buffer.reset_buffer()
        self._reset_stream_state()

    def _drop_extra_pre_encoded(self) -> int:
        if self.step_num == 0:
            return 0
        return int(self.model.encoder.streaming_cfg.drop_extra_pre_encoded)

    def feed_audio_16k(self, samples: np.ndarray) -> str:
        import torch

        audio = np.clip(samples.astype(np.float32), -1.0, 1.0)
        if audio.size < 1:
            return self._last_text
        self.buffer.append_audio(audio, stream_id=-1)
        latest = self._last_text
        while True:
            chunk = self.buffer.get_next_chunk()
            if chunk is None:
                break
            audio_chunk, chunk_lengths = chunk
            with torch.inference_mode():
                (
                    self.pred_out_stream,
                    transcribed_texts,
                    self.cache_last_channel,
                    self.cache_last_time,
                    self.cache_last_channel_len,
                    self.previous_hypotheses,
                ) = self.model.conformer_stream_step(
                    processed_signal=audio_chunk.to(self.compute_dtype),
                    processed_signal_length=chunk_lengths,
                    cache_last_channel=self.cache_last_channel,
                    cache_last_time=self.cache_last_time,
                    cache_last_channel_len=self.cache_last_channel_len,
                    keep_all_outputs=self.buffer.is_buffer_empty(),
                    previous_hypotheses=self.previous_hypotheses,
                    previous_pred_out=self.pred_out_stream,
                    drop_extra_pre_encoded=self._drop_extra_pre_encoded(),
                    return_transcription=True,
                )
            texts = _extract_transcriptions(transcribed_texts)
            if texts and texts[0]:
                latest = texts[0]
            self.step_num += 1
        self._last_text = latest
        return latest

    def release(self) -> None:
        self.model = None
        self.buffer = None
