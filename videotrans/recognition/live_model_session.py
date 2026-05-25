"""Warm in-process ASR sessions for live chunked recognition."""
from typing import Any, List, Optional, Union

import numpy as np

from videotrans.configure.config import logger
from videotrans.recognition.model_assets import local_dir_for
from videotrans.recognition import FASTER_WHISPER, OPENAI_WHISPER, QWENASR
from videotrans.recognition.qwen_asr_load import load_qwen_asr, transcribe_files
from videotrans.recognition.qwen_asr_lang import qwen_asr_language_name


class LiveModelSession:
    def __init__(
        self,
        recogn_type: int,
        model_name: str,
        *,
        detect_language: str = "auto",
        is_cuda: bool = False,
    ):
        self.recogn_type = int(recogn_type)
        self.model_name = model_name or ""
        self.detect_language = detect_language or "auto"
        self.is_cuda = is_cuda
        self.model: Any = None
        self._qwen_lang = qwen_asr_language_name(self.detect_language)

    def ensure_loaded(self) -> None:
        if self.model is not None:
            return
        if self.recogn_type == QWENASR:
            self.model = load_qwen_asr(self.model_name, is_cuda=self.is_cuda)
        elif self.recogn_type in (FASTER_WHISPER, OPENAI_WHISPER):
            from faster_whisper import WhisperModel

            local_dir = str(local_dir_for(self.recogn_type, self.model_name))
            compute_type = "float16" if self.is_cuda else "int8"
            try:
                self.model = WhisperModel(
                    local_dir,
                    device="cuda" if self.is_cuda else "cpu",
                    compute_type=compute_type,
                )
            except Exception:
                self.model = WhisperModel(
                    local_dir,
                    device="cuda" if self.is_cuda else "cpu",
                    compute_type="float32",
                )
        else:
            raise ValueError(f"LiveModelSession unsupported recogn_type={self.recogn_type}")

    def release(self) -> None:
        self.model = None

    def transcribe_chunk(
        self,
        audio_16k: Union[np.ndarray, str],
    ) -> str:
        self.ensure_loaded()
        if self.recogn_type == QWENASR:
            path = audio_16k if isinstance(audio_16k, str) else self._write_temp_wav(audio_16k)
            texts = transcribe_files(
                self.model, [path], language=self._qwen_lang
            )
            return texts[0] if texts else ""

        lang = None
        if self.detect_language and self.detect_language != "auto":
            lang = self.detect_language.split("-")[0]
        segments, _ = self.model.transcribe(
            audio_16k,
            language=lang,
            vad_filter=True,
            without_timestamps=True,
        )
        parts = []
        for seg in segments:
            t = (seg.text or "").strip()
            if t:
                parts.append(t)
        return " ".join(parts)

    def _write_temp_wav(self, audio: np.ndarray) -> str:
        import tempfile
        import wave

        fd, path = tempfile.mkstemp(suffix=".wav")
        import os

        os.close(fd)
        samples = np.clip(audio, -1.0, 1.0)
        int16 = (samples * 32767).astype(np.int16)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(int16.tobytes())
        return path
