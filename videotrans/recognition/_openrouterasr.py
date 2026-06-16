import base64
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import requests
from tenacity import (
    after_log,
    before_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_fixed,
)

from videotrans.configure.config import logger, params, settings
from videotrans.configure.excepts import NO_RETRY_EXCEPT, SpeechToTextError, StopTask
from videotrans.recognition._base import BaseRecogn
from videotrans.task.taskcfg import SrtItem
from videotrans.util import tools
from videotrans.util.help_http_debug import (
    format_openrouter_stt_debug_block,
    openai_transcription_url,
)

SUPPORTED_AUDIO_FORMATS = {
    "wav", "mp3", "flac", "ogg", "m4a", "aac", "opus", "aiff", "pcm16", "pcm24",
}


def audio_format_from_path(path: str) -> str:
    ext = Path(path).suffix.lower().lstrip(".")
    return ext if ext in SUPPORTED_AUDIO_FORMATS else "wav"


def build_transcribe_payload(
    *,
    model: str,
    audio_path: str,
    language: Optional[str] = None,
) -> dict:
    """OpenRouter STT: POST /audio/transcriptions with JSON input_audio."""
    b64 = base64.b64encode(Path(audio_path).read_bytes()).decode("utf-8")
    payload = {
        "model": model,
        "input_audio": {
            "data": b64,
            "format": audio_format_from_path(audio_path),
        },
    }
    if language:
        payload["language"] = language
    return payload


def parse_transcript(data: dict) -> str:
    if not isinstance(data, dict):
        return ""
    if data.get("text"):
        return str(data["text"]).strip()
    choices = data.get("choices") or []
    if choices:
        ch = choices[0]
        if isinstance(ch, dict):
            if ch.get("text"):
                return str(ch["text"]).strip()
            msg = ch.get("message") or {}
            if isinstance(msg, dict) and msg.get("content"):
                content = msg["content"]
                if isinstance(content, str):
                    return content.strip()
                if isinstance(content, list):
                    return "".join(
                        str(x.get("text", ""))
                        for x in content
                        if isinstance(x, dict)
                    ).strip()
    return ""


@dataclass
class OpenRouterASRRecogn(BaseRecogn):

    def __post_init__(self):
        super().__post_init__()
        self.api_url = tools.process_openai_api(
            params.get("openrouter_asr_url") or "https://openrouter.ai/api/v1"
        )
        self.api_key = params.get("openrouter_asr_key", "")

    def _request_headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if "openrouter" in self.api_url.lower():
            headers["HTTP-Referer"] = "https://pyvideotrans.com"
            headers["X-Title"] = "pyvideotrans"
        return headers

    def _transcribe(self, *, audio_path: str, model: str, language: str) -> str:
        payload = build_transcribe_payload(
            model=model,
            audio_path=audio_path,
            language=language or None,
        )
        url = openai_transcription_url(self.api_url)
        proxies = None
        if self.proxy_str:
            proxies = {"http": self.proxy_str, "https": self.proxy_str}
        try:
            response = requests.post(
                url,
                json=payload,
                headers=self._request_headers(),
                proxies=proxies,
                timeout=600,
            )
            if response.status_code in (401, 403, 404):
                raise StopTask(response.text)
            response.raise_for_status()
            return parse_transcript(response.json())
        except StopTask:
            raise
        except Exception as e:
            from videotrans.configure.excepts import get_msg_from_except

            debug = format_openrouter_stt_debug_block(
                base_url=self.api_url,
                api_key=self.api_key,
                model=model,
                audio_path=audio_path,
                language=language or None,
                proxy=self.proxy_str,
            )
            raise SpeechToTextError(f"{get_msg_from_except(e)}\n\n{debug}") from e

    @retry(
        retry=retry_if_not_exception_type(NO_RETRY_EXCEPT),
        stop=(stop_after_attempt(settings.get("retry_nums"))),
        wait=wait_fixed(2),
        before=before_log(logger, logging.INFO),
        after=after_log(logger, logging.INFO),
    )
    def _transcribe_chunk(self, *, audio_path: str, model: str, language: str) -> str:
        return self._transcribe(audio_path=audio_path, model=model, language=language)

    def _exec(self) -> Union[List[SrtItem], None]:
        if self._exit():
            return
        if not self.api_key:
            raise SpeechToTextError("OpenRouter ASR API key is not configured")

        model = (self.model_name or params.get("openrouter_asr_model") or "").strip()
        if not model:
            raise SpeechToTextError("No OpenRouter ASR model selected")

        language = (self.detect_language or "")[:2].lower()
        raws = self.cut_audio()
        if not raws:
            return None

        ok_nums = 0
        err = ""
        for i, it in enumerate(raws):
            if self._exit():
                return
            try:
                txt = self._transcribe_chunk(
                    audio_path=it["filename"],
                    model=model,
                    language=language,
                )
            except StopTask:
                raise
            except SpeechToTextError as e:
                err = str(e)
                raise
            except Exception as e:
                err = str(e)
                continue
            if txt:
                it["text"] = txt
                ok_nums += 1
                self.signal(text=f"{i + 1}/{len(raws)}")
                self.signal(
                    text=f'{it["line"]}\n{it["time"]}\n{txt}\n\n',
                    type="subtitle",
                )

        if ok_nums < 1:
            raise SpeechToTextError(err or "OpenRouter ASR returned no transcript")
        return raws