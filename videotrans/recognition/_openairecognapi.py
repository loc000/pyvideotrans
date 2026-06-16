# zh_recogn 识别
import base64
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List,  Union

import httpx
import requests
from openai import OpenAI
from videotrans.configure.config import params,  logger, TEMP_DIR
from videotrans.configure.excepts import SpeechToTextError, StopTask
from videotrans.recognition._base import BaseRecogn
from videotrans.task.taskcfg import SrtItem
from videotrans.util import tools
from videotrans.util.help_http_debug import format_request_debug_block


@dataclass
class OpenaiAPIRecogn(BaseRecogn):

    def _transcribe_create(self, client, *, audio_path: str, **kwargs):
        """Call transcriptions.create; on failure attach curl debug info."""
        create_kwargs = dict(kwargs)
        model = create_kwargs.get("model", params.get("openairecognapi_model", "whisper-1"))
        language = create_kwargs.get("language")
        prompt = create_kwargs.get("prompt") or params.get("openairecognapi_prompt", "")
        response_format = create_kwargs.get("response_format", "json")
        extra_form = {}
        for key in (
            "chunking_strategy",
            "timestamp_granularities",
        ):
            if key in create_kwargs:
                val = create_kwargs[key]
                if isinstance(val, list):
                    extra_form[key] = ",".join(str(x) for x in val)
                else:
                    extra_form[key] = str(val)
        try:
            with open(audio_path, "rb") as file:
                create_kwargs["file"] = (
                    os.path.basename(audio_path),
                    file.read(),
                )
                return client.audio.transcriptions.create(**create_kwargs)
        except Exception as e:
            from videotrans.configure.excepts import get_msg_from_except

            debug = format_request_debug_block(
                base_url=self.api_url,
                api_key=params.get("openairecognapi_key", ""),
                model=model,
                audio_path=audio_path,
                language=language,
                prompt=prompt,
                response_format=response_format,
                proxy=self.proxy_str,
                extra_form=extra_form or None,
            )
            raise SpeechToTextError(
                f"{get_msg_from_except(e)}\n\n{debug}"
            ) from e

    def __post_init__(self):
        super().__post_init__()
        self.api_url = params.get('openairecognapi_url', '')
        u = (self.api_url or "").lower()
        self._is_openrouter = "openrouter.ai" in u or "openrouter" in u


    def _exec(self) -> Union[List[SrtItem], None]:
        if self._exit(): return
        model_name = params.get("openairecognapi_model", '')
        # 如果是 gpt-4o-transcribe-diarize 说话人识别默认
        if model_name.lower() == 'gpt-4o-transcribe-diarize':
            return self._diarize()
        # 如果是第三方或 gpt-4o-模型
        if not re.search(r'api\.openai\.com/v1', self.api_url) or model_name.find(
                'gpt-4o-') > -1:
            return self._thrid_api()

        mp3_tmp = TEMP_DIR + f'/recogn{time.time()}.mp3'
        tools.runffmpeg([
            "-y",
            "-i",
            Path(self.audio_file).as_posix(),
            "-ac",
            "1",
            "-ar",
            "16000",
            mp3_tmp
        ])

        self.audio_file = mp3_tmp
        if not Path(self.audio_file).is_file():
            raise SpeechToTextError(f'No {self.audio_file}')
        # 发送请求
        raws = []
        client = OpenAI(api_key=params.get('openairecognapi_key', ''), base_url=self.api_url,
                        http_client=httpx.Client(proxy=self.proxy_str))
        transcript = self._transcribe_create(
            client,
            audio_path=self.audio_file,
            model=model_name,
            prompt=params.get('openairecognapi_prompt', ''),
            language=self.detect_language[:2].lower(),
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )
        if not hasattr(transcript, 'segments'):
            return self._thrid_api()
        for i, it in enumerate(transcript.segments):
            raws.append(SrtItem(
                line=len(raws) + 1,
                start_time=it.start * 1000,
                end_time=it.end * 1000,
                text=it.text,
                time=tools.ms_to_time_string(ms=it.start * 1000) + ' --> ' + tools.ms_to_time_string(
                    ms=it.end * 1000),
            ))
        return raws

    def _thrid_api(self)->Union[List[SrtItem], None]:
        # 发送请求
        raws = self.cut_audio()
        model_name = params.get("openairecognapi_model", 'whisper-1')
        language = self.detect_language[:2].lower()
        prompt = params.get('openairecognapi_prompt', '')

        if self._is_openrouter:
            for i, it in enumerate(raws):
                if self._exit():
                    return
                txt = self._openrouter_transcribe(
                    audio_path=it['filename'],
                    model=model_name,
                    language=language,
                    prompt=prompt,
                )
                if txt and txt.strip():
                    raws[i]['text'] = txt
            return raws

        client = OpenAI(
            api_key=params.get('openairecognapi_key', ''),
            base_url=self.api_url,
            http_client=httpx.Client(proxy=self.proxy_str or None)
        )
        for i, it in enumerate(raws):
            transcript = self._transcribe_create(
                client,
                audio_path=it['filename'],
                model=model_name,
                prompt=prompt,
                language=language,
                response_format="json",
            )
            if not hasattr(transcript, 'text') or not transcript.text or not transcript.text.strip():
                continue
            raws[i]['text'] = transcript.text
        return raws

    def _openrouter_transcribe(self, *, audio_path: str, model: str, language: str, prompt: str = "") -> str:
        """OpenRouter /audio/transcriptions expects JSON + base64 (not multipart/form-data)."""
        try:
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
            b64 = base64.b64encode(audio_bytes).decode("utf-8")

            # derive format from the actual chunk file (wav / mp3 etc.)
            ext = Path(audio_path).suffix.lower().lstrip(".")
            audio_format = ext if ext in {"wav", "mp3", "flac", "ogg", "m4a", "aac", "opus"} else "wav"

            # normalize base url -> .../audio/transcriptions
            base = (self.api_url or "https://openrouter.ai/api/v1").rstrip("/")
            if base.endswith("/audio/transcriptions"):
                url = base
            else:
                url = f"{base}/audio/transcriptions"

            payload = {
                "model": model,
                "input_audio": {
                    "data": b64,
                    "format": audio_format,
                },
            }
            if language:
                payload["language"] = language
            # Note: "prompt" (for context biasing) is not part of the standard OpenRouter STTRequest schema.
            # Temperature is supported by the schema but rarely needed for ASR.

            headers = {
                "Authorization": f"Bearer {params.get('openairecognapi_key', '')}",
                "Content-Type": "application/json",
                # Optional but recommended for OpenRouter leaderboards
                "HTTP-Referer": "https://pyvideotrans.com",
                "X-Title": "pyvideotrans",
            }

            proxies = None
            if self.proxy_str:
                proxies = {"http": self.proxy_str, "https": self.proxy_str}

            resp = requests.post(url, json=payload, headers=headers, proxies=proxies, timeout=600)
            resp.raise_for_status()
            data = resp.json()

            if isinstance(data, dict):
                # Most common shape from OpenRouter STT docs/examples
                if "text" in data and data["text"]:
                    return data["text"]
                # Some providers may return choices-style or message content
                if "choices" in data and data["choices"]:
                    ch = data["choices"][0]
                    if isinstance(ch, dict):
                        if "text" in ch:
                            return ch["text"]
                        msg = ch.get("message") or {}
                        if isinstance(msg, dict) and msg.get("content"):
                            # fallback for chat-style responses
                            c = msg["content"]
                            if isinstance(c, list):
                                return "".join(x.get("text", "") for x in c if isinstance(x, dict))
                            return str(c)
            return ""
        except Exception as e:
            from videotrans.configure.excepts import get_msg_from_except
            debug = self._format_openrouter_debug(audio_path, model, language, prompt)
            raise SpeechToTextError(f"{get_msg_from_except(e)}\n\n{debug}") from e

    def _format_openrouter_debug(self, audio_path: str, model: str, language: str, prompt: str) -> str:
        """Produce a helpful debug block + correct curl for the JSON+base64 format."""
        base = (self.api_url or "https://openrouter.ai/api/v1").rstrip("/")
        url = base if base.endswith("/audio/transcriptions") else f"{base}/audio/transcriptions"
        name = Path(audio_path).name
        key = params.get("openairecognapi_key", "")
        key_display = (key[:10] + "..." + key[-4:]) if key and len(key) > 14 else (key or "<empty>")

        lines = [
            "--- Request (for troubleshooting) ---",
            f"POST {url}",
            f"model={model}",
            f"input_audio.data=<base64 of {name}> (file: {Path(audio_path).resolve()})",
            f"input_audio.format={Path(audio_path).suffix.lower().lstrip('.') or 'wav'}",
        ]
        if language:
            lines.append(f"language={language}")
        if prompt:
            lines.append(f"prompt={prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        lines.append("")
        lines.append("Equivalent curl (JSON + base64):")
        lines.append('curl -X POST ' + url + ' \\')
        lines.append('  -H "Authorization: Bearer ' + key_display + '" \\')
        lines.append('  -H "Content-Type: application/json" \\')
        lang_part = (language or "")
        lines.append('  -d \'{"model": "' + model + '", "input_audio": {"data": "<PUT-BASE64-HERE>", "format": "wav"}, "language": "' + lang_part + '"}\'')
        return "\n".join(lines)

    def _diarize(self)->Union[List[SrtItem], None]:
        client = OpenAI(
            api_key=params.get('openairecognapi_key', ''),
            base_url=self.api_url
        )
        raws = []
        speaker_list = []
        speaker_name = []
        transcript = self._transcribe_create(
            client,
            audio_path=self.audio_file,
            model='gpt-4o-transcribe-diarize',
            language=self.detect_language[:2].lower(),
            chunking_strategy="auto",
            response_format="diarized_json",
        )

        if not hasattr(transcript, 'segments') or not transcript.segments:
            raise StopTask('No support gpt-4o-transcribe-diarize')
        for it in transcript.segments:
            raws.append(SrtItem(
                line=len(raws) + 1,
                start_time=it.start * 1000,
                end_time=it.end * 1000,
                text=it.text,
                time=tools.ms_to_time_string(ms=it.start * 1000) + ' --> ' + tools.ms_to_time_string(
                    ms=it.end * 1000),
                ))
            if self.max_speakers>-1:
                sp = getattr(it,"speaker",'-')
                speaker_list.append(sp)
                if sp not in speaker_name:
                    speaker_name.append(sp)

        if speaker_name:
            try:
                #默认未识别出后的回退说话人
                next_spk=f'spk{len(speaker_name)}'
                for i,it in enumerate(speaker_list):
                    if it=='-':
                        speaker_list[i]=next_spk
                    else:
                        speaker_list[i]=f'spk{speaker_name.index(it)}'
                if speaker_list:
                    Path(f'{self.cache_folder}/speaker.json').write_text(json.dumps(speaker_list), encoding='utf-8')
            except Exception as e:
                logger.exception(f'说话人重排序出错，忽略{e}',exc_info=True)
        return raws