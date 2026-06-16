"""Format HTTP API calls as curl commands for troubleshooting."""
import json
import os
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional


def openai_transcription_url(base_url: str) -> str:
    """Match OpenAI Python SDK: POST {base_url}/audio/transcriptions."""
    base = (base_url or "https://api.openai.com/v1").strip().rstrip("/")
    return f"{base}/audio/transcriptions"


def _curl_quote(value: str) -> str:
    return shlex.quote(str(value))


def format_openai_transcription_curl(
    *,
    base_url: str,
    api_key: str,
    model: str,
    audio_path: str,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    response_format: str = "json",
    proxy: Optional[str] = None,
    extra_form: Optional[Dict[str, str]] = None,
    mask_key: bool = True,
) -> str:
    """Build an equivalent curl command for OpenAI audio transcriptions."""
    url = openai_transcription_url(base_url)
    audio_path = Path(audio_path).resolve().as_posix()
    key_display = (
        (api_key[:8] + "..." if len(api_key) > 8 else api_key)
        if mask_key and api_key
        else (api_key or "<empty>")
    )

    lines = [f"curl -X POST {_curl_quote(url)} \\"]
    if proxy:
        lines.append(f"  --proxy {_curl_quote(proxy)} \\")
    lines.append(f'  -H "Authorization: Bearer {key_display}" \\')
    lines.append(f"  -F {_curl_quote(f'file=@{audio_path}')} \\")
    lines.append(f"  -F {_curl_quote(f'model={model}')} \\")
    if language:
        lines.append(f"  -F {_curl_quote(f'language={language}')} \\")
    if prompt:
        lines.append(f"  -F {_curl_quote(f'prompt={prompt}')} \\")
    if response_format:
        lines.append(f"  -F {_curl_quote(f'response_format={response_format}')} \\")
    if extra_form:
        for k, v in extra_form.items():
            if v is not None and v != "":
                lines.append(f"  -F {_curl_quote(f'{k}={v}')} \\")
    if lines[-1].endswith(" \\"):
        lines[-1] = lines[-1][:-2]
    return "\n".join(lines)


def format_request_debug_block(
    *,
    base_url: str,
    api_key: str,
    model: str,
    audio_path: str,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    response_format: str = "json",
    proxy: Optional[str] = None,
    extra_form: Optional[Dict[str, str]] = None,
) -> str:
    """Human-readable block appended to API error messages."""
    url = openai_transcription_url(base_url)
    audio_name = os.path.basename(audio_path)
    lines = [
        "--- Request (for troubleshooting) ---",
        f"POST {url}",
        f"model={model}",
        f"file=@{audio_name} ({Path(audio_path).resolve()})",
    ]
    if language:
        lines.append(f"language={language}")
    if prompt:
        lines.append(f"prompt={prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    if response_format:
        lines.append(f"response_format={response_format}")
    if proxy:
        lines.append(f"proxy={proxy}")
    if extra_form:
        for k, v in extra_form.items():
            if v is not None and v != "":
                lines.append(f"{k}={v}")
    lines.append("")
    lines.append("Equivalent curl:")
    lines.append(
        format_openai_transcription_curl(
            base_url=base_url,
            api_key=api_key,
            model=model,
            audio_path=audio_path,
            language=language,
            prompt=prompt,
            response_format=response_format,
            proxy=proxy,
            extra_form=extra_form,
        )
    )
    return "\n".join(lines)


def format_openrouter_stt_debug_block(
    *,
    base_url: str,
    api_key: str,
    model: str,
    audio_path: str,
    language: Optional[str] = None,
    proxy: Optional[str] = None,
) -> str:
    """Debug block for OpenRouter JSON STT: POST /audio/transcriptions."""
    url = openai_transcription_url(base_url)
    audio_name = os.path.basename(audio_path)
    fmt = Path(audio_path).suffix.lower().lstrip(".") or "wav"
    key_display = _mask_api_key(api_key)
    lines = [
        "--- Request (for troubleshooting) ---",
        f"POST {url}",
        f"model={model}",
        f"input_audio.data=<base64 of {audio_name}> ({Path(audio_path).resolve()})",
        f"input_audio.format={fmt}",
    ]
    if language:
        lines.append(f"language={language}")
    if proxy:
        lines.append(f"proxy={proxy}")
    lines.append("")
    lines.append("Equivalent curl (JSON + base64):")
    lines.append(f"curl -X POST {_curl_quote(url)} \\")
    if proxy:
        lines.append(f"  --proxy {_curl_quote(proxy)} \\")
    lines.append(f'  -H "Authorization: Bearer {key_display}" \\')
    lines.append('  -H "Content-Type: application/json" \\')
    lang_json = f', "language": "{language}"' if language else ""
    lines.append(
        '  -d \'{"model": "'
        + model
        + '", "input_audio": {"data": "<PUT-BASE64-HERE>", "format": "'
        + fmt
        + '"'
        + lang_json
        + "}}'"
    )
    return "\n".join(lines)


def openai_chat_completions_url(base_url: str) -> str:
    """Match OpenAI Python SDK: POST {base_url}/chat/completions."""
    base = (base_url or "https://api.openai.com/v1").strip().rstrip("/")
    return f"{base}/chat/completions"


def _mask_api_key(api_key: str, mask_key: bool = True) -> str:
    if not api_key:
        return "<empty>"
    if not mask_key:
        return api_key
    return api_key[:8] + "..." if len(api_key) > 8 else api_key


def _truncate_messages_for_debug(
    messages: List[Dict[str, Any]], max_content_len: int = 200
) -> List[Dict[str, Any]]:
    out = []
    for msg in messages:
        m = dict(msg)
        content = m.get("content")
        if isinstance(content, str) and len(content) > max_content_len:
            m["content"] = content[:max_content_len] + "...(truncated)"
        elif isinstance(content, list):
            truncated = []
            for part in content:
                p = dict(part)
                if p.get("type") == "input_audio":
                    ia = p.get("input_audio") or {}
                    data = ia.get("data", "")
                    p["input_audio"] = {
                        "format": ia.get("format"),
                        "data": f"<base64 {len(data)} chars>",
                    }
                truncated.append(p)
            m["content"] = truncated
        out.append(m)
    return out


def _chat_body_for_debug(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    body = {}
    for key, val in kwargs.items():
        if val is None:
            continue
        if key == "messages" and isinstance(val, list):
            body[key] = _truncate_messages_for_debug(val)
        elif key == "timeout":
            continue
        else:
            body[key] = val
    return body


def format_openai_chat_completions_curl(
    *,
    base_url: str,
    api_key: str,
    kwargs: Dict[str, Any],
    proxy: Optional[str] = None,
    mask_key: bool = True,
) -> str:
    url = openai_chat_completions_url(base_url)
    body = _chat_body_for_debug(kwargs)
    payload = json.dumps(body, ensure_ascii=False, default=str)
    key_display = _mask_api_key(api_key, mask_key)

    lines = [f"curl -X POST {_curl_quote(url)} \\"]
    if proxy:
        lines.append(f"  --proxy {_curl_quote(proxy)} \\")
    lines.append(f'  -H "Authorization: Bearer {key_display}" \\')
    lines.append('  -H "Content-Type: application/json" \\')
    lines.append(f"  -d {_curl_quote(payload)}")
    return "\n".join(lines)


def format_chat_request_debug_block(
    *,
    base_url: str,
    api_key: str,
    kwargs: Dict[str, Any],
    proxy: Optional[str] = None,
) -> str:
    url = openai_chat_completions_url(base_url)
    body = _chat_body_for_debug(kwargs)
    lines = [
        "--- Request (for troubleshooting) ---",
        f"POST {url}",
        f"model={body.get('model', '')}",
    ]
    if proxy:
        lines.append(f"proxy={proxy}")
    msgs = body.get("messages") or []
    for i, m in enumerate(msgs):
        role = m.get("role", "?")
        content = m.get("content", "")
        preview = (
            (content[:120] + "...")
            if isinstance(content, str) and len(content) > 120
            else content
        )
        lines.append(f"messages[{i}] role={role}: {preview}")
    lines.append("")
    lines.append("Equivalent curl:")
    lines.append(
        format_openai_chat_completions_curl(
            base_url=base_url,
            api_key=api_key,
            kwargs=kwargs,
            proxy=proxy,
        )
    )
    return "\n".join(lines)
