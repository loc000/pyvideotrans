"""Subprocess worker for Nemotron 3.5 ASR batch recognition."""
import json
import traceback
from pathlib import Path
from typing import List, Tuple, Union

from videotrans.configure.config import logger
from videotrans.process.stt_fun import _write_log
from videotrans.recognition.nemotron_asr_load import load_nemotron_asr
from videotrans.task.taskcfg import SrtItem


def nemotron_asr_fun(
    cut_audio_list=None,
    logs_file=None,
    is_cuda=False,
    audio_file=None,
    model_name=None,
    detect_language=None,
    device_index=0,
) -> Tuple[Union[List[SrtItem], bool], Union[str, None]]:
    from videotrans.configure.config import _suppress_third_party_console_noise

    _suppress_third_party_console_noise()
    try:
        _write_log(
            logs_file,
            json.dumps(
                {"type": "logs", "text": f"Load Nemotron ASR cuda={is_cuda}"}
            ),
        )
        model = load_nemotron_asr(
            model_name or "",
            is_cuda=is_cuda,
            detect_language=detect_language,
        )
        srts: List[SrtItem] = [
            SrtItem(**item)
            for item in json.loads(
                Path(cut_audio_list).read_text(encoding="utf-8")
            )
        ]
        paths = [it.get("filename") for it in srts if it.get("filename")]
        if paths:
            texts = model.transcribe(audio=paths, batch_size=1)
            if isinstance(texts, list):
                for i, it in enumerate(srts):
                    if i < len(texts):
                        t = texts[i]
                        it["text"] = (t.text if hasattr(t, "text") else str(t)).strip()
            else:
                if srts:
                    srts[0]["text"] = str(texts).strip()
        for it in srts:
            if not it.get("text"):
                logger.warning(
                    f"Nemotron ASR returned empty text for {it.get('filename')}"
                )
            _write_log(
                logs_file,
                json.dumps({"type": "subtitle", "text": it.get("text", "")}),
            )
        return srts, None
    except Exception as e:
        msg = traceback.format_exc()
        return False, f"{e}:{msg}"
