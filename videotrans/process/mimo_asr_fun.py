"""Subprocess worker for MiMo-V2.5-ASR local recognition."""
import json
import traceback
from pathlib import Path
from typing import List, Tuple, Union

from videotrans.configure.config import logger
from videotrans.process.stt_fun import _write_log
from videotrans.task.taskcfg import SrtItem


def mimo_asr_fun(
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
    from videotrans.recognition.mimo_asr_load import load_mimo_asr, transcribe_files

    try:
        _write_log(
            logs_file,
            json.dumps({"type": "logs", "text": f"Load MiMo-V2.5-ASR cuda={is_cuda}"}),
        )
        model = load_mimo_asr(model_name or "", is_cuda=is_cuda)
        srts: List[SrtItem] = [
            SrtItem(**item)
            for item in json.loads(
                Path(cut_audio_list).read_text(encoding="utf-8")
            )
        ]
        for i, it in enumerate(srts):
            path = it.get("filename") or ""
            if not path or not Path(path).is_file():
                continue
            texts = transcribe_files(
                model, [path], detect_language=detect_language
            )
            it["text"] = texts[0] if texts else ""
            if not it["text"]:
                logger.warning(
                    f"MiMo ASR returned empty text for {path} (segment {i + 1})"
                )
            _write_log(
                logs_file,
                json.dumps(
                    {"type": "subtitle", "text": it["text"]},
                ),
            )
        return srts, None
    except Exception as e:
        msg = traceback.format_exc()
        return False, f"{e}:{msg}"
