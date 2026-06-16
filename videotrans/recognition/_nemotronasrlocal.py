import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Union

from videotrans.configure.config import TEMP_DIR, logger
from videotrans.recognition._base import BaseRecogn
from videotrans.recognition import check_nemotron_asr_installed
from videotrans.task.taskcfg import SrtItem
from videotrans.process.nemotron_asr_fun import nemotron_asr_fun


@dataclass
class NemotronasrlocalRecogn(BaseRecogn):

    def _download(self):
        from videotrans import recognition
        from videotrans.recognition.model_assets import ensure_assets

        ensure_assets(
            recognition.NEMOTRON_ASR,
            self.model_name,
            detect_language=self.detect_language,
            callback=self._process_callback,
        )

    def _exec(self) -> Union[List[SrtItem], None]:
        if self._exit():
            return
        check_nemotron_asr_installed()

        logs_file = f"{TEMP_DIR}/{self.uuid}/nemotron-asr-{time.time()}.log"
        title = "Nemotron-3.5-ASR"
        cut_audio_list_file = (
            f"{TEMP_DIR}/{self.uuid}/cut_audio_list_{time.time()}.json"
        )
        Path(cut_audio_list_file).write_text(
            json.dumps([asdict(item) for item in self.cut_audio()]),
            encoding="utf-8",
        )
        kwargs = {
            "cut_audio_list": cut_audio_list_file,
            "logs_file": logs_file,
            "is_cuda": self.is_cuda,
            "audio_file": self.audio_file,
            "model_name": self.model_name,
            "detect_language": self.detect_language,
        }
        jsdata = self._new_process(
            callback=nemotron_asr_fun, title=title, is_cuda=self.is_cuda, kwargs=kwargs
        )
        logger.debug(f"Nemotron-asr returned: {jsdata=}")
        return jsdata
