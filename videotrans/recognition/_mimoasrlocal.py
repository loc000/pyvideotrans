import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Union

from videotrans.configure.config import TEMP_DIR, logger
from videotrans.recognition._base import BaseRecogn
from videotrans.recognition import check_mimo_asr_installed
from videotrans.task.taskcfg import SrtItem
from videotrans.process.mimo_asr_fun import mimo_asr_fun


@dataclass
class MimoasrlocalRecogn(BaseRecogn):

    def _download(self):
        from videotrans import recognition
        from videotrans.recognition.model_assets import ensure_assets

        ensure_assets(
            recognition.MIMO_ASR,
            self.model_name,
            detect_language=self.detect_language,
            callback=self._process_callback,
        )

    def _exec(self) -> Union[List[SrtItem], None]:
        if self._exit():
            return
        check_mimo_asr_installed()

        logs_file = f"{TEMP_DIR}/{self.uuid}/mimo-asr-{time.time()}.log"
        title = "MiMo-V2.5-ASR"
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
            callback=mimo_asr_fun, title=title, is_cuda=self.is_cuda, kwargs=kwargs
        )
        logger.debug(f"MiMo-asr returned: {jsdata=}")
        return jsdata
