import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List,  Union
from videotrans.configure.config import TEMP_DIR
from videotrans.process import paraformer, funasr_mlt
from videotrans.recognition._base import BaseRecogn
from videotrans.task.taskcfg import SrtItem
from videotrans.util import tools


@dataclass
class FunasrRecogn(BaseRecogn):

    def _download(self):
        from videotrans import recognition
        from videotrans.recognition.model_assets import ensure_assets

        ensure_assets(
            recognition.FUNASR_CN,
            self.model_name,
            detect_language=self.detect_language,
            callback=self._process_callback,
        )

    def _resolve_model_name(self) -> str:
        from videotrans.recognition.model_assets import resolve_funasr_model_name

        return resolve_funasr_model_name(self.model_name, self.detect_language)

    def _exec(self) -> Union[List[SrtItem], None]:
        if self._exit():
            return
        self.model_name = self._resolve_model_name()
        self.signal(text=f"load {self.model_name}")
        logs_file = f'{TEMP_DIR}/{self.uuid}/funasr-{self.detect_language}-{time.time()}.log'
        if self.model_name != 'paraformer-zh':
            cut_audio_list_file = f'{TEMP_DIR}/{self.uuid}/cut_audio_list_{time.time()}.json'
            Path(cut_audio_list_file).write_text( json.dumps( [ asdict(item) for item in self.cut_audio() ] ),encoding='utf-8')
        else:
            cut_audio_list_file=None
        kwars = {
            "cut_audio_list":   cut_audio_list_file,
            "detect_language": self.detect_language,
            "model_name": self.model_name,
            "logs_file": logs_file,
            "is_cuda": self.is_cuda,
            "audio_file": self.audio_file,
            "max_speakers": self.max_speakers,
            "cache_folder": self.cache_folder

        }
        raws=self._new_process(callback=paraformer if self.model_name == 'paraformer-zh' else funasr_mlt,title=f'STT use {self.model_name}',is_cuda=self.is_cuda,kwargs=kwars)
        return raws
