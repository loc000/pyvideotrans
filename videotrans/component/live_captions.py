import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QPoint, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QCloseEvent, QDesktopServices, QIcon, QMouseEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import sys

from videotrans import recognition, translator
from videotrans.component.realtime_engine import (
    CaptureDevice,
    CheckAudioDevices,
    ChunkedRecognWorker,
    LIVE_SHERPA_RECOGN,
    Worker,
    models_ready,
)
from videotrans.component.realtime_ui_base import RealtimeUiBase
from videotrans.configure.config import HOME_DIR, ROOT_DIR, TEMP_DIR, app_cfg, params, settings, tr
from videotrans.task.taskcfg import SrtItem
from videotrans.util import tools
from videotrans.util.help_srt import ms_to_time_string

LIVE_CAPTIONS_DIR = f"{HOME_DIR}/live_captions"

_TRANSLATE_MODEL_CHANNELS = {
    translator.LOCALLLM_INDEX: "localllm_model",
    translator.GEMINI_INDEX: "gemini_model",
    translator.CHATGPT_INDEX: "chatgpt_model",
    translator.AZUREGPT_INDEX: "azure_model",
    translator.ZIJIE_INDEX: "zijiehuoshan_model",
}

# ASR channels that show the CUDA checkbox (local GPU models)
_CUDA_RECOGN_TYPES = {
    recognition.FASTER_WHISPER,
    recognition.OPENAI_WHISPER,
    recognition.QWENASR,
    recognition.FUNASR_CN,
    recognition.HUGGINGFACE_ASR,
    recognition.WHISPER_NET,
}


@dataclass
class CaptionSegment:
    line: int
    start_ms: int
    end_ms: int
    source: str
    target: str = ""


def segments_to_srt(segments: List[CaptionSegment], use_target: bool = False) -> str:
    blocks = []
    for seg in segments:
        text = seg.target.strip() if use_target and seg.target.strip() else seg.source
        if not text:
            continue
        startraw = ms_to_time_string(ms=seg.start_ms)
        endraw = ms_to_time_string(ms=seg.end_ms)
        blocks.append(f"{seg.line}\n{startraw} --> {endraw}\n{text}")
    return "\n\n".join(blocks) + ("\n\n" if blocks else "")


class TranslateSegmentWorker(QThread):
    done = Signal(int, str)
    error = Signal(str)

    def __init__(
        self,
        segment_idx: int,
        text: str,
        translate_type: int,
        source_code: str,
        target_code: str,
        parent=None,
    ):
        super().__init__(parent)
        self.segment_idx = segment_idx
        self.text = text
        self.translate_type = translate_type
        self.source_code = source_code
        self.target_code = target_code

    def run(self):
        try:
            item = SrtItem(text=self.text, line=1, start_time=0, end_time=1000)
            result = translator.run(
                translate_type=self.translate_type,
                text_list=[item],
                source_code=self.source_code,
                target_code=self.target_code,
                is_test=True,
            )
            if result and len(result) > 0 and (result[0].get("text") or "").strip():
                self.done.emit(self.segment_idx, result[0]["text"].strip())
            else:
                self.error.emit(tr("Translate result is empty"))
        except Exception as e:
            self.error.emit(str(e))


class CaptionOverlayWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
            | Qt.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._drag_pos: Optional[QPoint] = None
        self._partial = ""
        self._committed = ""
        self._target = ""
        self._bilingual = False
        self._font_size = 28
        self._opacity = 0.75
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(75)
        self._resize_timer.timeout.connect(lambda: QWidget.adjustSize(self))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        self.label = QLabel()
        self.label.setWordWrap(True)
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)
        self._apply_style()
        self.resize(900, 120)

    def _apply_style(self):
        alpha = int(self._opacity * 255)
        self.label.setStyleSheet(
            f"color: #ffffff; font-size: {self._font_size}px; font-weight: bold;"
            f"background-color: rgba(0, 0, 0, {alpha});"
            "padding: 12px 20px; border-radius: 8px;"
        )

    def set_appearance(self, font_size: int, opacity: float):
        self._font_size = font_size
        self._opacity = max(0.3, min(1.0, opacity))
        self._apply_style()

    def set_bilingual(self, on: bool):
        self._bilingual = on
        self._refresh()

    def update_partial(self, text: str):
        self._partial = text or ""
        self._refresh()

    def update_committed(self, source: str, target: str = ""):
        self._committed = source or ""
        self._target = target or ""
        self._partial = ""
        self._refresh()

    def _refresh(self):
        lines = []
        if self._committed:
            if self._bilingual and self._target:
                lines.append(self._committed)
                lines.append(self._target)
            elif self._target:
                lines.append(self._target)
            else:
                lines.append(self._committed)
        if self._partial:
            lines.append(self._partial)
        self.label.setText("\n".join(lines))
        self._resize_timer.start()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._drag_pos is not None and event.buttons() & Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._drag_pos = None


class LiveCaptionsWindow(RealtimeUiBase, QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(720, 520)
        self.setWindowTitle(tr("Live captions"))
        self.setWindowIcon(QIcon(f"{ROOT_DIR}/videotrans/styles/icon.ico"))

        self.worker = None
        self.transcribing = False
        self.overlay: Optional[CaptionOverlayWidget] = None
        self.segments: List[CaptionSegment] = []
        self._session_start = 0.0
        self._last_segment_end_ms = 0
        self._last_segment_time = 0.0
        self._merge_buffer = ""
        self._merge_buffer_start_ms = 0
        self._trans_worker: Optional[TranslateSegmentWorker] = None
        self._pending_trans: Optional[tuple] = None
        self._line_counter = 0
        self._model_downloading = False
        self._download_task: Optional[DownloadModel] = None

        root = QVBoxLayout(self)

        source_row = QHBoxLayout()
        source_row.addWidget(QLabel(tr("Audio source")))
        self.source_mode = QComboBox()
        self.source_mode.addItem(tr("Microphone"), "mic")
        self.source_mode.addItem(tr("System sound"), "system")
        saved_source = params.get("live_caption_audio_source", "mic")
        self.source_mode.setCurrentIndex(0 if saved_source != "system" else 1)
        self.source_mode.currentIndexChanged.connect(self._on_source_mode_change)
        source_row.addWidget(self.source_mode)
        source_row.addStretch()
        root.addLayout(source_row)

        mic_row = QHBoxLayout()
        self.combo = QComboBox()
        self.combo.setMinimumWidth(200)
        self.checkbtn = QPushButton(tr("Detect audio input"))
        self.checkbtn.clicked.connect(self.populate_audio_devices)
        mic_row.addWidget(self.combo)
        mic_row.addWidget(self.checkbtn)
        self.start_button = QPushButton(tr("Start live captions"))
        self.start_button.clicked.connect(self.toggle_transcription)
        mic_row.addWidget(self.start_button)
        mic_row.addStretch()
        root.addLayout(mic_row)

        asr_row = QHBoxLayout()
        asr_row.addWidget(QLabel(tr("ASR channel")))
        self.recogn_combo = QComboBox()
        self.recogn_combo.setMinimumWidth(180)
        self.recogn_combo.currentIndexChanged.connect(self._on_recogn_type_change)
        asr_row.addWidget(self.recogn_combo)

        asr_row.addWidget(QLabel(tr("ASR model")))
        self.recogn_model = QComboBox()
        self.recogn_model.setMinimumWidth(140)
        self.recogn_model.currentTextChanged.connect(self._on_recogn_model_change)
        asr_row.addWidget(self.recogn_model)
        asr_row.addStretch()
        root.addLayout(asr_row)

        asr_lang_row = QHBoxLayout()
        asr_lang_row.addWidget(QLabel(tr("Recognition language")))
        self.recogn_lang = QComboBox()
        self.recogn_lang.addItems(list(translator.LANGNAME_DICT.values()) + ["auto"])
        try:
            rl_idx = int(
                params.get("live_caption_recogn_language", params.get("stt_source_language", 0))
            )
        except (TypeError, ValueError):
            rl_idx = 0
        self.recogn_lang.setCurrentIndex(min(rl_idx, self.recogn_lang.count() - 1))
        asr_lang_row.addWidget(self.recogn_lang)

        self.chk_cuda = QCheckBox(tr("Enable CUDA"))
        self.chk_cuda.setChecked(bool(params.get("live_caption_cuda", params.get("stt_cuda", False))))
        self.chk_cuda.toggled.connect(self._check_cuda)
        asr_lang_row.addWidget(self.chk_cuda)
        asr_lang_row.addStretch()
        root.addLayout(asr_lang_row)

        chunk_row = QHBoxLayout()
        self.chunk_label = QLabel(tr("Chunk interval sec"))
        chunk_row.addWidget(self.chunk_label)
        self.chunk_spin = QSpinBox()
        self.chunk_spin.setRange(2, 15)
        self.chunk_spin.setValue(int(params.get("live_caption_chunk_sec", 4)))
        self.chunk_spin.setToolTip(tr("Chunk interval help"))
        chunk_row.addWidget(self.chunk_spin)
        chunk_row.addStretch()
        root.addLayout(chunk_row)

        trans_row = QHBoxLayout()
        trans_row.addWidget(QLabel(tr("Translation channel")))
        self.translate_type = QComboBox()
        self.translate_type.addItems(translator.TRANSLASTE_NAME_LIST)
        self.translate_type.setCurrentIndex(int(params.get("live_caption_translate_type", params.get("trans_translate_type", 0))))
        self.translate_type.currentIndexChanged.connect(self._on_translate_type_change)
        trans_row.addWidget(self.translate_type)

        trans_row.addWidget(QLabel(tr("Source language")))
        self.source_lang = QComboBox()
        self.source_lang.addItems(["-"] + list(translator.LANGNAME_DICT.values()))
        self.source_lang.setCurrentIndex(int(params.get("live_caption_source_language", params.get("trans_source_language", 0))))
        trans_row.addWidget(self.source_lang)

        trans_row.addWidget(QLabel(tr("Target language")))
        self.target_lang = QComboBox()
        self.target_lang.addItems(["-"] + list(translator.LANGNAME_DICT.values()))
        self.target_lang.setCurrentIndex(int(params.get("live_caption_target_language", params.get("trans_target_language", 1))))
        trans_row.addWidget(self.target_lang)
        root.addLayout(trans_row)

        trans_model_row = QHBoxLayout()
        self.trans_model_label = QLabel(tr("Translation model"))
        trans_model_row.addWidget(self.trans_model_label)
        self.trans_model_list = QComboBox()
        self.trans_model_list.setMinimumWidth(180)
        self.trans_model_list.currentTextChanged.connect(self._on_trans_model_change)
        trans_model_row.addWidget(self.trans_model_list)
        trans_model_row.addStretch()
        root.addLayout(trans_model_row)

        opts_row = QHBoxLayout()
        self.chk_translate = QCheckBox(tr("Show live translation"))
        self.chk_translate.setChecked(params.get("live_caption_show_translate", False))
        opts_row.addWidget(self.chk_translate)
        self.chk_bilingual = QCheckBox(tr("Bilingual overlay"))
        self.chk_bilingual.setChecked(params.get("live_caption_bilingual", True))
        opts_row.addWidget(self.chk_bilingual)
        self.chk_top = QCheckBox(tr("Always on top"))
        self.chk_top.setChecked(True)
        opts_row.addWidget(self.chk_top)
        root.addLayout(opts_row)

        appear_row = QHBoxLayout()
        appear_row.addWidget(QLabel(tr("Font size")))
        self.font_spin = QSpinBox()
        self.font_spin.setRange(14, 48)
        self.font_spin.setValue(int(params.get("live_caption_font_size", 28)))
        self.font_spin.valueChanged.connect(self._on_appearance_change)
        appear_row.addWidget(self.font_spin)
        appear_row.addWidget(QLabel(tr("Opacity")))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(30, 100)
        self.opacity_slider.setValue(int(float(params.get("live_caption_opacity", 0.75)) * 100))
        self.opacity_slider.valueChanged.connect(self._on_appearance_change)
        appear_row.addWidget(self.opacity_slider)
        appear_row.addStretch()
        root.addLayout(appear_row)

        self.download_log = QPlainTextEdit()
        self.download_log.setReadOnly(True)
        self.download_log.setMaximumHeight(72)
        self.download_log.setStyleSheet("color:#148cd2;background:transparent;border:none;")
        self.download_log.hide()
        root.addWidget(self.download_log)

        self.status_label = QLabel()
        self.status_label.setStyleSheet("color:#148cd2;")
        root.addWidget(self.status_label)

        btn_row = QHBoxLayout()
        self.export_srt_btn = QPushButton(tr("Export SRT"))
        self.export_srt_btn.clicked.connect(self.export_srt)
        btn_row.addWidget(self.export_srt_btn)
        self.export_txt_btn = QPushButton(tr("Export to TXT"))
        self.export_txt_btn.clicked.connect(self.export_txt)
        btn_row.addWidget(self.export_txt_btn)
        self.copy_btn = QPushButton(tr("Copy"))
        self.copy_btn.clicked.connect(self.copy_text)
        btn_row.addWidget(self.copy_btn)
        self.clear_btn = QPushButton(tr("Clear"))
        self.clear_btn.clicked.connect(self.clear_session)
        btn_row.addWidget(self.clear_btn)
        root.addLayout(btn_row)

        self.btn_opendir = QPushButton(
            f"{tr('Recording files are stored in')}: {LIVE_CAPTIONS_DIR}"
        )
        self.btn_opendir.setStyleSheet("background-color:transparent;border:0;color:#ddd")
        self.btn_opendir.clicked.connect(self.open_dir)
        root.addWidget(self.btn_opendir)

        self._init_recogn_combo()
        self._on_recogn_type_change()
        self._on_translate_type_change()
        QTimer.singleShot(500, self.populate_audio_devices)

    def _on_appearance_change(self):
        if self.overlay:
            self.overlay.set_appearance(
                self.font_spin.value(), self.opacity_slider.value() / 100.0
            )
        self._save_params()

    def _init_recogn_combo(self):
        self.recogn_combo.clear()
        self.recogn_combo.addItem(tr("Realtime Paraformer (Local)"), LIVE_SHERPA_RECOGN)
        for i, name in enumerate(recognition.RECOGN_NAME_LIST):
            self.recogn_combo.addItem(name, i)
        try:
            saved = int(params.get("live_caption_recogn_type", params.get("stt_recogn_type", LIVE_SHERPA_RECOGN)))
        except (TypeError, ValueError):
            saved = LIVE_SHERPA_RECOGN
        for idx in range(self.recogn_combo.count()):
            if self.recogn_combo.itemData(idx) == saved:
                self.recogn_combo.setCurrentIndex(idx)
                break

    def _on_recogn_type_change(self):
        recogn_type = self.recogn_combo.currentData()
        is_sherpa = recogn_type == LIVE_SHERPA_RECOGN
        self.chunk_label.setVisible(not is_sherpa)
        self.chunk_spin.setVisible(not is_sherpa)
        self.chk_cuda.setVisible(recogn_type in _CUDA_RECOGN_TYPES)

        if not is_sherpa:
            if recogn_type == recognition.Faster_Whisper_XXL and not self._show_xxl_select():
                return
            if recogn_type == recognition.Whisper_CPP and not self._show_cpp_select():
                return
            if recognition.is_input_api(recogn_type=recogn_type) is not True:
                return

        if recogn_type in recognition.ALLOW_CHANGE_MODEL:
            self.recogn_model.setDisabled(False)
            self.recogn_model.clear()
            models = recognition.get_model_by_type(recogn_type)
            self.recogn_model.addItems(models)
            saved_model = params.get("live_caption_model_name", params.get("stt_model_name", ""))
            if saved_model in models:
                self.recogn_model.setCurrentText(saved_model)
        else:
            self.recogn_model.setDisabled(True)

        if is_sherpa:
            self.status_label.setText("")
        else:
            self.status_label.setText(
                tr("Chunk recognition status").format(int(self.chunk_spin.value()))
            )
        self._ensure_models()

    def _on_recogn_model_change(self):
        if self.recogn_model.isEnabled():
            params["live_caption_model_name"] = self.recogn_model.currentText()
            params.save()

    def _check_cuda(self, state: bool):
        if state:
            try:
                import torch
                if not torch.cuda.is_available():
                    tools.show_error(tr("nocuda"))
                    self.chk_cuda.setChecked(False)
                    return False
            except ImportError:
                tools.show_error(tr("nocuda"))
                self.chk_cuda.setChecked(False)
                return False
        return True

    def _show_xxl_select(self) -> bool:
        if sys.platform != "win32":
            tools.show_error(tr("faster-whisper-xxl.exe is only available on Windows"))
            return False
        if not settings.get("Faster_Whisper_XXL") or not Path(
            settings.get("Faster_Whisper_XXL", "")
        ).exists():
            from PySide6.QtWidgets import QFileDialog

            exe, _ = QFileDialog.getOpenFileName(
                self,
                tr("Select faster-whisper-xxl.exe"),
                "C:/",
                "Files(*.exe)",
            )
            if exe:
                settings["Faster_Whisper_XXL"] = Path(exe).as_posix()
                return True
            return False
        return True

    def _show_cpp_select(self) -> bool:
        cpp_path = settings.get("Whisper_cpp", "")
        if not cpp_path or not Path(cpp_path).exists():
            from videotrans.component.set_cpp import SetWhisperCPP

            dialog = SetWhisperCPP()
            if dialog.exec():
                cpp_path = dialog.get_values()
                if cpp_path and Path(cpp_path).is_file():
                    return True
            tools.show_error(tr("Must be selected, otherwise it cannot be used"))
            return False
        return True

    def _on_translate_type_change(self):
        idx = self.translate_type.currentIndex()
        if idx in _TRANSLATE_MODEL_CHANNELS:
            key = _TRANSLATE_MODEL_CHANNELS[idx]
            models = [m.strip() for m in settings.get(key, "").split(",") if m.strip()]
            self.trans_model_list.clear()
            self.trans_model_list.addItems(models)
            current = params.get(key, "")
            if current in models:
                self.trans_model_list.setCurrentText(current)
            self.trans_model_label.setVisible(True)
            self.trans_model_list.setVisible(True)
        else:
            self.trans_model_label.setVisible(False)
            self.trans_model_list.setVisible(False)

    def _on_trans_model_change(self):
        idx = self.translate_type.currentIndex()
        name = self.trans_model_list.currentText()
        if idx in _TRANSLATE_MODEL_CHANNELS and name:
            params[_TRANSLATE_MODEL_CHANNELS[idx]] = name
            params.save()

    def _save_params(self):
        params["live_caption_translate_type"] = self.translate_type.currentIndex()
        params["live_caption_source_language"] = self.source_lang.currentIndex()
        params["live_caption_target_language"] = self.target_lang.currentIndex()
        params["live_caption_show_translate"] = self.chk_translate.isChecked()
        params["live_caption_bilingual"] = self.chk_bilingual.isChecked()
        params["live_caption_font_size"] = self.font_spin.value()
        params["live_caption_opacity"] = self.opacity_slider.value() / 100.0
        params["live_caption_audio_source"] = self.source_mode.currentData() or "mic"
        params["live_caption_recogn_type"] = self.recogn_combo.currentData()
        params["live_caption_model_name"] = self.recogn_model.currentText()
        params["live_caption_recogn_language"] = self.recogn_lang.currentIndex()
        params["live_caption_cuda"] = self.chk_cuda.isChecked()
        params["live_caption_chunk_sec"] = self.chunk_spin.value()
        params.save()

    def _on_source_mode_change(self):
        self.populate_audio_devices()

    def _system_audio_hint(self) -> str:
        if sys.platform == "win32":
            return tr("System audio hint Windows")
        if sys.platform == "darwin":
            return tr("System audio hint macOS")
        return tr("System audio hint Linux")

    def _get_lang_codes(self):
        translate_type = self.translate_type.currentIndex()
        source_code, target_code = translator.get_source_target_code(
            show_source=self.source_lang.currentText(),
            show_target=self.target_lang.currentText(),
            translate_type=translate_type,
        )
        return translate_type, source_code, target_code

    def open_dir(self):
        self.open_recordings_dir(LIVE_CAPTIONS_DIR)

    def populate_audio_devices(self):
        mode = self.source_mode.currentData() or "mic"

        def _get_dev(data):
            self.checkbtn.setDisabled(False)
            payload = json.loads(data) if isinstance(data, str) else data
            if payload.get("empty"):
                self.combo.clear()
                if mode == "system":
                    self.checkbtn.setText(tr("No system audio device found"))
                    self.status_label.setText(self._system_audio_hint())
                else:
                    self.checkbtn.setText(tr("No valid microphone exists"))
                    self.status_label.setText("")
                return
            self.combo.clear()
            for d in payload["devices"]:
                dev = CaptureDevice.from_dict(d)
                self.combo.addItem(dev.display_name, dev.to_dict())
            self.combo.setCurrentIndex(payload.get("default", 0))
            if not self._model_downloading:
                self.status_label.setText("")

        self.checkbtn.setDisabled(True)
        task = CheckAudioDevices(source_mode=mode, parent=self)
        task.devices.connect(_get_dev)
        task.start()
        self._ensure_models()

    def _show_download_progress(self, msg: str, log_widget=None):
        if not msg:
            return
        log = log_widget or self.download_log
        log.show()
        log.setPlainText(msg)
        self.status_label.setText(msg[:200])
        sb = log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _set_download_ui_busy(self, busy: bool, **kwargs):
        check_button = kwargs.get("check_button", self.checkbtn)
        start_button = kwargs.get("start_button", self.start_button)
        self._model_downloading = busy
        if check_button is not None:
            check_button.setDisabled(busy)
        if start_button is not None and not self.transcribing:
            start_button.setDisabled(busy)
        if busy:
            self.download_log.show()
        elif models_ready() or self.recogn_combo.currentData() != LIVE_SHERPA_RECOGN:
            self.download_log.hide()

    def _ensure_models(self):
        if self.recogn_combo.currentData() != LIVE_SHERPA_RECOGN:
            self.start_button.setDisabled(False)
            return
        self._ensure_sherpa_models(
            log_widget=self.download_log,
            check_button=self.checkbtn,
            start_button=self.start_button,
            on_complete=lambda: self.checkbtn.setDisabled(False),
        )

    def _download_callback(self, msg):
        self._show_download_progress(msg, self.download_log)
        if msg.startswith("Error:"):
            self._set_download_ui_busy(
                False, check_button=self.checkbtn, start_button=self.start_button
            )
            self.checkbtn.setDisabled(False)
            QMessageBox.critical(self, "Error", msg)
        elif msg.endswith(" end"):
            self._set_download_ui_busy(
                False, check_button=self.checkbtn, start_button=self.start_button
            )
            self._show_download_progress(tr("Model download complete"), self.download_log)
            QTimer.singleShot(2000, self.download_log.hide)

    def _show_overlay(self):
        if self.overlay is None:
            self.overlay = CaptionOverlayWidget()
        self.overlay.set_appearance(
            self.font_spin.value(), self.opacity_slider.value() / 100.0
        )
        self.overlay.set_bilingual(self.chk_bilingual.isChecked())
        screen = QApplication.primaryScreen().availableGeometry()
        self.overlay.move(
            screen.center().x() - self.overlay.width() // 2,
            screen.bottom() - self.overlay.height() - 80,
        )
        if self.chk_top.isChecked():
            self.overlay.setWindowFlags(
                self.overlay.windowFlags() | Qt.WindowStaysOnTopHint
            )
        self.overlay.show()

    def _hide_overlay(self):
        if self.overlay:
            self.overlay.hide()

    def toggle_transcription(self):
        recogn_type = self.recogn_combo.currentData()
        if recogn_type == LIVE_SHERPA_RECOGN and not models_ready():
            self._ensure_models()
            self._show_download_progress(tr("Please wait"))
            return
        if not self.transcribing:
            if recogn_type != LIVE_SHERPA_RECOGN:
                if recogn_type == recognition.Faster_Whisper_XXL and not self._show_xxl_select():
                    return
                if recogn_type == recognition.Whisper_CPP and not self._show_cpp_select():
                    return
                if recognition.is_input_api(recogn_type=recogn_type) is not True:
                    return
                langcode = translator.get_audio_code(
                    show_source=self.recogn_lang.currentText()
                )
                allow = recognition.is_allow_lang(
                    langcode=langcode,
                    recogn_type=recogn_type,
                    model_name=self.recogn_model.currentText(),
                )
                if allow is not True:
                    return tools.show_error(str(allow))
            if self.chk_translate.isChecked():
                _, target_code = self._get_lang_codes()[1:]
                if self.target_lang.currentText() == "-":
                    return tools.show_error(tr("fanyimoshi1"))
                rs = translator.is_allow_translate(
                    translate_type=self.translate_type.currentIndex(),
                    show_target=target_code,
                )
                if rs is not True:
                    return
            self._save_params()
            proxy = params.get("proxy", "") or app_cfg.proxy
            if proxy:
                app_cfg.proxy = proxy
                tools.set_proxy(proxy)
            self.status_label.setText(tr("Please wait"))
            if recogn_type != LIVE_SHERPA_RECOGN:
                self._show_download_progress(tr("Downloading please wait"))
            self._session_start = time.time()
            self._last_segment_end_ms = 0
            self._last_segment_time = self._session_start
            capture = self.combo.currentData()
            if not capture:
                return tools.show_error(
                    tr("No system audio device found")
                    if (self.source_mode.currentData() or "mic") == "system"
                    else tr("No valid microphone exists")
                )
            if recogn_type == LIVE_SHERPA_RECOGN:
                self.worker = Worker(capture_device=capture, record_dir=LIVE_CAPTIONS_DIR)
            else:
                langcode = translator.get_audio_code(
                    show_source=self.recogn_lang.currentText()
                )
                cache_folder = f"{TEMP_DIR}/live_captions/{time.strftime('%Y%m%d_%H%M%S')}"
                self.worker = ChunkedRecognWorker(
                    capture_device=capture,
                    recogn_type=recogn_type,
                    model_name=self.recogn_model.currentText(),
                    detect_language=langcode,
                    is_cuda=self.chk_cuda.isChecked(),
                    chunk_sec=self.chunk_spin.value(),
                    cache_folder=cache_folder,
                    record_dir=LIVE_CAPTIONS_DIR,
                )
            self.worker.new_word.connect(self._on_partial)
            self.worker.new_segment.connect(self._on_segment)
            self.worker.ready.connect(self._on_ready)
            self.worker.error.connect(self._on_capture_error)
            if hasattr(self.worker, "progress"):
                self.worker.progress.connect(self._download_callback)
            self.worker.start()
            self._show_overlay()
            self.start_button.setText(tr("Stop live captions"))
            self.transcribing = True
        else:
            self._stop_transcription()

    def _stop_transcription(self):
        self._stop_worker()
        if self._trans_worker and self._trans_worker.isRunning():
            self._trans_worker.wait(3000)
            self._trans_worker = None
        self._pending_trans = None
        self._hide_overlay()
        self.start_button.setText(tr("Start live captions"))
        self.transcribing = False
        self.status_label.setText(tr("Stopped"))

    def _on_ready(self):
        if self.download_log.isVisible() and not self._model_downloading:
            QTimer.singleShot(1500, self.download_log.hide)
        if self.recogn_combo.currentData() == LIVE_SHERPA_RECOGN:
            self.status_label.setText(tr("Please speak"))
        else:
            self.status_label.setText(
                tr("Chunk recognition status").format(int(self.chunk_spin.value()))
            )

    def _on_capture_error(self, msg: str):
        self.status_label.setText(msg[:200])
        if self.transcribing:
            self._stop_transcription()
        QMessageBox.critical(self, tr("Error"), msg)

    def _on_partial(self, text: str):
        if self.overlay:
            self.overlay.update_partial(text)

    def _on_segment(self, text: str):
        now = time.time()
        now_ms = int((now - self._session_start) * 1000)
        merge_sec = float(params.get("live_caption_trans_merge_sec", 0))

        if (
            merge_sec > 0
            and self._merge_buffer
            and (now - self._last_segment_time) < merge_sec
        ):
            self._merge_buffer = f"{self._merge_buffer} {text}".strip()
            text = self._merge_buffer
            start_ms = self._merge_buffer_start_ms
        else:
            start_ms = self._last_segment_end_ms
            self._merge_buffer = text
            self._merge_buffer_start_ms = start_ms

        end_ms = max(start_ms + 500, now_ms)
        self._line_counter += 1
        seg = CaptionSegment(
            line=self._line_counter,
            start_ms=start_ms,
            end_ms=end_ms,
            source=text,
        )
        self.segments.append(seg)
        self._last_segment_end_ms = end_ms
        self._last_segment_time = now
        self._merge_buffer = ""

        if self.overlay:
            self.overlay.update_committed(text)

        if self.chk_translate.isChecked():
            self._queue_translate(len(self.segments) - 1, text)

    def _queue_translate(self, idx: int, text: str):
        if self._trans_worker and self._trans_worker.isRunning():
            self._pending_trans = (idx, text)
            return
        self._start_translate(idx, text)

    def _start_translate(self, idx: int, text: str):
        translate_type, source_code, target_code = self._get_lang_codes()
        self._trans_worker = TranslateSegmentWorker(
            idx, text, translate_type, source_code, target_code, self
        )
        self._trans_worker.done.connect(self._on_translated)
        self._trans_worker.error.connect(self._on_translate_error)
        self._trans_worker.finished.connect(self._on_translate_finished)
        self._trans_worker.start()

    def _on_translated(self, idx: int, translated: str):
        if 0 <= idx < len(self.segments):
            self.segments[idx].target = translated
            seg = self.segments[idx]
            if self.overlay and idx == len(self.segments) - 1:
                self.overlay.set_bilingual(self.chk_bilingual.isChecked())
                self.overlay.update_committed(seg.source, seg.target)

    def _on_translate_error(self, msg: str):
        self.status_label.setText(msg[:200])

    def _on_translate_finished(self):
        self._trans_worker = None
        if self._pending_trans:
            idx, text = self._pending_trans
            self._pending_trans = None
            self._start_translate(idx, text)

    def export_srt(self):
        if not self.segments:
            return tools.show_error(tr("No result, no need to save"))
        use_target = self.chk_translate.isChecked()
        content = segments_to_srt(self.segments, use_target=use_target)
        file_name, _ = QFileDialog.getSaveFileName(
            self, tr("Export SRT"), "", "Subtitles files(*.srt)"
        )
        if file_name:
            if not file_name.endswith(".srt"):
                file_name += ".srt"
            Path(file_name).write_text(content, encoding="utf-8")

    def export_txt(self):
        if not self.segments:
            return
        lines = []
        for seg in self.segments:
            if self.chk_translate.isChecked() and seg.target:
                if self.chk_bilingual.isChecked():
                    lines.append(seg.source)
                    lines.append(seg.target)
                else:
                    lines.append(seg.target)
            else:
                lines.append(seg.source)
        text = "\n".join(lines)
        file_name, _ = QFileDialog.getSaveFileName(
            self, tr("Export to TXT"), "", "Text files (*.txt)"
        )
        if file_name:
            if not file_name.endswith(".txt"):
                file_name += ".txt"
            Path(file_name).write_text(text, encoding="utf-8")

    def copy_text(self):
        lines = []
        for seg in self.segments:
            if self.chk_translate.isChecked() and seg.target:
                lines.append(
                    f"{seg.source}\n{seg.target}"
                    if self.chk_bilingual.isChecked()
                    else seg.target
                )
            else:
                lines.append(seg.source)
        QApplication.clipboard().setText("\n".join(lines))

    def clear_session(self):
        self.segments.clear()
        self._line_counter = 0
        self._last_segment_end_ms = 0
        if self.overlay:
            self.overlay.update_committed("", "")

    def closeEvent(self, event: QCloseEvent):
        if self.transcribing:
            self._stop_transcription()
        self._save_params()
        super().closeEvent(event)
