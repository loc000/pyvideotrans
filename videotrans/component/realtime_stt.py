import json
from pathlib import Path

from PySide6.QtCore import QTimer, Qt, QUrl
from PySide6.QtGui import QCloseEvent, QDesktopServices, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from videotrans.component.realtime_engine import (
    CaptureDevice,
    CheckAudioDevices,
    Worker,
    models_ready,
)
from videotrans.component.realtime_ui_base import RealtimeUiBase
from videotrans.configure.config import HOME_DIR, ROOT_DIR, tr

REALTIME_STT_DIR = f"{HOME_DIR}/realtime_stt"


class RealTimeWindow(RealtimeUiBase, QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(1000, 500)
        self.setWindowTitle(
            tr("Real-time speech-to-text")
            + " "
            + tr("Only supports Chinese and English language recognition")
        )
        self.setWindowIcon(QIcon(f"{ROOT_DIR}/videotrans/styles/icon.ico"))
        self.layout = QVBoxLayout(self)

        self.mic_layout = QHBoxLayout()
        self.combo = QComboBox()
        self.combo.setMinimumWidth(250)
        self.checkbtn = QPushButton()
        self.checkbtn.setText(tr("Detection microphone"))
        self.checkbtn.clicked.connect(self.populate_mics)
        self.mic_layout.addWidget(self.combo)
        self.mic_layout.addWidget(self.checkbtn)

        self.start_button = QPushButton(tr("Initiating real-time transcription"))
        self.start_button.setCursor(Qt.PointingHandCursor)
        self.start_button.setMinimumHeight(30)
        self.start_button.setMinimumWidth(150)
        self.start_button.clicked.connect(self.toggle_transcription)
        self.mic_layout.addWidget(self.start_button)
        self.mic_layout.addStretch()
        self.layout.addLayout(self.mic_layout)

        self.realtime_text = QPlainTextEdit()
        self.realtime_text.setReadOnly(True)
        self.realtime_text.setStyleSheet("background: transparent; border: none;")
        self.realtime_text.setMaximumHeight(80)
        self.layout.addWidget(self.realtime_text)

        self.textedit = QPlainTextEdit()
        self.textedit.setReadOnly(True)
        self.textedit.setMinimumHeight(400)
        self.textedit.setStyleSheet("color:#ffffff")
        self.layout.addWidget(self.textedit)

        self.button_layout = QHBoxLayout()
        self.export_button = QPushButton(tr("Export to TXT"))
        self.export_button.clicked.connect(self.export_txt)
        self.export_button.setCursor(Qt.PointingHandCursor)
        self.export_button.setMinimumHeight(35)
        self.button_layout.addWidget(self.export_button)

        self.copy_button = QPushButton(tr("Copy"))
        self.copy_button.setMinimumHeight(35)
        self.copy_button.setCursor(Qt.PointingHandCursor)
        self.copy_button.clicked.connect(self.copy_textedit)
        self.button_layout.addWidget(self.copy_button)

        self.clear_button = QPushButton(tr("Clear"))
        self.clear_button.setMinimumHeight(35)
        self.clear_button.setCursor(Qt.PointingHandCursor)
        self.clear_button.clicked.connect(self.clear_textedit)
        self.button_layout.addWidget(self.clear_button)
        self.layout.addLayout(self.button_layout)

        self.btn_opendir = QPushButton(
            f"{tr('Recording files are stored in')}: {REALTIME_STT_DIR}"
        )
        self.btn_opendir.setStyleSheet(
            "background-color:transparent;border:0;color:#ddd"
        )
        self.btn_opendir.clicked.connect(self.open_dir)
        self.layout.addWidget(self.btn_opendir)

        self.worker = None
        self.transcribing = False
        QTimer.singleShot(500, self.populate_mics)

    def _process_callback(self, msg):
        self.realtime_text.setPlainText(msg)
        if msg.startswith("Error:") or msg.endswith(" end"):
            self.start_button.setDisabled(False)
        if msg.startswith("Error:"):
            QMessageBox.critical(self, "Error", msg)

    def open_dir(self):
        self.open_recordings_dir(REALTIME_STT_DIR)

    def populate_mics(self):
        def _get_dev(data):
            self.checkbtn.setDisabled(False)
            payload = json.loads(data) if isinstance(data, str) else data
            if payload.get("empty"):
                self.combo.clear()
                self.checkbtn.setText(tr("No valid microphone exists"))
                return
            self.combo.clear()
            for d in payload["devices"]:
                dev = CaptureDevice.from_dict(d)
                self.combo.addItem(dev.display_name, dev.to_dict())
            self.combo.setCurrentIndex(payload.get("default", 0))

        self.checkbtn.setDisabled(True)
        task = CheckAudioDevices(source_mode="mic", parent=self)
        task.devices.connect(_get_dev)
        task.start()
        self._ensure_sherpa_models(
            check_button=self.checkbtn,
            start_button=self.start_button,
            on_complete=lambda: self.checkbtn.setDisabled(False),
        )

    def toggle_transcription(self):
        if not models_ready():
            self._ensure_sherpa_models(
                check_button=self.checkbtn,
                start_button=self.start_button,
            )
            return
        if not self.transcribing:
            self.realtime_text.setPlainText(tr("Please wait"))
            capture = self.combo.currentData()
            if not capture:
                return
            self.worker = Worker(
                capture_device=capture, record_dir=REALTIME_STT_DIR
            )
            self.worker.new_word.connect(self.update_realtime)
            self.worker.new_segment.connect(self.append_segment)
            self.worker.ready.connect(self.update_realtime_ready)
            self.worker.start()
            self.start_button.setText(tr("Real-time transcription"))
            self.transcribing = True
        else:
            self._stop_worker()
            self.start_button.setText(tr("Initiating real-time transcription"))
            self.transcribing = False
            remaining_text = self.realtime_text.toPlainText().strip()
            if remaining_text:
                self.textedit.appendPlainText(remaining_text)
                scrollbar = self.textedit.verticalScrollBar()
                scrollbar.setValue(scrollbar.maximum())
            self.realtime_text.clear()

    def update_realtime(self, text):
        self.realtime_text.setPlainText(text)
        scrollbar = self.realtime_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_realtime_ready(self):
        self.realtime_text.setPlainText(tr("Please speak"))

    def append_segment(self, text):
        self.textedit.appendPlainText(text)
        scrollbar = self.textedit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def export_txt(self):
        text = self.textedit.toPlainText().strip()
        if not text:
            return
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save TXT", "", "Text files (*.txt)"
        )
        if file_name:
            if not file_name.endswith(".txt"):
                file_name += ".txt"
            with open(file_name, "w", encoding="utf-8") as f:
                f.write(text)

    def copy_textedit(self):
        text = self.textedit.toPlainText()
        QApplication.clipboard().setText(text)

    def clear_textedit(self):
        self.textedit.clear()

    def closeEvent(self, event: QCloseEvent):
        if self.transcribing:
            self.toggle_transcription()
        super().closeEvent(event)
