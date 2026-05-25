"""Shared download and worker lifecycle helpers for realtime STT UIs."""
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QMessageBox, QPlainTextEdit, QPushButton, QWidget

from videotrans.component.realtime_engine import DownloadModel, models_ready
from videotrans.configure.config import tr


class RealtimeUiBase:
    """Mixin: sherpa model download, progress UI, worker stop."""

    _download_task: Optional[DownloadModel] = None
    _model_downloading: bool = False
    worker = None
    transcribing: bool = False

    def _show_download_progress(self, msg: str, log_widget: Optional[QPlainTextEdit] = None):
        if not msg:
            return
        if log_widget is not None:
            log_widget.show()
            log_widget.setPlainText(msg)
            sb = log_widget.verticalScrollBar()
            sb.setValue(sb.maximum())

    def _set_download_ui_busy(
        self,
        busy: bool,
        *,
        check_button: Optional[QPushButton] = None,
        start_button: Optional[QPushButton] = None,
    ):
        self._model_downloading = busy
        if check_button is not None:
            check_button.setDisabled(busy)
        if start_button is not None and not self.transcribing:
            start_button.setDisabled(busy)

    def _ensure_sherpa_models(
        self,
        *,
        log_widget: Optional[QPlainTextEdit] = None,
        check_button: Optional[QPushButton] = None,
        start_button: Optional[QPushButton] = None,
        on_complete: Optional[Callable] = None,
    ):
        if models_ready():
            self._set_download_ui_busy(
                False, check_button=check_button, start_button=start_button
            )
            if on_complete:
                on_complete()
            return
        if self._download_task and self._download_task.isRunning():
            return
        self._set_download_ui_busy(
            True, check_button=check_button, start_button=start_button
        )
        self._show_download_progress(tr("Please wait"), log_widget)
        self._download_task = DownloadModel(self)
        self._download_task.down.connect(
            lambda m: self._on_sherpa_download_msg(
                m, log_widget, check_button, start_button, on_complete
            )
        )
        self._download_task.finished.connect(self._on_sherpa_download_finished)
        self._download_task.start()

    def _on_sherpa_download_finished(self):
        self._download_task = None

    def _on_sherpa_download_msg(
        self,
        msg: str,
        log_widget: Optional[QPlainTextEdit],
        check_button: Optional[QPushButton],
        start_button: Optional[QPushButton],
        on_complete: Optional[Callable],
    ):
        self._show_download_progress(msg, log_widget)
        if msg.startswith("Error:"):
            self._set_download_ui_busy(
                False, check_button=check_button, start_button=start_button
            )
            if check_button is not None:
                check_button.setDisabled(False)
            QMessageBox.critical(self, "Error", msg)
        elif msg.endswith(" end"):
            self._set_download_ui_busy(
                False, check_button=check_button, start_button=start_button
            )
            self._show_download_progress(tr("Model download complete"), log_widget)
            if log_widget is not None:
                QTimer.singleShot(2000, log_widget.hide)
            if on_complete:
                on_complete()

    def _stop_worker(self, worker=None):
        w = worker if worker is not None else self.worker
        if w:
            w.running = False
            w.wait()
        if worker is None:
            self.worker = None

    @staticmethod
    def open_recordings_dir(path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))
