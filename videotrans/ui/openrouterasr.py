from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt

from videotrans.configure.config import tr, params, settings
from videotrans.util import tools


class Ui_openrouterasrform(object):
    def setupUi(self, openrouterasrform):
        self.has_done = False
        openrouterasrform.setObjectName("openrouterasrform")
        openrouterasrform.setWindowModality(QtCore.Qt.NonModal)
        openrouterasrform.resize(600, 500)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(openrouterasrform.sizePolicy().hasHeightForWidth())
        openrouterasrform.setSizePolicy(sizePolicy)
        openrouterasrform.setMaximumSize(QtCore.QSize(600, 600))

        self.label_0 = QtWidgets.QLabel(openrouterasrform)
        self.label_0.setGeometry(QtCore.QRect(10, 10, 580, 35))
        self.label_0.setText(tr("OpenRouter ASR uses audio/transcriptions with input_audio"))

        self.label = QtWidgets.QLabel(openrouterasrform)
        self.label.setGeometry(QtCore.QRect(10, 45, 130, 35))
        self.label.setMinimumSize(QtCore.QSize(0, 35))
        self.label.setObjectName("label")
        self.openrouter_asr_url = QtWidgets.QLineEdit(openrouterasrform)
        self.openrouter_asr_url.setGeometry(QtCore.QRect(150, 45, 431, 35))
        self.openrouter_asr_url.setMinimumSize(QtCore.QSize(0, 35))
        self.openrouter_asr_url.setObjectName("openrouter_asr_url")

        self.label_2 = QtWidgets.QLabel(openrouterasrform)
        self.label_2.setGeometry(QtCore.QRect(10, 95, 130, 35))
        self.label_2.setMinimumSize(QtCore.QSize(0, 35))
        self.label_2.setSizeIncrement(QtCore.QSize(0, 35))
        self.label_2.setObjectName("label_2")
        self.openrouter_asr_key = QtWidgets.QLineEdit(openrouterasrform)
        self.openrouter_asr_key.setGeometry(QtCore.QRect(150, 95, 431, 35))
        self.openrouter_asr_key.setMinimumSize(QtCore.QSize(0, 35))
        self.openrouter_asr_key.setObjectName("openrouter_asr_key")

        self.label_prompt = QtWidgets.QLabel(openrouterasrform)
        self.label_prompt.setGeometry(QtCore.QRect(10, 140, 130, 35))
        self.label_prompt.setMinimumSize(QtCore.QSize(0, 35))
        self.label_prompt.setSizeIncrement(QtCore.QSize(0, 35))
        self.label_prompt.setObjectName("label_prompt")
        self.openrouter_asr_prompt = QtWidgets.QLineEdit(openrouterasrform)
        self.openrouter_asr_prompt.setGeometry(QtCore.QRect(150, 140, 431, 35))
        self.openrouter_asr_prompt.setMinimumSize(QtCore.QSize(0, 35))
        self.openrouter_asr_prompt.setObjectName("openrouter_asr_prompt")

        self.label_3 = QtWidgets.QLabel(openrouterasrform)
        self.label_3.setGeometry(QtCore.QRect(10, 190, 121, 16))
        self.label_3.setObjectName("label_3")
        self.openrouter_asr_model = QtWidgets.QComboBox(openrouterasrform)
        self.openrouter_asr_model.setGeometry(QtCore.QRect(150, 185, 431, 35))
        self.openrouter_asr_model.setMinimumSize(QtCore.QSize(0, 35))
        self.openrouter_asr_model.setObjectName("openrouter_asr_model")

        self.label_allmodels = QtWidgets.QLabel(openrouterasrform)
        self.label_allmodels.setGeometry(QtCore.QRect(10, 220, 571, 21))
        self.label_allmodels.setObjectName("label_allmodels")
        self.label_allmodels.setText(
            tr("Fill in all available models, separated by commas. After filling in, you can select them above"))

        self.edit_allmodels = QtWidgets.QPlainTextEdit(openrouterasrform)
        self.edit_allmodels.setGeometry(QtCore.QRect(10, 250, 571, 100))
        self.edit_allmodels.setObjectName("edit_allmodels")

        self.set_openrouterasr = QtWidgets.QPushButton(openrouterasrform)
        self.set_openrouterasr.setGeometry(QtCore.QRect(10, 410, 93, 35))
        self.set_openrouterasr.setMinimumSize(QtCore.QSize(0, 35))
        self.set_openrouterasr.setObjectName("set_openrouterasr")

        self.test_openrouterasr = QtWidgets.QPushButton(openrouterasrform)
        self.test_openrouterasr.setGeometry(QtCore.QRect(130, 415, 93, 30))
        self.test_openrouterasr.setMinimumSize(QtCore.QSize(0, 30))
        self.test_openrouterasr.setObjectName("test_openrouterasr")

        help_btn = QtWidgets.QPushButton(openrouterasrform)
        help_btn.setGeometry(QtCore.QRect(250, 415, 120, 30))
        help_btn.setStyleSheet("background-color: rgba(255, 255, 255,0)")
        help_btn.setObjectName("help_btn")
        help_btn.setCursor(Qt.PointingHandCursor)
        help_btn.setText(tr("Fill out the tutorial"))
        help_btn.clicked.connect(
            lambda: tools.open_url(
                url='https://openrouter.ai/docs/guides/overview/multimodal/audio'
            )
        )

        self.retranslateUi(openrouterasrform)
        QtCore.QMetaObject.connectSlotsByName(openrouterasrform)

    def update_ui(self):
        allmodels_str = settings.get('openrouter_asr_model', '')
        allmodels = [x.strip() for x in str(allmodels_str).split(',') if x.strip()]
        self.openrouter_asr_model.clear()
        self.openrouter_asr_model.addItems(allmodels)
        self.edit_allmodels.setPlainText(allmodels_str)

        self.openrouter_asr_key.setText(params.get("openrouter_asr_key", ''))
        self.openrouter_asr_prompt.setText(params.get("openrouter_asr_prompt", ''))
        self.openrouter_asr_url.setText(
            params.get("openrouter_asr_url", '') or "https://openrouter.ai/api/v1"
        )
        if params.get('openrouter_asr_model', '') in allmodels:
            self.openrouter_asr_model.setCurrentText(params.get("openrouter_asr_model", ''))

    def retranslateUi(self, openrouterasrform):
        openrouterasrform.setWindowTitle(tr("OpenRouter ASR"))
        self.label_3.setText(tr("Model"))
        self.set_openrouterasr.setText(tr("Save"))
        self.test_openrouterasr.setText(tr("Test"))
        self.openrouter_asr_url.setPlaceholderText("https://openrouter.ai/api/v1")
        self.openrouter_asr_url.setToolTip(tr("OpenRouter or compatible audio/transcriptions API base URL"))
        self.openrouter_asr_key.setPlaceholderText("Secret key")
        self.openrouter_asr_key.setToolTip(tr("OpenRouter API key for speech recognition"))
        self.label.setText(tr("API URL"))
        self.label_2.setText(tr("SK"))
        self.label_prompt.setText(tr("Prompt"))