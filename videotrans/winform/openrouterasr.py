def openwin():
    from PySide6 import QtWidgets
    from videotrans.configure.config import tr, params, settings, app_cfg
    from videotrans.util import tools
    from videotrans import recognition
    from videotrans.util.TestSTT import TestSTT

    def feed(d):
        if d.startswith("ok"):
            QtWidgets.QMessageBox.information(winobj, "ok", d[3:])
        else:
            tools.show_error(d)
        winobj.test_openrouterasr.setText(tr("Test"))

    def test():
        key = winobj.openrouter_asr_key.text()
        prompt = winobj.openrouter_asr_prompt.text()
        url = tools.process_openai_api(winobj.openrouter_asr_url.text().strip())
        model = winobj.openrouter_asr_model.currentText()

        params["openrouter_asr_key"] = key
        params["openrouter_asr_url"] = url
        params["openrouter_asr_model"] = model
        params["openrouter_asr_prompt"] = prompt
        winobj.test_openrouterasr.setText(tr("Testing..."))
        task = TestSTT(
            parent=winobj,
            recogn_type=recognition.OPENROUTER_ASR,
            model_name=model,
        )
        task.uito.connect(feed)
        task.start()

    def save_openrouterasr():
        key = winobj.openrouter_asr_key.text()
        prompt = winobj.openrouter_asr_prompt.text()
        url = tools.process_openai_api(winobj.openrouter_asr_url.text().strip())
        model = winobj.openrouter_asr_model.currentText()

        params["openrouter_asr_key"] = key
        params["openrouter_asr_url"] = url
        params["openrouter_asr_model"] = model
        params["openrouter_asr_prompt"] = prompt
        params.save()
        winobj.close()

    def setallmodels():
        t = winobj.edit_allmodels.toPlainText().strip().replace('，', ',').rstrip(',')
        current_text = winobj.openrouter_asr_model.currentText()
        winobj.openrouter_asr_model.clear()
        winobj.openrouter_asr_model.addItems([x for x in t.split(',') if x.strip()])
        if current_text:
            winobj.openrouter_asr_model.setCurrentText(current_text)
        settings['openrouter_asr_model'] = t
        settings.save()

    from videotrans.component.set_form import OpenRouterASRForm
    winobj = OpenRouterASRForm()
    app_cfg.child_forms['openrouterasr'] = winobj
    winobj.update_ui()
    winobj.set_openrouterasr.clicked.connect(save_openrouterasr)
    winobj.test_openrouterasr.clicked.connect(test)
    winobj.edit_allmodels.textChanged.connect(setallmodels)
    winobj.show()