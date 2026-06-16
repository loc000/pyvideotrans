from videotrans.recognition.nemotron_asr_lang import nemotron_target_lang


class TestNemotronTargetLang:
    def test_auto_empty(self):
        assert nemotron_target_lang(None) == "auto"
        assert nemotron_target_lang("auto") == "auto"

    def test_english(self):
        assert nemotron_target_lang("en") == "en-US"
        assert nemotron_target_lang("en-gb") == "en-GB"

    def test_chinese(self):
        assert nemotron_target_lang("zh-cn") == "zh-CN"

    def test_unknown_maps_auto(self):
        assert nemotron_target_lang("xx") == "auto"
