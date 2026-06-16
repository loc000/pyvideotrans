from videotrans.recognition.mimo_asr_lang import mimo_asr_audio_tag


class TestMimoAsrAudioTag:
    def test_auto_empty(self):
        assert mimo_asr_audio_tag(None) == ""
        assert mimo_asr_audio_tag("auto") == ""

    def test_chinese(self):
        assert mimo_asr_audio_tag("zh-cn") == "<chinese>"

    def test_english(self):
        assert mimo_asr_audio_tag("en") == "<english>"

    def test_unsupported_returns_empty(self):
        assert mimo_asr_audio_tag("ja") == ""
