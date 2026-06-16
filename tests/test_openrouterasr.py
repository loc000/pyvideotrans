import base64
import tempfile
from pathlib import Path

from videotrans import recognition
from videotrans.recognition.model_assets import ExecutionMode, execution_mode
from videotrans.recognition._openrouterasr import (
    audio_format_from_path,
    build_transcribe_payload,
    parse_transcript,
)
from videotrans.util.help_http_debug import (
    format_openrouter_stt_debug_block,
    openai_transcription_url,
)


class TestAudioFormat:
    def test_wav(self):
        assert audio_format_from_path("/tmp/chunk.wav") == "wav"

    def test_mp3(self):
        assert audio_format_from_path("/tmp/chunk.mp3") == "mp3"

    def test_unknown_fallback(self):
        assert audio_format_from_path("/tmp/chunk.xyz") == "wav"


class TestBuildPayload:
    def test_payload_structure(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFFfake")
            path = f.name
        try:
            payload = build_transcribe_payload(
                model="qwen/qwen3-asr-flash-2026-02-10",
                audio_path=path,
                language="zh",
            )
            assert payload["model"] == "qwen/qwen3-asr-flash-2026-02-10"
            assert payload["language"] == "zh"
            ia = payload["input_audio"]
            assert ia["format"] == "wav"
            assert base64.b64decode(ia["data"]) == b"RIFFfake"
            assert "messages" not in payload
        finally:
            Path(path).unlink(missing_ok=True)


class TestParseTranscript:
    def test_text_field(self):
        assert parse_transcript({"text": " hello "}) == "hello"

    def test_choices_fallback(self):
        data = {"choices": [{"message": {"content": " hello "}}]}
        assert parse_transcript(data) == "hello"

    def test_empty(self):
        assert parse_transcript({}) == ""


class TestTranscriptionUrl:
    def test_openrouter_url(self):
        url = openai_transcription_url("https://openrouter.ai/api/v1")
        assert url == "https://openrouter.ai/api/v1/audio/transcriptions"


class TestDebugBlock:
    def test_debug_mentions_json_base64(self):
        block = format_openrouter_stt_debug_block(
            base_url="https://openrouter.ai/api/v1",
            api_key="sk-test",
            model="qwen/qwen3-asr-flash-2026-02-10",
            audio_path="/tmp/audio.wav",
            language="zh",
        )
        assert "/audio/transcriptions" in block
        assert "input_audio" in block
        assert "PUT-BASE64-HERE" in block


class TestExecutionMode:
    def test_openrouter_asr_api_mode(self):
        assert execution_mode(recognition.OPENROUTER_ASR) == ExecutionMode.API