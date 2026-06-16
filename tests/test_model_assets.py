from videotrans import recognition
from videotrans.recognition.model_assets import (
    ExecutionMode,
    execution_mode,
    local_dir_for,
    mimo_tokenizer_dir,
    resolve_assets,
    resolve_funasr_model_name,
)


class TestLocalDirFor:
    def test_faster_whisper_tiny(self):
        p = local_dir_for(recognition.FASTER_WHISPER, "tiny")
        assert "models--Systran--faster-whisper-tiny" in str(p).replace("\\", "/")

    def test_qwen_local_dir(self):
        p = local_dir_for(recognition.QWENASR, "1.7B")
        assert "Qwen3-ASR-1.7B" in str(p)

    def test_huggingface_repo(self):
        p = local_dir_for(recognition.HUGGINGFACE_ASR, "nvidia/parakeet-ctc-1.1b")
        assert "nvidia--parakeet-ctc-1.1b" in str(p).replace("\\", "/")

    def test_mimo_asr_local_dir(self):
        p = local_dir_for(
            recognition.MIMO_ASR,
            "LeaderboardModel1/MiMo-V2.5-ASR-AutoRound-W4A16-RTN",
        )
        assert "LeaderboardModel1--MiMo-V2.5-ASR-AutoRound-W4A16-RTN" in str(p).replace(
            "\\", "/"
        )

    def test_mimo_tokenizer_dir(self):
        p = mimo_tokenizer_dir()
        assert "MiMo-Audio-Tokenizer" in str(p).replace("\\", "/")

    def test_nemotron_asr_local_dir(self):
        p = local_dir_for(
            recognition.NEMOTRON_ASR,
            recognition.NEMOTRON_ASR_MODEL,
        )
        assert "nvidia--nemotron-3.5-asr-streaming-0.6b" in str(p).replace("\\", "/")


class TestResolveFunasrModelName:
    def test_paraformer_non_zh_en(self):
        name = resolve_funasr_model_name("paraformer-zh", "fr")
        assert "MLT" in name

    def test_sensevoice_alias(self):
        assert resolve_funasr_model_name("SenseVoiceSmall", "zh") == "iic/SenseVoiceSmall"


class TestResolveAssets:
    def test_qwen_has_backend(self):
        assets = resolve_assets(recognition.QWENASR, "1.7B", detect_language="zh-cn")
        assert len(assets) >= 1
        assert assets[0].backend in ("ms", "hf")

    def test_faster_whisper_hf(self):
        assets = resolve_assets(recognition.FASTER_WHISPER, "tiny")
        assert assets[0].backend == "hf"
        assert assets[0].repo_id

    def test_mimo_dual_hf_assets(self):
        assets = resolve_assets(
            recognition.MIMO_ASR,
            recognition.DEFAULT_MIMO_ASR_MODEL,
        )
        assert len(assets) == 2
        assert all(a.backend == "hf" for a in assets)
        repos = {a.repo_id for a in assets}
        assert recognition.MIMO_AUDIO_TOKENIZER_REPO in repos
        assert recognition.DEFAULT_MIMO_ASR_MODEL in repos

    def test_nemotron_hf_asset(self):
        assets = resolve_assets(
            recognition.NEMOTRON_ASR,
            recognition.NEMOTRON_ASR_MODEL,
        )
        assert len(assets) == 1
        assert assets[0].backend == "hf"
        assert assets[0].repo_id == recognition.NEMOTRON_ASR_MODEL


class TestExecutionMode:
    def test_live_qwen_inline(self):
        assert execution_mode(recognition.QWENASR, live=True) == ExecutionMode.INLINE

    def test_live_faster_inline(self):
        assert (
            execution_mode(recognition.FASTER_WHISPER, live=True)
            == ExecutionMode.INLINE
        )

    def test_batch_subprocess(self):
        assert (
            execution_mode(recognition.FASTER_WHISPER, live=False)
            == ExecutionMode.SUBPROCESS
        )
