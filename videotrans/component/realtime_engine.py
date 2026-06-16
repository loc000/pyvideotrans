import json
import queue
import sys
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import onnxruntime
import sherpa_onnx
import sounddevice as sd
from PySide6.QtCore import QThread, Signal

from videotrans.configure.config import ROOT_DIR, HOME_DIR, TEMP_DIR, tr, logger
from videotrans.util import tools
from videotrans.recognition.live_model_session import LiveModelSession
from videotrans.recognition.model_assets import execution_mode, ExecutionMode

# Live captions: sherpa streaming entry (not in RECOGN_NAME_LIST)
LIVE_SHERPA_RECOGN = -1

CTC_MODEL_FILE = f"{ROOT_DIR}/models/onnx/ctc.model.onnx"
PAR_ENCODER = f"{ROOT_DIR}/models/onnx/encoder.onnx"
PAR_DECODER = f"{ROOT_DIR}/models/onnx/decoder.onnx"
PAR_TOKENS = f"{ROOT_DIR}/models/onnx/tokens.txt"

REALTIME_MODEL_URL = (
    "https://modelscope.cn/models/himyworld/videotrans/resolve/master/realtimestt.zip"
)

SYSTEM_DEVICE_KEYWORDS = (
    "loopback",
    "stereo mix",
    "stereomix",
    "what u hear",
    "monitor",
    "blackhole",
    "vb-audio",
    "vb audio",
    "cable output",
    "wave out",
    "wasapi",
)


@dataclass
class CaptureDevice:
    kind: str  # mic | sd_input | wasapi_loopback
    index: int
    name: str
    sample_rate: int
    channels: int
    display_name: str = ""

    def __post_init__(self):
        if not self.display_name:
            prefix = "[System] " if self.kind != "mic" else ""
            self.display_name = f"{prefix}{self.name}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CaptureDevice":
        return cls(
            kind=data["kind"],
            index=int(data["index"]),
            name=data["name"],
            sample_rate=int(data.get("sample_rate", 48000)),
            channels=int(data.get("channels", 1)),
            display_name=data.get("display_name", ""),
        )


def models_ready() -> bool:
    return (
        Path(PAR_ENCODER).exists()
        and Path(CTC_MODEL_FILE).exists()
        and Path(PAR_DECODER).exists()
    )


def _pyaudiowpatch_available() -> bool:
    if sys.platform != "win32":
        return False
    try:
        import pyaudiowpatch  # noqa: F401
        return True
    except ImportError:
        return False


def get_wasapi_loopback_device() -> Optional[CaptureDevice]:
    if not _pyaudiowpatch_available():
        return None
    import pyaudiowpatch as pyaudio

    pa = pyaudio.PyAudio()
    try:
        info = pa.get_default_wasapi_loopback()
        if not info:
            return None
        channels = max(1, int(info.get("maxInputChannels", 2)))
        rate = int(info.get("defaultSampleRate", 48000))
        return CaptureDevice(
            kind="wasapi_loopback",
            index=int(info["index"]),
            name=info.get("name", "WASAPI Loopback"),
            sample_rate=rate,
            channels=channels,
            display_name=f"[System] {info.get('name', 'WASAPI Loopback')}",
        )
    except Exception:
        return None
    finally:
        pa.terminate()


def enumerate_mic_devices() -> List[CaptureDevice]:
    result = []
    try:
        all_devices = sd.query_devices()
    except Exception:
        return result
    default_input = sd.default.device[0] if sd.default.device else None
    for i, d in enumerate(all_devices):
        if d.get("max_input_channels", 0) < 1:
            continue
        ch = int(d["max_input_channels"])
        rate = int(d.get("default_samplerate") or 48000)
        result.append(
            CaptureDevice(
                kind="mic",
                index=i,
                name=d["name"],
                sample_rate=rate,
                channels=min(ch, 2),
            )
        )
    if result and default_input is not None:
        result.sort(key=lambda x: 0 if x.index == default_input else 1)
    return result


def enumerate_system_devices() -> List[CaptureDevice]:
    result: List[CaptureDevice] = []
    seen_names = set()

    wasapi = get_wasapi_loopback_device()
    if wasapi:
        result.append(wasapi)
        seen_names.add(wasapi.name.lower())

    try:
        all_devices = sd.query_devices()
    except Exception:
        return result

    for i, d in enumerate(all_devices):
        if d.get("max_input_channels", 0) < 1:
            continue
        name_lower = d["name"].lower()
        if not any(kw in name_lower for kw in SYSTEM_DEVICE_KEYWORDS):
            continue
        if name_lower in seen_names:
            continue
        seen_names.add(name_lower)
        ch = int(d["max_input_channels"])
        rate = int(d.get("default_samplerate") or 48000)
        result.append(
            CaptureDevice(
                kind="sd_input",
                index=i,
                name=d["name"],
                sample_rate=rate,
                channels=min(ch, 2),
            )
        )
    return result


def enumerate_devices(source_mode: str) -> List[CaptureDevice]:
    if source_mode == "system":
        return enumerate_system_devices()
    return enumerate_mic_devices()


def _to_mono_float32(samples: np.ndarray, channels: int) -> np.ndarray:
    arr = np.asarray(samples, dtype=np.float32)
    if channels > 1:
        if arr.ndim == 1:
            n = len(arr) // channels
            if n < 1:
                return arr.reshape(-1)
            arr = arr[: n * channels].reshape(-1, channels)
        arr = arr.mean(axis=1)
    return arr.reshape(-1)


class _SounddeviceCapture:
    def __init__(self, device: CaptureDevice):
        self.device = device
        self.sample_rate = device.sample_rate
        self.channels = device.channels
        ch = device.channels if device.kind == "sd_input" else min(device.channels, 2)
        self._stream = sd.InputStream(
            device=device.index,
            channels=ch,
            dtype="float32",
            samplerate=self.sample_rate,
        )

    def start(self):
        self._stream.start()

    def read(self, frames: int) -> np.ndarray:
        data, _ = self._stream.read(frames)
        return _to_mono_float32(data, self.channels)

    def stop(self):
        self._stream.stop()
        self._stream.close()


class _WasapiLoopbackCapture:
    def __init__(self, device: CaptureDevice):
        import pyaudiowpatch as pyaudio

        self.device = device
        self.sample_rate = device.sample_rate
        self.channels = device.channels
        self._pa = pyaudio.PyAudio()
        self._frames = max(int(0.1 * device.sample_rate), 256)
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=device.channels,
            rate=device.sample_rate,
            input=True,
            frames_per_buffer=self._frames,
            input_device_index=device.index,
        )

    def start(self):
        pass

    def read(self, frames: int) -> np.ndarray:
        raw = self._stream.read(frames, exception_on_overflow=False)
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return _to_mono_float32(arr, self.channels)

    def stop(self):
        try:
            self._stream.stop_stream()
            self._stream.close()
        except Exception:
            pass
        try:
            self._pa.terminate()
        except Exception:
            pass


def open_capture_stream(device: CaptureDevice):
    if device.kind == "wasapi_loopback":
        return _WasapiLoopbackCapture(device)
    return _SounddeviceCapture(device)


class OnnxModel:
    def __init__(self):
        session_opts = onnxruntime.SessionOptions()
        session_opts.log_severity_level = 3
        self.sess = onnxruntime.InferenceSession(CTC_MODEL_FILE, session_opts)
        self._init_punct()
        self._init_tokens()

    def _init_punct(self):
        meta = self.sess.get_modelmeta().custom_metadata_map
        punct = meta["punctuations"].split("|")
        self.id2punct = punct
        self.punct2id = {p: i for i, p in enumerate(punct)}
        self.dot = self.punct2id["。"]
        self.comma = self.punct2id["，"]
        self.pause = self.punct2id["、"]
        self.quest = self.punct2id["？"]
        self.underscore = self.punct2id["_"]

    def _init_tokens(self):
        meta = self.sess.get_modelmeta().custom_metadata_map
        tokens = meta["tokens"].split("|")
        self.id2token = tokens
        self.token2id = {t: i for i, t in enumerate(tokens)}
        unk = meta["unk_symbol"]
        assert unk in self.token2id, unk
        self.unk_id = self.token2id[unk]

    def __call__(self, text: str) -> str:
        word_list = text.split()
        words = []
        for w in word_list:
            s = ""
            for c in w:
                if len(c.encode()) > 1:
                    if s == "":
                        s = c
                    elif len(s[-1].encode()) > 1:
                        s += c
                    else:
                        words.append(s)
                        s = c
                else:
                    if s == "":
                        s = c
                    elif len(s[-1].encode()) > 1:
                        words.append(s)
                        s = c
                    else:
                        s += c
            if s:
                words.append(s)

        ids = []
        for w in words:
            if len(w[0].encode()) > 1:
                for c in w:
                    ids.append(self.token2id.get(c, self.unk_id))
            else:
                ids.append(self.token2id.get(w, self.unk_id))

        segment_size = 30
        num_segments = (len(ids) + segment_size - 1) // segment_size
        punctuations = []
        max_len = 200
        last = -1
        for i in range(num_segments):
            this_start = i * segment_size
            this_end = min(this_start + segment_size, len(ids))
            if last != -1:
                this_start = last
            inputs = ids[this_start:this_end]
            out = self.sess.run(
                [self.sess.get_outputs()[0].name],
                {
                    self.sess.get_inputs()[0].name: np.array(
                        inputs, dtype=np.int32
                    ).reshape(1, -1),
                    self.sess.get_inputs()[1].name: np.array(
                        [len(inputs)], dtype=np.int32
                    ),
                },
            )[0]
            out = out[0].argmax(axis=-1).tolist()
            dot_index = -1
            comma_index = -1
            for k in range(len(out) - 1, 1, -1):
                if out[k] in (self.dot, self.quest):
                    dot_index = k
                    break
                if comma_index == -1 and out[k] == self.comma:
                    comma_index = k
            if dot_index == -1 and len(inputs) >= max_len and comma_index != -1:
                dot_index = comma_index
                out[dot_index] = self.dot
            if dot_index == -1:
                if last == -1:
                    last = this_start
                if i == num_segments - 1:
                    dot_index = len(inputs) - 1
            else:
                last = this_start + dot_index + 1
            if dot_index != -1:
                punctuations += out[: dot_index + 1]

        ans = []
        for i, p in enumerate(punctuations):
            t = self.id2token[ids[i]]
            if ans and len(ans[-1][0].encode()) == 1 and len(t[0].encode()) == 1:
                ans.append(" ")
            ans.append(t)
            if p != self.underscore:
                ans.append(self.id2punct[p])
        return "".join(ans)


def create_recognizer():
    return sherpa_onnx.OnlineRecognizer.from_paraformer(
        tokens=PAR_TOKENS,
        encoder=PAR_ENCODER,
        decoder=PAR_DECODER,
        num_threads=4,
        sample_rate=16000,
        feature_dim=80,
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=1.2,
        rule3_min_utterance_length=20,
    )


class Worker(QThread):
    new_word = Signal(str)
    new_segment = Signal(str)
    ready = Signal()
    error = Signal(str)

    def __init__(
        self,
        device_idx=None,
        capture_device: Union[CaptureDevice, dict, None] = None,
        record_dir=None,
        parent=None,
    ):
        super().__init__(parent)
        self.record_dir = record_dir or f"{HOME_DIR}/realtime_stt"
        self.running = False
        if capture_device is not None:
            if isinstance(capture_device, dict):
                self.capture_device = CaptureDevice.from_dict(capture_device)
            else:
                self.capture_device = capture_device
        elif device_idx is not None:
            devs = enumerate_mic_devices()
            match = next((d for d in devs if d.index == device_idx), None)
            if match:
                self.capture_device = match
            else:
                self.capture_device = CaptureDevice(
                    kind="mic",
                    index=int(device_idx),
                    name="Microphone",
                    sample_rate=48000,
                    channels=1,
                )
        else:
            self.capture_device = None

    def run(self):
        if self.capture_device is None:
            self.error.emit("No capture device")
            return

        punct_model = OnnxModel()
        recognizer = create_recognizer()
        stream = recognizer.create_stream()
        sample_rate = self.capture_device.sample_rate
        samples_per_read = max(int(0.1 * sample_rate), 256)

        try:
            capture = open_capture_stream(self.capture_device)
        except Exception as e:
            self.error.emit(str(e))
            return

        wav_path = self.record_dir
        Path(wav_path).mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H-%M-%S")
        txt_file = open(f"{wav_path}/{timestamp}.txt", "a", encoding="utf-8")
        wav_file = wave.open(f"{wav_path}/{timestamp}.wav", "wb")
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        try:
            capture.start()
            self.ready.emit()
            self.running = True
            last_result = ""
            while self.running:
                samples = capture.read(samples_per_read)
                if samples.size < 1:
                    continue
                samples_int16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
                wav_file.writeframes(samples_int16.tobytes())
                stream.accept_waveform(sample_rate, samples)
                while recognizer.is_ready(stream):
                    recognizer.decode_stream(stream)
                is_endpoint = recognizer.is_endpoint(stream)
                result = recognizer.get_result(stream)
                if result != last_result:
                    self.new_word.emit(result)
                    last_result = result
                if is_endpoint:
                    if result:
                        punctuated = punct_model(result)
                        txt_file.write(punctuated + "\n")
                        txt_file.flush()
                        self.new_segment.emit(punctuated)
                    recognizer.reset(stream)
                    last_result = ""
        except Exception as e:
            self.error.emit(str(e))
        finally:
            wav_file.close()
            txt_file.close()
            capture.stop()


def _write_mono_wav(path: str, samples: np.ndarray, sample_rate: int):
    samples_int16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples_int16.tobytes())


def prefetch_recogn_model(
    recogn_type: int,
    model_name: str,
    callback=None,
    *,
    detect_language: str = "auto",
) -> None:
    """Download local ASR model files before live chunked recognition."""
    from videotrans.recognition.model_assets import ensure_assets

    ensure_assets(
        recogn_type,
        model_name,
        detect_language=detect_language,
        callback=callback,
    )


def _resample_to_16k_mono(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    import librosa

    audio = samples.astype(np.float32)
    if sample_rate == 16000:
        return audio
    return librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)


class _ChunkRecognizeThread(QThread):
    """Dequeue audio chunks and run ASR without blocking capture."""

    def __init__(self, owner: "ChunkedRecognWorker"):
        super().__init__()
        self._owner = owner

    def run(self):
        owner = self._owner
        while owner.running or not owner._work_queue.empty():
            try:
                item = owner._work_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                owner._transcribe_chunk(*item)
            except Exception as e:
                logger.exception(f"[live] recognize chunk failed: {e}", exc_info=True)
                owner.error.emit(str(e))
            finally:
                owner._work_queue.task_done()


class ChunkedRecognWorker(QThread):
    """Rolling audio chunks: capture thread enqueues; recognize thread transcribes."""

    new_word = Signal(str)
    new_segment = Signal(str)
    ready = Signal()
    error = Signal(str)
    progress = Signal(str)

    def __init__(
        self,
        capture_device: Union[CaptureDevice, dict],
        recogn_type: int,
        model_name: str,
        detect_language: str,
        is_cuda: bool = False,
        chunk_sec: float = 4.0,
        cache_folder: str = None,
        record_dir=None,
        parent=None,
    ):
        super().__init__(parent)
        if isinstance(capture_device, dict):
            self.capture_device = CaptureDevice.from_dict(capture_device)
        else:
            self.capture_device = capture_device
        self.recogn_type = int(recogn_type)
        self.model_name = model_name or ""
        self.detect_language = detect_language or "auto"
        self.is_cuda = is_cuda
        self.chunk_sec = max(2.0, float(chunk_sec))
        self.cache_folder = cache_folder or f"{TEMP_DIR}/live_captions_chunks"
        self.record_dir = record_dir or f"{HOME_DIR}/live_captions"
        from videotrans.recognition._constants import LIVE_CAPTIONS_UUID_PREFIX

        self.recogn_uuid = f"{LIVE_CAPTIONS_UUID_PREFIX}_{time.strftime('%Y%m%d_%H%M%S')}"
        self.running = False
        self._live_session: Optional[LiveModelSession] = None
        self._recogn_thread: Optional[_ChunkRecognizeThread] = None
        self._work_queue: queue.Queue = queue.Queue(maxsize=1)
        self._dropped_chunks = 0

    def _enqueue_chunk(
        self, samples: np.ndarray, sample_rate: int, chunk_idx: int
    ) -> None:
        item = (samples, sample_rate, chunk_idx)
        try:
            self._work_queue.put_nowait(item)
        except queue.Full:
            try:
                self._work_queue.get_nowait()
                self._dropped_chunks += 1
                logger.debug(
                    f"[live] dropped pending chunk (total={self._dropped_chunks})"
                )
            except queue.Empty:
                pass
            try:
                self._work_queue.put_nowait(item)
            except queue.Full:
                self._dropped_chunks += 1

    def _transcribe_chunk(
        self, samples: np.ndarray, sample_rate: int, chunk_idx: int
    ) -> None:
        from videotrans import recognition

        if samples.size < sample_rate * 0.5:
            return
        audio_16k = _resample_to_16k_mono(samples, sample_rate)
        mode = execution_mode(self.recogn_type, live=True)

        if mode == ExecutionMode.INLINE:
            if self._live_session is None:
                self._live_session = LiveModelSession(
                    self.recogn_type,
                    self.model_name,
                    detect_language=self.detect_language,
                    is_cuda=self.is_cuda,
                )
            text = self._live_session.transcribe_chunk(audio_16k)
            if text:
                self.new_segment.emit(text)
            else:
                logger.debug(f"[live] empty inline text chunk={chunk_idx}")
            return

        Path(self.cache_folder).mkdir(parents=True, exist_ok=True)
        wav16_path = f"{self.cache_folder}/chunk_{chunk_idx}_16k.wav"
        _write_mono_wav(wav16_path, audio_16k, 16000)
        raw_subs = recognition.run(
            audio_file=wav16_path,
            cache_folder=self.cache_folder,
            model_name=self.model_name,
            detect_language=self.detect_language,
            recogn_type=self.recogn_type,
            is_cuda=self.is_cuda,
            uuid=self.recogn_uuid,
        )
        if not raw_subs:
            return
        for item in raw_subs:
            text = (item.get("text") or "").strip()
            if text:
                self.new_segment.emit(text)

    def run(self):
        if self.capture_device is None:
            self.error.emit("No capture device")
            return

        try:
            self.progress.emit(tr("Downloading please wait"))
            prefetch_recogn_model(
                self.recogn_type,
                self.model_name,
                detect_language=self.detect_language,
                callback=lambda m: self.progress.emit(str(m)),
            )
            mode = execution_mode(self.recogn_type, live=True)
            if mode == ExecutionMode.INLINE:
                self.progress.emit(tr("Please wait"))
                self._live_session = LiveModelSession(
                    self.recogn_type,
                    self.model_name,
                    detect_language=self.detect_language,
                    is_cuda=self.is_cuda,
                )
                self._live_session.ensure_loaded()
        except Exception as e:
            self.error.emit(str(e))
            return

        sample_rate = self.capture_device.sample_rate
        samples_per_read = max(int(0.1 * sample_rate), 256)

        try:
            capture = open_capture_stream(self.capture_device)
        except Exception as e:
            self.error.emit(str(e))
            return

        Path(self.record_dir).mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H-%M-%S")
        session_wav = wave.open(f"{self.record_dir}/{timestamp}.wav", "wb")
        session_wav.setnchannels(1)
        session_wav.setsampwidth(2)
        session_wav.setframerate(sample_rate)

        buffer_chunks: List[np.ndarray] = []
        chunk_idx = 0
        last_flush = time.time()
        self._recogn_thread = _ChunkRecognizeThread(self)
        self._recogn_thread.start()

        try:
            capture.start()
            self.ready.emit()
            self.running = True
            while self.running:
                samples = capture.read(samples_per_read)
                if samples.size < 1:
                    continue
                samples_int16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(
                    np.int16
                )
                session_wav.writeframes(samples_int16.tobytes())
                buffer_chunks.append(samples.copy())
                now = time.time()
                if now - last_flush >= self.chunk_sec:
                    combined = (
                        np.concatenate(buffer_chunks)
                        if buffer_chunks
                        else np.array([], dtype=np.float32)
                    )
                    buffer_chunks = []
                    last_flush = now
                    self.new_word.emit(tr("Chunk recognizing"))
                    self._enqueue_chunk(combined, sample_rate, chunk_idx)
                    chunk_idx += 1

            if buffer_chunks:
                combined = np.concatenate(buffer_chunks)
                self._enqueue_chunk(combined, sample_rate, chunk_idx)
            self._work_queue.join()
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.running = False
            if self._recogn_thread:
                self._recogn_thread.wait(30000)
                self._recogn_thread = None
            session_wav.close()
            if self._live_session:
                self._live_session.release()
                self._live_session = None
            capture.stop()


class DownloadModel(QThread):
    down = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

    def run(self):
        tools.down_zip(
            f"{ROOT_DIR}/models", REALTIME_MODEL_URL, self._process_callback
        )

    def _process_callback(self, msg):
        self.down.emit(msg)


class CheckAudioDevices(QThread):
    devices = Signal(str)

    def __init__(self, source_mode: str = "mic", parent=None):
        super().__init__(parent)
        self.source_mode = source_mode

    def run(self):
        devs = enumerate_devices(self.source_mode)
        if not devs:
            self.devices.emit(
                json.dumps({"devices": [], "default": 0, "source_mode": self.source_mode, "empty": True})
            )
            return
        payload = {
            "devices": [d.to_dict() for d in devs],
            "default": 0,
            "source_mode": self.source_mode,
            "empty": False,
        }
        self.devices.emit(json.dumps(payload))


class CheckMics(CheckAudioDevices):
    """Backward compatible: microphone-only device scan."""

    def __init__(self, parent=None):
        super().__init__(source_mode="mic", parent=parent)
