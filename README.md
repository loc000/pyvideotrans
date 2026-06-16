
> Sponsors: **[Recall.ai](https://www.recall.ai/product/meeting-transcription-api?utm_source=github&utm_medium=sponsorship&utm_campaign=jianchang512-pyvideotrans) - Meeting Transcription API**
>
> If you’re looking for a transcription API for meetings, consider checking out **[Recall.ai](https://www.recall.ai/product/meeting-transcription-api?utm_source=github&utm_medium=sponsorship&utm_campaign=jianchang512-pyvideotrans)** , an API that works with Zoom, Google Meet, Microsoft Teams, and more




# pyVideoTrans 

<div align="center">

**A Powerful Open Source Video Translation / Audio Transcription / AI Dubbing / Subtitle Translation Tool**

[中文](docs/README_CN.md) | [**Documentation**](https://pyvideotrans.com) | [**Online Q&A**](https://bbs.pyvideotrans.com)

[![License](https://img.shields.io/badge/License-GPL_v3-blue.svg)](LICENSE)   [![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)   [![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)]()

</div>

**pyVideoTrans** is dedicated to seamlessly converting videos from one language to another, offering a complete workflow that includes speech recognition, subtitle translation, multi-role dubbing, and audio-video synchronization. It supports both local offline deployment and a wide variety of mainstream online APIs.

<img width="1566" height="912" alt="image" src="https://github.com/user-attachments/assets/7410b17d-9903-4919-954a-31764e246c15" />


---

## 🔀 This Fork vs Upstream

This repository is a fork of [jianchang512/pyvideotrans](https://github.com/jianchang512/pyvideotrans). It keeps the upstream workflow (ASR → translation → TTS → mux) but adds recognition and realtime features that are not in the original repo yet.

| Area | Upstream | This fork (`loc000/pyvideotrans`) |
| :--- | :--- | :--- |
| **Live captions** | Basic realtime STT UI | Full **live captions** window with bilingual overlay, configurable font/opacity, and shared realtime engine |
| **Realtime ASR** | Per-channel ad-hoc logic | Unified **`realtime_engine`** + **`model_assets`** registry for batch and streaming ASR |
| **ASR channels** | Standard upstream set | Adds **OpenRouter** (dedicated `openrouter_asr_*` settings), **MiMo-V2.5-ASR (Local)**, and **Nemotron-3.5-ASR Streaming (Local)** |
| **OpenRouter STT** | Via generic OpenAI Speech-to-Text settings | Separate **OpenRouter ASR** channel calling `POST /audio/transcriptions` with `input_audio` (independent from translation `openrouter_key`) |
| **API troubleshooting** | Basic error messages | **HTTP debug blocks** (request URL, model, redacted curl) on failed OpenAI-compatible translation/STT calls |
| **Tests** | Upstream coverage | Extra unit tests for OpenRouter ASR, MiMo ASR, and Nemotron ASR language/model helpers |

**Clone this fork:**

```bash
git clone git@github.com:loc000/pyvideotrans.git
cd pyvideotrans
uv sync
```

**Sync with upstream** (optional):

```bash
git remote add upstream https://github.com/jianchang512/pyvideotrans.git
git fetch upstream
git merge upstream/main
```

> **API keys:** `videotrans/params.json` and `videotrans/cfg.json` are gitignored. Configure keys in the GUI (Menu → Keys) after cloning; they are never committed to this repo.

See [docs/architecture.md](docs/architecture.md) for implementation details on live captions, MiMo ASR, and the realtime pipeline.

---

## ✨ Core Features

> [Technical Architecture and Principles](docs/architecture.md)

- **🎥 Fully Automatic Video Translation**: One-click workflow: Speech Recognition (ASR) -> Subtitle Translation -> Speech Synthesis (TTS) -> Video Synthesis.
- **🎙️ Audio Transcription / Subtitle Generation**: Batch convert audio/video to SRT subtitles, supporting **Speaker Diarization** to distinguish between different roles.
- **🗣️ Multi-Role AI Dubbing**: Assign different AI dubbing voices to different speakers.
- **🧬 Voice Cloning**: Integrates models like **F5-TTS, CosyVoice, GPT-SoVITS** for zero-shot voice cloning.
- **🧠 Powerful Model Support**: 
  - **ASR**: Faster-Whisper (Local), OpenAI Whisper, Alibaba Qwen, ByteDance Volcano, Azure, Google, **OpenRouter**, **MiMo-V2.5-ASR**, **Nemotron streaming ASR**, etc.
  - **LLM Translation**: DeepSeek, ChatGPT, Claude, Gemini, MiniMax, Ollama (Local), Alibaba Bailian, etc.
  - **TTS**: Edge-TTS (Free), OpenAI, Azure, Minimaxi, ChatTTS, ChatterBox, etc.
- **🖥️ Interactive Editing**: Supports pausing and manual proofreading at each stage (recognition, translation, dubbing) to ensure accuracy.
- **🛠️ Utility Toolkit**: Includes auxiliary tools such as vocal separation, video/subtitle merging, audio-video alignment, and transcript matching.
- **💻 Command Line Interface (CLI)**: Supports headless operation, convenient for server deployment or batch processing.

<img width="2752" height="1536" alt="unnamed" src="https://github.com/user-attachments/assets/960e9e34-84a4-425d-b582-f726623475a8" />

---

## 🚀 Quick Start (Windows Users)

We provide a pre-packaged `.exe` version for Windows 10/11 users, requiring no Python environment configuration.

1. **Download**: [Click to download the latest pre-packaged version](https://github.com/jianchang512/pyvideotrans/releases)
2. **Unzip**: Extract the compressed file to a path (e.g., `D:\pyVideoTrans`).
3. **Run**: Double-click `sp.exe` inside the folder to launch.

> **Note**: 
> *   Do not run directly from within the compressed archive.
> *   To use GPU acceleration, ensure **CUDA 12.8** and **cuDNN 9.11** are installed.

---

## 🛠️ Source Deployment (macOS / Linux / Windows Developers)

We recommend using **[`uv`](https://docs.astral.sh/uv/)** for package management for faster speed and better environment isolation.

### 1. Prerequisites

*   **Python**: Recommended version 3.10 --> 3.12
*   **FFmpeg**: Must be installed and configured in the environment variables.
    *   **macOS**: `brew install ffmpeg libsndfile git`
    *   **Linux (Ubuntu/Debian)**: `sudo apt-get install ffmpeg libsndfile1-dev`
    *   **Windows**: [Download FFmpeg](https://ffmpeg.org/download.html) and configure Path, or place `ffmpeg.exe` and `ffprobe.exe` directly in the project directory.

### 2. Install uv (If not installed)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 3. Clone and Install

```bash
# 1. Clone the repository (Ensure path has no spaces/Chinese characters)
#    Fork (this repo):
git clone git@github.com:loc000/pyvideotrans.git
#    Upstream original:
# git clone https://github.com/jianchang512/pyvideotrans.git
cd pyvideotrans

# 2. Install dependencies (uv automatically syncs environment)
uv sync

# If you need local channels for qwen-tts and qwen-asr, please execute `uv sync --extra qwen-tts --extra qwen-asr`

```

### 4. Launch Software

**Launch GUI**:
```bash
uv run sp.py
```

**Use CLI**:

> [View documentation for detailed parameters](https://pyvideotrans.com/cli)

```bash
# Video Translation Example
uv run cli.py --task vtv --name "./video.mp4" --source_language_code zh --target_language_code en

# Audio to Subtitle Example
uv run cli.py --task stt --name "./audio.wav" --model_name large-v3
```

### 5. (Optional) GPU Acceleration Configuration

If you have an NVIDIA graphics card, execute the following commands to install the CUDA-supported PyTorch version:

```bash
# Uninstall CPU version
uv remove torch torchaudio

# Install CUDA version (Example for CUDA 12.x)
uv add torch==2.7 torchaudio==2.7 --index-url https://download.pytorch.org/whl/cu128
uv add nvidia-cublas-cu12 nvidia-cudnn-cu12
```

---

## 🧩 Supported Channels & Models (Partial)

| Category | Channel/Model | Description |
| :--- | :--- | :--- |
| **ASR (Speech Recognition)** | **Faster-Whisper** (Local) | Recommended, fast speed, high accuracy |
| | WhisperX / Parakeet | Supports timestamp alignment & speaker diarization |
| | Alibaba Qwen3-ASR / ByteDance Volcano | Online API, excellent for Chinese |
| | **OpenRouter** (fork) | Dedicated ASR channel; `openrouter_asr_*` settings, `/audio/transcriptions` |
| | **MiMo-V2.5-ASR** (fork, local) | Xiaomi MiMo local ASR with bundled tokenizer |
| | **Nemotron-3.5-ASR Streaming** (fork, local) | NVIDIA Nemotron streaming ASR for live/batch use |
| | **Live captions** (fork) | Real-time subtitles with optional translation overlay |
| **Translation (LLM/MT)** | **DeepSeek** / ChatGPT | Supports context understanding, more natural translation |
| | MiniMax AI | MiniMax M2.7 LLM, latest flagship model, OpenAI-compatible |
| | Google / Microsoft | Traditional machine translation, fast speed |
| | Ollama / M2M100 | Fully local offline translation |
| **TTS (Speech Synthesis)** | **Edge-TTS** | Microsoft free interface, natural effect |
| | **F5-TTS / CosyVoice** | Supports **Voice Cloning**, requires local deployment |
| | GPT-SoVITS / ChatTTS | High-quality open-source TTS |
| | 302.AI / OpenAI / Azure | High-quality commercial API |

---

## 📚 Documentation & Support

*   **Official Documentation**: [https://pyvideotrans.com](https://pyvideotrans.com) (Includes detailed tutorials, API configuration guides, FAQ)
*   **Online Q&A Community**: [https://bbs.pyvideotrans.com](https://bbs.pyvideotrans.com) (Submit error logs for automated AI analysis and answers)

## ⚠️ Disclaimer

This software is an open-source, free, non-commercial project. Users are solely responsible for any legal consequences arising from the use of this software (including but not limited to calling third-party APIs or processing copyrighted video content). Please comply with local laws and regulations and the terms of use of relevant service providers.

## 🙏 Acknowledgements

This project mainly relies on the following open-source projects (partial):

*   [FFmpeg](https://github.com/FFmpeg/FFmpeg)
*   [PySide6](https://pypi.org/project/PySide6/)
*   [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
*   [openai-whisper](https://github.com/openai/whisper)
*   [edge-tts](https://github.com/rany2/edge-tts)
*   [F5-TTS](https://github.com/SWivid/F5-TTS)
*   [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)

---

*Fork maintained by [loc000](https://github.com/loc000) · Based on [jianchang512/pyvideotrans](https://github.com/jianchang512/pyvideotrans)*
