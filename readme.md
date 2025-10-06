# 🎙️ YouTube Transcription & Channel Monitor

Automate **YouTube video transcription and summarization** using **Whisper**, **PyTorch**, **Hugging Face**, and **yt-dlp**.  
Supports **single video transcription** or **continuous channel monitoring** with GPU acceleration.

---

## 🚀 Features

- 🎧 **Automatic transcription** of YouTube videos via OpenAI Whisper.
- 🧠 **Summarization** using Hugging Face Transformers (`facebook/bart-large-cnn`) or Ollama models.
- ⚡ **GPU acceleration** (CUDA) for fast inference.
- 🔁 **Channel monitoring mode** — detect new uploads and process automatically.
- 🧩 **Automatic save system** for transcriptions and summaries.
- 🧱 **Modular structure** — reusable functions and clean architecture.

---
## 🧩 Scripts Overview

### 1. `monitor.py`
Core script that can:
- Transcribe a **single video**
- Monitor a **channel** continuously or in **one-shot mode**

**Main Functions:**
- `download_audio(url)` → downloads audio via `yt-dlp` and converts to `.mp3`
- `transcribe_audio(file)` → uses Whisper to generate text
- `summarize_text(text)` → summarizes long text using BART
- `get_latest_videos(channel_url)` → fetches latest valid videos from a channel
- `monitor_channel(channel_url)` → polls for new videos and auto-transcribes them

---

## Setup Instructions (Linux / WSL)

### Create a virtual environment
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### Upgrade pip
```bash
pip install --upgrade pip

```
### Install PyTorch with CUDA 12.1
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

```
⚠️ If you don’t have an NVIDIA GPU or CUDA installed, you can install CPU-only versions:
```bash
pip install torch torchvision torchaudio

```
### Install project dependencies
```bash
pip install -r requirements.txt
sudo apt install ffmpeg -y

ffmpeg -version
ffprobe -version
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

```
### Usage Examples

#### Transcribe a single video
```bash
python monitor.py --video "https://www.youtube.com/watch?v=abcd123xyz"

```
Result:
Transcription saved in transcriptions/transcription_<videoid>_<timestamp>.md

Summary: created only if you pass the `--summarize` flag. To create a summary along with the transcription, run:
```bash
python monitor.py --video "https://www.youtube.com/watch?v=abcd123xyz" --summarize
```

#### Monitor a Youtube Channel
Continuously check for new videos and transcribe them automatically.
```bash
python monitor.py --monitor "https://www.youtube.com/@ProfessorBellei"
```
You can also adjust polling interval:
```bash
python monitor.py --monitor "https://www.youtube.com/@ProfessorBellei" --poll-interval 900
```
(default = 600 seconds)

### One-shot mode (check once and exit)
```bash
python monitor.py --monitor "https://www.youtube.com/@ProfessorBellei" --once
```


### Using Ollama for summarization (optional)

You can use an Ollama model to perform summarization instead of the default Hugging Face BART pipeline. Pass the `--ollama-model` flag with the Ollama model name (for example: `gpt-oss:latest`). If this flag is not provided the script falls back to the built-in `facebook/bart-large-cnn` summarizer.

Example (one-shot monitor using an Ollama model):
```bash
python monitor.py --monitor "https://www.youtube.com/@ProfessorBellei" --once --summarize --ollama-model "gpt-oss:latest"
```

Notes:
- Summaries are produced only when `--summarize` is provided. By default the script only downloads and transcribes videos.
- The default summarizer (when `--summarize` is used but not `--ollama-model`) is Hugging Face `facebook/bart-large-cnn`.
- To use Ollama you must have the Ollama service available (locally or reachable) and the Python `ollama` package installed in your environment. Pass the model identifier (for example `gpt-oss:latest`) via `--ollama-model`.



### Notes

Make sure to use Python 3.12 for compatibility with PyTorch CUDA builds.
On Windows, WSL2 is recommended for GPU support.
BART model fine-tuned on CNN Daily Mail is effective for:
* Text summarization
* Translation
* Comprehension tasks (e.g., text classification, QA)

### References
* [OpenAI Whisper](https://github.com/openai/whisper)
* [Hugging Face BART Model - fine-tuned on CNN Daily Mail](https://huggingface.co/facebook/bart-large-cnn)
* [PyTorch CUDA](https://pytorch.org/get-started/locally/)
* [yt-dlp](https://github.com/yt-dlp/yt-dlp)