# YouTube Transcription Project

This project uses Whisper, PyTorch, Hugging Face and other libraries to transcribe and summarize YouTube videos.  
To fully leverage GPU acceleration, it's recommended to run on **Linux or WSL** with **CUDA support**.

---

## Setup Instructions (Linux / WSL)

### 1. Create a virtual environment
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 2. Upgrade pip
```bash
pip install --upgrade pip

```
### 3. Install PyTorch with CUDA 12.1
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

```
⚠️ If you don’t have an NVIDIA GPU or CUDA installed, you can install CPU-only versions:
```bash
pip install torch torchvision torchaudio

```
### 4. Install project dependencies
```bash
pip install -r requirements.txt
sudo apt install ffmpeg -y

ffmpeg -version
ffprobe -version

```
### 5. Quick Test
```python
import torch

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

```

### Notes

Make sure to use Python 3.12 for compatibility with PyTorch CUDA builds.

WSL2 is recommended on Windows for GPU support.