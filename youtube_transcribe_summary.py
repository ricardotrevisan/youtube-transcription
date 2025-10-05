# youtube_transcribe_summary.py
from datetime import datetime
import os
import whisper
from transformers import pipeline
import yt_dlp
import whisper
import torch

print(torch.cuda.is_available())  # True se a GPU estiver pronta
print(torch.cuda.get_device_name(0))  # Nome da GPU

device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("base").to(device)
print(model.device)

def download_audio(url: str, output_base: str = "audio") -> str:
    """
    Baixa o áudio de um vídeo do YouTube e retorna o caminho completo do arquivo MP3.
    """
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_base,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    audio_file = output_base + ".mp3"
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Arquivo de áudio não encontrado: {audio_file}")

    print(f"Áudio baixado em: {audio_file}")
    return audio_file


def get_video_metadata(url: str) -> dict:
    ydl_opts = {'skip_download': True, 'quiet': True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception:
        return {}

    title = info.get('title') or info.get('fulltitle') or ''
    upload_date = info.get('upload_date')
    publish_date = None
    if upload_date and len(str(upload_date)) == 8:
        s = str(upload_date)
        publish_date = f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    webpage_url = info.get('webpage_url') or url
    return {'title': title, 'publish_date': publish_date, 'url': webpage_url}

def transcribe_audio(audio_file: str, model_size: str = "base") -> str:
    """
    Transcreve o áudio usando Whisper.
    """
    model = whisper.load_model(model_size).to(device)
    result = model.transcribe(audio_file, language="pt")
    transcription = result["text"]
    print("Transcrição completa.")
    return transcription

def summarize_text(text: str, max_length: int = 150, min_length: int = 40, **kwargs) -> str:
    """
    Resume o texto usando Hugging Face transformers.
    """
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=max_length, min_length=min_length, **kwargs)
    return summary[0]["summary_text"]


def chunk_text(text, max_chars=1000):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start+max_chars])
        start += max_chars
    return chunks

def save_locally(content: str, raw: bool = False, filename: str = None, title: str = None, publish_date: str = None, url: str = None) -> str:
    os.makedirs("transcriptions", exist_ok=True)
    if raw and not filename:
        filename = "transcriptions/raw.md"
    elif not filename:
        filename = f"transcriptions/transcription_{datetime.now().strftime('%Y%m%d')}.md"

    with open(filename, "w", encoding="utf-8") as f:
        header_lines = []
        if title:
            header_lines.append(f"Title: {title}")
        if publish_date:
            header_lines.append(f"Published: {publish_date}")
        if url:
            header_lines.append(f"URL: {url}")

        if header_lines:
            f.write("\n".join(header_lines) + "\n\n---\n\n")

        f.write(content)

    return filename


if __name__ == "__main__":
    # URL do vídeo
    url = "https://www.youtube.com/watch?v=tCw7BO180aI"
    url = "https://www.youtube.com/watch?v=xWs4r31-O3M"

    # 1. Baixar áudio
    audio_file = download_audio(url)

    # 2. Transcrever áudio
    transcription = transcribe_audio(audio_file)
    meta = get_video_metadata(url)
    save_locally(transcription, raw=True, title=meta.get('title'), publish_date=meta.get('publish_date'), url=meta.get('url'))
    print("Transcrição (primeiros 1000 chars):\n", transcription[:1000])

    # 3. Resumir
    chunks = chunk_text(transcription)

    summaries = [summarize_text(chunk, max_length=150, min_length=50, do_sample=False) for chunk in chunks]
    final_summary = " ".join(summaries)
    
    print("Resumo:\n", final_summary)
    save_locally(final_summary, raw=False, title=meta.get('title'), publish_date=meta.get('publish_date'), url=meta.get('url'))
    
    # 4. Resumir o resumo para um formato mais compacto
    final_chunks = chunk_text(final_summary, max_chars=500)  # reduzir o tamanho para segurança
    final_summaries = [
        summarize_text(chunk, max_length=100, min_length=40, do_sample=False)
        for chunk in final_chunks
    ]
    final_summary_short = " ".join(final_summaries)
    print("\n\n\nResumo final compacto:\n", final_summary_short)