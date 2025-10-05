# youtube_transcribe_summary.py
from datetime import datetime, timedelta
import os
import whisper
from transformers import pipeline
import yt_dlp
import whisper
import torch
import time
import json
import argparse
import ollama
from urllib.parse import urlparse, parse_qs

print("CUDA available:", torch.cuda.is_available())
try:
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))  # Nome da GPU
except Exception:
    # ignore if no device name is available
    pass

# Define device string for model loading elsewhere
device = "cuda" if torch.cuda.is_available() else "cpu"

def download_audio(url: str, output_base: str = "audio") -> str:
    """
    Baixa o áudio de um vídeo do YouTube e retorna o caminho completo do arquivo MP3.
    """
    # ensure yt-dlp writes a temporary file and ffmpeg postprocessor will create .mp3
    ydl_opts = {
        'format': 'bestaudio/best',
        # include extension placeholder so yt-dlp handles extensions predictably
        'outtmpl': f"{output_base}.%(ext)s",
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

def transcribe_audio(audio_file: str, model_size: str = "base") -> str:
    """
    Transcreve o áudio usando Whisper.
    """
    model = whisper.load_model(model_size).to(device)
    result = model.transcribe(audio_file, language="pt")
    transcription = result["text"]
    print("Transcrição completa.")
    return transcription


def transcribe_with_model(model, audio_file: str) -> str:
    """Transcribe using an already-loaded Whisper model instance (avoids reloading)."""
    result = model.transcribe(audio_file, language="pt")
    transcription = result["text"]
    print("Transcrição completa (reuso de modelo).")
    return transcription

def summarize_text(text: str, max_length: int = 150, min_length: int = 40, use_ollama: bool = False, ollama_model: str = None, **kwargs) -> str:
    """
    Resume o texto usando Hugging Face transformers ou Ollama se especificado.
    """
    if use_ollama:
        print("Using Ollama for summarization")
        # allow caller to specify the Ollama model name; fall back to previous default
        chosen_model = ollama_model or "gpt-oss:latest"
        response = ollama.generate(model=chosen_model, prompt=f"Please summarize the following text + bullet points with main ideas:\n\n{text}")
        return response['response']
    else:
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

def save_locally(content: str, raw: bool = False, filename: str = None, video_id: str = None, title: str = None, publish_date: str = None, url: str = None) -> str:
    """Save transcription content into `transcriptions/`.

    - If filename is provided, use it.
    - Else if video_id is provided, create `transcriptions/transcription_<videoid>_<timestamp>.md`.
    - Else if raw=True write to `transcriptions/raw.md` (backwards compatibility).
    - Else create `transcriptions/transcription_<date>_<timestamp>.md`.
    Returns the filename written.
    """
    os.makedirs("transcriptions", exist_ok=True)
    os.makedirs("summaries", exist_ok=True)
    if filename:
        out = filename
    elif video_id:
        out = f"transcriptions/transcription_{video_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    elif raw:
        out = "transcriptions/raw.md"
    else:
        out = f"transcriptions/transcription_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    with open(out, "w", encoding="utf-8") as f:
        # write metadata header if present
        header_lines = []
        if title:
            header_lines.append(f"Title: {title}")
        if publish_date:
            header_lines.append(f"Published: {publish_date}")
        if url:
            header_lines.append(f"URL: {url}")
        if video_id:
            header_lines.append(f"Video ID: {video_id}")

        if header_lines:
            f.write("\n".join(header_lines) + "\n\n---\n\n")

        f.write(content)
    return out

def extract_video_id(url: str) -> str | None:
    """Try to extract a YouTube video id from a URL's query string (v=) or path.

    Returns None if not found.
    """
    try:
        p = urlparse(url)
        qs = parse_qs(p.query)
        if 'v' in qs and qs['v']:
            return qs['v'][0]
        # handle youtu.be short links
        if p.netloc in ('youtu.be', 'www.youtu.be') and p.path:
            return p.path.lstrip('/')
        # fallback: if path contains /shorts/<id>
        parts = p.path.strip('/').split('/')
        if len(parts) >= 2 and parts[0] == 'shorts':
            return parts[1]
    except Exception:
        return None
    return None

def get_video_metadata(url: str) -> dict:
    """Extract basic metadata (title, publish_date, url) for a YouTube video using yt_dlp.

    Returns a dict with keys: title (str), publish_date (YYYY-MM-DD or None), url (str).
    """
    ydl_opts = {'skip_download': True, 'quiet': True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception:
        return {}

    title = info.get('title') or info.get('fulltitle') or ''
    upload_date = info.get('upload_date')  # often YYYYMMDD
    publish_date = None
    if upload_date and len(str(upload_date)) == 8:
        s = str(upload_date)
        publish_date = f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    elif info.get('release_timestamp'):
        try:
            publish_date = datetime.utcfromtimestamp(int(info.get('release_timestamp'))).strftime('%Y-%m-%d')
        except Exception:
            publish_date = None

    webpage_url = info.get('webpage_url') or url
    return {'title': title, 'publish_date': publish_date, 'url': webpage_url}

def resolve_channel_id(channel_url: str) -> str | None:
    """Try to resolve a channel handle or URL to a canonical channel ID (UC...)

    Returns None if resolution fails.
    """
    # quick heuristics
    if not channel_url:
        return None
    # if the URL already contains /channel/<id>
    try:
        p = urlparse(channel_url)
        parts = p.path.strip('/').split('/')
        if len(parts) >= 2 and parts[0] == 'channel':
            return parts[1]
        # if it's a handle @something or /user/ or /c/, try yt_dlp info extraction
    except Exception:
        pass

    ydl_opts = {'skip_download': True, 'quiet': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(channel_url, download=False)
            # info may include 'channel_id' or 'uploader_id'
            for key in ('channel_id', 'uploader_id', 'id'):
                if key in info and isinstance(info[key], str) and info[key].startswith('UC'):
                    return info[key]
        except Exception:
            return None
    return None

def get_latest_videos(channel_url: str, max_results: int = 10):
    """
    Retorna os vídeos recentes válidos de um canal do YouTube.
    Cada item contém: id, title, url.
    """

    # Força o yt-dlp a resolver vídeos reais (sem playlists, shorts, etc.)
    ydl_opts = {
        'extract_flat': True,
        'skip_download': True,
        'quiet': True,
    }

    # Garante que a URL aponte para /videos (a aba de vídeos do canal)
    if '/videos' not in channel_url:
        channel_url = channel_url.rstrip('/') + '/videos'

    videos = []
    print(f"Fetching videos from: {channel_url}")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(channel_url, download=False)

        entries = info.get('entries', [])
        for e in entries[:max_results]:
            video_id = e.get('id')
            if not video_id or len(video_id) != 11:
                continue

            title = e.get('title') or e.get('fulltitle') or ''
            url = f"https://www.youtube.com/watch?v={video_id}"

            # Try to extract a publish date from the flat entry if available.
            publish_date = None
            upload_date = e.get('upload_date') or e.get('uploader_date')
            if upload_date and len(str(upload_date)) == 8:
                s = str(upload_date)
                publish_date = f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
                print(f"Parsed publish date {publish_date} from upload_date {upload_date}")
            else:
                # try timestamp-like fields
                for ts_key in ('release_timestamp', 'timestamp'):
                    if e.get(ts_key):
                        try:
                            publish_date = datetime.utcfromtimestamp(int(e.get(ts_key))).strftime('%Y-%m-%d')
                            print(f"Parsed publish date {publish_date} from {ts_key} {e.get(ts_key)}")
                            break
                        except Exception:
                            pass

            # If we couldn't determine a publish_date from the flat entry, try fetching
            # the full video metadata (slower) as a fallback so we can apply the 2-day cutoff.
            if not publish_date:
                try:
                    meta_full = get_video_metadata(url)
                    if meta_full and meta_full.get('publish_date'):
                        publish_date = meta_full.get('publish_date')
                        print(f"Fallback publish date from full metadata: {publish_date} for {video_id}")
                except Exception:
                    pass

            videos.append({
                'id': video_id,
                'title': title,
                'url': url,
                'publish_date': publish_date,
            })

    except Exception as e:
        print(f"[ERRO] Falha ao extrair vídeos: {e}")

    # Diagnóstico básico se nada for retornado
    if not videos:
        print(f"Nenhum vídeo encontrado em {channel_url}")

    return videos

def _seen_filepath():
    os.makedirs('transcriptions', exist_ok=True)
    return os.path.join('transcriptions', 'seen_videos.json')

def load_seen():
    path = _seen_filepath()
    if not os.path.exists(path):
        return set()
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return set(data)
    except Exception:
        return set()

def save_seen(seen_set):
    path = _seen_filepath()
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(list(seen_set), f, ensure_ascii=False, indent=2)

def monitor_channel(channel_url: str, poll_interval: int = 600, model_size: str = 'base', ollama_model: str = None):
    """Continuously poll a channel for new videos and transcribe them.

    - channel_url: something yt_dlp accepts (channel page, handle, or uploads playlist)
    - poll_interval: seconds between polls
    """
    seen = load_seen()
    print(f"Loaded {len(seen)} seen video ids")

    # load model once and reuse
    model = whisper.load_model(model_size).to(device)
    
    while True:
        try:
            videos = get_latest_videos(channel_url, max_results=5)
            # filter out videos older than 2 days
            cutoff = datetime.utcnow().date() - timedelta(days=2)
            recent_videos = []
            for v in videos:
                pd = v.get('publish_date')
                if pd:
                    try:
                        pd_date = datetime.strptime(pd, '%Y-%m-%d').date()
                        if pd_date < cutoff:
                            # skip old videos
                            continue
                    except Exception:
                        # if parsing fails, keep the video (be permissive)
                        pass
                recent_videos.append(v)

            new_videos = [v for v in recent_videos if v['id'] not in seen]
            if new_videos:
                print(f"Found {len(new_videos)} new video(s)")
            # process oldest-first
            for v in reversed(new_videos):
                print(f"Processing new video: {v['title']} ({v['url']})")
                try:
                    vid = v.get('id') or extract_video_id(v.get('url', ''))
                    base = f"audio_{vid}" if vid else f"audio_{int(time.time())}"
                    audio_file = download_audio(v['url'], output_base=base)
                    transcription = transcribe_with_model(model, audio_file)
                    meta = get_video_metadata(v.get('url') or v.get('webpage_url') or '')
                    save_locally(transcription, video_id=vid, title=meta.get('title'), publish_date=meta.get('publish_date'), url=meta.get('url'))
                    # Create and save a summary for the transcription (same behavior as --once)
                    try:
                        if ollama_model:
                            final_summary = summarize_text(transcription, max_length=1500, min_length=50, do_sample=False, use_ollama=True, ollama_model=ollama_model)
                        else:
                            chunks = chunk_text(transcription)
                            summaries = [summarize_text(chunk, max_length=150, min_length=50, do_sample=False) for chunk in chunks]
                            final_summary = " ".join(summaries)
                        summary_filename = f"summaries/summary_transcription_{vid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                        save_locally(final_summary, filename=summary_filename, raw=False, video_id=vid, title=meta.get('title'), publish_date=meta.get('publish_date'), url=meta.get('url'))
                    except Exception as se:
                        print(f"Failed to summarize {vid}: {se}")
                    if vid:
                        seen.add(vid)
                        save_seen(seen)
                except Exception as e:
                    print(f"Failed to process {v['id']}: {e}")
        except Exception as e:
            print(f"Monitor error: {e}")

        time.sleep(poll_interval)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="YouTube channel monitor and transcriber")
    parser.add_argument('--monitor', nargs='?', const='https://www.youtube.com/@ProfessorBellei', help='Monitor a channel URL or handle for new uploads (default handle used if no value).')
    parser.add_argument('--poll-interval', type=int, default=600, help='Seconds between channel polls (default 600).')
    parser.add_argument('--model-size', default='base', help='Whisper model size to load (default: base).')
    parser.add_argument('--once', action='store_true', help='When used with --monitor do a single pass then exit.')
    parser.add_argument('--video', help='Transcribe a single video URL and exit.')
    parser.add_argument('--ollama-model', help='Use Ollama for summarization and specify the model name (e.g. "gpt-oss:latest"). If set, Ollama will be used for summaries.', default=None)

    args = parser.parse_args()

    if args.video:
        print(f"Transcribing single video: {args.video}")
        vid = extract_video_id(args.video)
        base = f"audio_{vid}" if vid else 'audio_single'
        audio_file = download_audio(args.video, output_base=base)
        transcription = transcribe_audio(audio_file, model_size=args.model_size)
        save_locally(transcription, video_id=vid)
    elif args.monitor:
        channel = args.monitor
        if args.once:
            # one-shot: check and process any new videos then exit
            seen = load_seen()
            model_local = whisper.load_model(args.model_size).to(device)
            videos = get_latest_videos(channel, max_results=5)
            cutoff = datetime.utcnow().date() - timedelta(days=2)
            recent_videos = []
            for v in videos:
                pd = v.get('publish_date')
                print(f"Video {v.get('title')} publish_date: {pd}")
                if pd:
                    try:
                        pd_date = datetime.strptime(pd, '%Y-%m-%d').date()
                        if pd_date < cutoff:
                            continue
                    except Exception:
                        pass
                recent_videos.append(v)

            new_videos = [v for v in recent_videos if v.get('id') not in seen]
            if not new_videos:
                print("No new videos found.")
            for v in reversed(new_videos):
                print(f"Processing: {v.get('title')} ({v.get('url')})")
                try:
                    vid = v.get('id') or extract_video_id(v.get('url', ''))
                    base = f"audio_{vid}" if vid else f"audio_{int(time.time())}"
                    audio_file = download_audio(v.get('url'), output_base=base)
                    transcription = transcribe_with_model(model_local, audio_file)
                    meta = get_video_metadata(v.get('url') or v.get('webpage_url') or '')
                    save_locally(transcription, video_id=vid, title=meta.get('title'), publish_date=meta.get('publish_date'), url=meta.get('url'))
                    if vid:
                        seen.add(vid)
                    if args.ollama_model:
                        final_summary = summarize_text(transcription, max_length=1500, min_length=50, do_sample=False, use_ollama=True, ollama_model=args.ollama_model)
                    else:
                        chunks = chunk_text(transcription)
                        summaries = [summarize_text(chunk, max_length=150, min_length=50, do_sample=False) for chunk in chunks]
                        final_summary = " ".join(summaries)
                    summary_filename = f"summaries/summary_transcription_{vid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                    save_locally(final_summary, filename=summary_filename, raw=False, video_id=vid, title=meta.get('title'), publish_date=meta.get('publish_date'), url=meta.get('url'))    
                except Exception as e:
                    print(f"Failed to process {v.get('id')}: {e}")
            save_seen(seen)
            print("One-shot monitoring complete.")
        else:
            print(f"Starting continuous monitor for {args.monitor} (poll every {args.poll_interval}s)")
            try:
                monitor_channel(args.monitor, poll_interval=args.poll_interval, model_size=args.model_size, ollama_model=args.ollama_model)
            except KeyboardInterrupt:
                print("Monitor stopped by user.")
    else:
        parser.print_help()