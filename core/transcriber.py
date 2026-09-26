import os
import time
import requests
from pydub import AudioSegment
from groq import Groq
from concurrent.futures import ThreadPoolExecutor

SARVAM_PIECE_SECONDS = 60   # reduced for reliability on an unstable connection
GROQ_PIECE_SECONDS = 45     # both 25s and 60s succeeded cleanly — 45s stays well inside that safe range

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "whisper-large-v3"

MAX_TRANSCRIBE_WORKERS = 3  # lowered — fewer simultaneous large transfers competing for bandwidth
MAX_RETRIES_PER_PIECE = 3   # application-level retry, on top of the SDK's own retry

_groq_client = None


def get_groq_client():
    global _groq_client
    if _groq_client is None:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set in environment / .env")
        _groq_client = Groq(
            api_key=GROQ_API_KEY,
            timeout=90.0,
            max_retries=1,  # SDK-level retry; we add our own loop below for network-level drops
        )
    return _groq_client


# ── Groq ──────────────────────────────────────────────────────────────────────

def _send_to_groq(piece_path: str) -> str:
    """Send one audio piece to Groq Whisper and return the transcript."""
    client = get_groq_client()
    with open(piece_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model=GROQ_MODEL,
            file=f,
            response_format="text",
        )
    return response


def _send_to_groq_with_retry(piece_path: str, piece_label: str) -> str:
    """
    Wraps _send_to_groq with manual retries + backoff, since your network
    has shown intermittent stalls even at sizes that previously succeeded —
    a single dropped connection shouldn't kill the whole transcription.
    """
    last_error = None
    for attempt in range(1, MAX_RETRIES_PER_PIECE + 1):
        try:
            return _send_to_groq(piece_path)
        except Exception as e:
            last_error = e
            wait = 2 ** attempt  # 2s, 4s, 8s
            print(f"  ⚠️  {piece_label} failed (attempt {attempt}/{MAX_RETRIES_PER_PIECE}): "
                  f"{type(e).__name__}. Retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"{piece_label} failed after {MAX_RETRIES_PER_PIECE} attempts") from last_error


def _split_into_pieces(chunk_path: str, piece_seconds: int, tag: str) -> list:
    """Export a chunk into numbered piece files and return their paths, in order."""
    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = piece_seconds * 1000
    piece_paths = []

    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece_path = f"{chunk_path}_{tag}_{i}.wav"
        audio[start: start + piece_ms].export(piece_path, format="wav")
        piece_paths.append(piece_path)

    return piece_paths


def transcribe_chunk_groq(chunk_path: str) -> str:
    """Split chunk into pieces and transcribe via Groq Whisper concurrently, with retries."""
    piece_paths = _split_into_pieces(chunk_path, GROQ_PIECE_SECONDS, "gr")
    total_pieces = len(piece_paths)

    def _transcribe_and_cleanup(indexed_piece):
        i, piece_path = indexed_piece
        label = f"Groq piece {i + 1}/{total_pieces}"
        try:
            print(f"  → {label} ...")
            return _send_to_groq_with_retry(piece_path, label)
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    with ThreadPoolExecutor(max_workers=min(MAX_TRANSCRIBE_WORKERS, total_pieces)) as executor:
        results = list(executor.map(_transcribe_and_cleanup, enumerate(piece_paths)))

    return " ".join(results).strip()


# ── Sarvam ────────────────────────────────────────────────────────────────────

def _send_to_sarvam(piece_path: str) -> str:
    """Send one WAV file to Sarvam and return the English transcript."""
    headers = {"api-subscription-key": SARVAM_API_KEY}

    with open(piece_path, "rb") as f:
        files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
        data = {"model": SARVAM_MODEL, "with_diarization": "false"}
        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    if not response.ok:
        print(f"\n❌ Sarvam returned {response.status_code}")
        print(f"Response body: {response.text}\n")
        response.raise_for_status()

    return response.json().get("transcript", "")


def _send_to_sarvam_with_retry(piece_path: str, piece_label: str) -> str:
    last_error = None
    for attempt in range(1, MAX_RETRIES_PER_PIECE + 1):
        try:
            return _send_to_sarvam(piece_path)
        except Exception as e:
            last_error = e
            wait = 2 ** attempt
            print(f"  ⚠️  {piece_label} failed (attempt {attempt}/{MAX_RETRIES_PER_PIECE}): "
                  f"{type(e).__name__}. Retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"{piece_label} failed after {MAX_RETRIES_PER_PIECE} attempts") from last_error


def transcribe_chunk_sarvam(chunk_path: str) -> str:
    """Split chunk into pieces and transcribe via Sarvam concurrently, with retries."""
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")

    piece_paths = _split_into_pieces(chunk_path, SARVAM_PIECE_SECONDS, "sv")
    total_pieces = len(piece_paths)

    def _transcribe_and_cleanup(indexed_piece):
        i, piece_path = indexed_piece
        label = f"Sarvam piece {i + 1}/{total_pieces}"
        try:
            print(f"  → {label} ...")
            return _send_to_sarvam_with_retry(piece_path, label)
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    with ThreadPoolExecutor(max_workers=min(MAX_TRANSCRIBE_WORKERS, total_pieces)) as executor:
        results = list(executor.map(_transcribe_and_cleanup, enumerate(piece_paths)))

    return " ".join(results).strip()


# ── Router ────────────────────────────────────────────────────────────────────

def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path)
    return transcribe_chunk_groq(chunk_path)


def transcribe_all(chunks: list, language: str = "english") -> str:
    full_transcript = ""

    engine = "Sarvam AI" if language.lower() == "hinglish" else "Groq Whisper"
    print(f"Using {engine} for transcription.")

    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")
        text = transcribe_chunk(chunk, language=language)
        full_transcript += text + " "

    print("Transcription complete.")
    return full_transcript.strip()