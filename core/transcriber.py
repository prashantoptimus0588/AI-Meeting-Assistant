import os
import requests
from pydub import AudioSegment
from groq import Groq

SARVAM_PIECE_SECONDS = 25
GROQ_PIECE_SECONDS = 25  # Groq also has a file size limit (~25MB)

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "whisper-large-v3"

_groq_client = None


def get_groq_client():
    global _groq_client
    if _groq_client is None:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set in environment / .env")
        _groq_client = Groq(api_key=GROQ_API_KEY)
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


def transcribe_chunk_groq(chunk_path: str) -> str:
    """Split chunk into pieces and transcribe via Groq Whisper sequentially."""
    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = GROQ_PIECE_SECONDS * 1000
    total_pieces = (len(audio) + piece_ms - 1) // piece_ms

    full_text = ""

    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece_path = f"{chunk_path}_gr_{i}.wav"
        audio[start: start + piece_ms].export(piece_path, format="wav")
        try:
            print(f"  → Groq piece {i + 1}/{total_pieces} ...")
            full_text += _send_to_groq(piece_path) + " "
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return full_text.strip()


# ── Sarvam ────────────────────────────────────────────────────────────────────

def _send_to_sarvam(piece_path: str) -> str:
    """Send one ≤30s WAV file to Sarvam and return the English transcript."""
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


def transcribe_chunk_sarvam(chunk_path: str) -> str:
    """Split chunk into 25s pieces and transcribe via Sarvam sequentially."""
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")

    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000
    total_pieces = (len(audio) + piece_ms - 1) // piece_ms

    full_text = ""

    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece_path = f"{chunk_path}_sv_{i}.wav"
        audio[start: start + piece_ms].export(piece_path, format="wav")
        try:
            print(f"  → Sarvam piece {i + 1}/{total_pieces} ...")
            full_text += _send_to_sarvam(piece_path) + " "
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return full_text.strip()


# ── Router ────────────────────────────────────────────────────────────────────

def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    """
    Route one chunk based on language:
    - hinglish → Sarvam (translates Hinglish to English)
    - english  → Groq Whisper (fast, online)
    """
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