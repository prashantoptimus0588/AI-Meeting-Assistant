"""
Finds the real threshold where Groq's transcription API starts failing —
tests increasing clip durations sliced from a longer source file, instead
of uploading a huge file directly (which would hit Groq's ~25MB limit).

Run with: python test_groq.py
"""
import os
from dotenv import load_dotenv
from groq import Groq
from pydub import AudioSegment

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("❌ GROQ_API_KEY not found in .env — set it first.")
    exit(1)

print(f"✅ GROQ_API_KEY loaded (starts with: {GROQ_API_KEY[:8]}...)")

client = Groq(api_key=GROQ_API_KEY, timeout=120.0, max_retries=1)

# Your existing 252MB file — used only as a SOURCE to slice small clips
# from. We never upload this file directly; that's what caused the last run
# to hang (way over Groq's ~25MB per-request limit).
SOURCE_FILE = "test_audio.wav"

if not os.path.exists(SOURCE_FILE):
    print(f"❌ {SOURCE_FILE} not found.")
    exit(1)

print(f"Loading source file (this may take a moment for a large file)...")
audio = AudioSegment.from_file(SOURCE_FILE)
duration_s = len(audio) / 1000
print(f"Source duration: {duration_s:.1f}s ({duration_s / 60:.1f} min)")

# Test increasing durations to find where it starts failing.
# Stop as soon as one fails — no point testing longer ones after that.
test_durations = [25, 60, 90, 150, 300, 600]

for seconds in test_durations:
    ms = seconds * 1000
    if ms > len(audio):
        print(f"\n⏭️  Skipping {seconds}s — source is only {duration_s:.1f}s long")
        break

    clip = audio[:ms]
    test_path = f"_test_{seconds}s.wav"
    clip.export(test_path, format="wav")
    size_mb = os.path.getsize(test_path) / (1024 * 1024)

    print(f"\n📤 Testing {seconds}s clip ({size_mb:.1f} MB)...")

    if size_mb > 25:
        print(f"⚠️  This clip is already over Groq's ~25MB limit — expect it to fail or reject.")

    try:
        with open(test_path, "rb") as f:
            response = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=f,
                response_format="text",
            )
        print(f"✅ SUCCESS at {seconds}s ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"❌ FAILED at {seconds}s ({size_mb:.1f} MB): {type(e).__name__}: {e}")
        os.remove(test_path)
        print(f"\n🛑 Stopping — found the breaking point between the last success and {seconds}s.")
        break
    finally:
        if os.path.exists(test_path):
            os.remove(test_path)

print("\nDone.")