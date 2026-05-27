from dotenv import load_dotenv
# Load environment variables
load_dotenv()

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)
from core.summarize import summarize, generate_title




# =========================
# INPUT CONFIGURATION
# =========================
SOURCE = "https://youtu.be/2wTRQqoDM1A?si=KdqQAluDOo4HUo8z"
LANGUAGE = "english"


def print_section(title: str, content: str):
    """Helper function to print formatted sections."""
    print("\n" + "=" * 60)
    print(f"{title}")
    print("=" * 60)
    print(content)
    print("=" * 60 + "\n")


def main():
    # =========================
    # AUDIO PROCESSING
    # =========================
    print("Processing input source...")
    chunks = process_input(SOURCE)

    # =========================
    # TRANSCRIPTION
    # =========================
    print("Transcribing audio...")
    transcript = transcribe_all(chunks, language=LANGUAGE)

    preview = (
        transcript[:500] + "..."
        if len(transcript) > 500
        else transcript
    )

    print_section("TRANSCRIPT PREVIEW", preview)

    # =========================
    # NLP EXTRACTION
    # =========================
    print("Generating insights...\n")

    title = generate_title(transcript)
    summary = summarize(transcript)

    questions = extract_questions(transcript)
    decisions = extract_key_decisions(transcript)
    action_items = extract_action_items(transcript)

    # =========================
    # OUTPUT
    # =========================
    print_section("📌 TITLE", title)
    print_section("📝 SUMMARY", summary)
    print_section("❓ QUESTIONS", questions)
    print_section("📌 KEY DECISIONS", decisions)
    print_section("✅ ACTION ITEMS", action_items)


if __name__ == "__main__":
    main()

