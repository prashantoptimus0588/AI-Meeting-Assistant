import os
import shutil
from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from uuid import uuid4

from core.extractor import extract_all
from core.summarize import generate_title_and_summary
from core.transcriber import transcribe_all
from utils.audio_processor import process_input
from core.rag_engine import build_rag_chain


UPLOAD_DIR = "uploads"
MAX_UPLOAD_MB = 50
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".mp3", ".wav", ".m4a"}
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(
    title="AI Meeting Assistant API",
    description="API for meeting transcription, summarization and RAG",
    version="1.0.0",
)


# Temporary in-memory storage
meetings = {}


class MeetingRequest(BaseModel):
    source: str
    language: str = "english"


class ChatRequest(BaseModel):
    question: str


def process_meeting(
    meeting_id: str,
    source: str,
    language: str,
    cleanup_dir: str | None = None,

):
    try:
        meetings[meeting_id]["status"] = "processing"

        # 1. Process audio
        chunks = process_input(source)

        # 2. Transcribe
        transcript = transcribe_all(chunks, language)
        print("Transcription complete.")

        # 3. Generate title and summary
        print("Generating title and summary...")
        title_and_summary = generate_title_and_summary(transcript)
        print("Title and summary complete.")
        
        
        # 4. Extract information
        print("Extracting information...")
        extracted = extract_all(transcript)
        print("Extraction complete.")
        
        # 5. Build RAG chain
        print("Building RAG chain...")
        rag_chain = build_rag_chain(transcript)
        print("RAG chain ready.")


        # Store RAG chain
        meetings[meeting_id]["rag_chain"] = rag_chain

        # Store analysis result
        meetings[meeting_id]["result"] = {
            "title": title_and_summary["title"],
            "summary": title_and_summary["summary"],
            "transcript": transcript,
            "action_items": extracted["action_items"],
            "key_decisions": extracted["key_decisions"],
            "open_questions": extracted["questions"],
        }

        # Mark processing as completed
        meetings[meeting_id]["status"] = "completed"

    except Exception as e:
        meetings[meeting_id]["status"] = "failed"
        meetings[meeting_id]["error"] = str(e)
        print(f"Meeting processing failed: {type(e).__name__}: {e}")

    finally:
        if cleanup_dir and os.path.isdir(cleanup_dir):
            shutil.rmtree(cleanup_dir, ignore_errors=True)


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/meetings")
def create_meeting(
    request: MeetingRequest,
    background_tasks: BackgroundTasks,
):
    meeting_id = str(uuid4())

    meetings[meeting_id] = {
        "status": "queued",
        "result": None,
        "error": None,
        "rag_chain": None,
    }

    background_tasks.add_task(
        process_meeting,
        meeting_id,
        request.source,
        request.language,
    )

    return {
        "meeting_id": meeting_id,
        "status": "queued",
    }


@app.get("/meetings/{meeting_id}/status")
def get_meeting_status(meeting_id: str):

    if meeting_id not in meetings:
        raise HTTPException(
            status_code=404,
            detail="Meeting not found",
        )

    meeting = meetings[meeting_id]

    response = {
        "meeting_id": meeting_id,
        "status": meeting["status"],
    }

    if meeting["status"] == "completed":
        response["result"] = meeting["result"]

    if meeting["status"] == "failed":
        response["error"] = meeting["error"]

    return response


@app.post("/meetings/{meeting_id}/chat")
def chat_with_meeting(
    meeting_id: str,
    request: ChatRequest,
):
    if meeting_id not in meetings:
        raise HTTPException(
            status_code=404,
            detail="Meeting not found",
        )

    meeting = meetings[meeting_id]

    if meeting["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail="Meeting processing is not completed yet",
        )

    rag_chain = meeting.get("rag_chain")

    if rag_chain is None:
        raise HTTPException(
            status_code=500,
            detail="RAG chain not available",
        )

    try:
        answer = rag_chain.invoke(request.question)

        return {
            "meeting_id": meeting_id,
            "question": request.question,
            "answer": answer,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )
        
    
@app.post("/meetings/upload")
def upload_meeting(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: str = Form("english"),
):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    meeting_id = str(uuid4())
    meeting_dir = os.path.join(UPLOAD_DIR, meeting_id)
    os.makedirs(meeting_dir, exist_ok=True)
    save_path = os.path.join(meeting_dir, f"input{ext}")

    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    written = 0
    try:
        with open(save_path, "wb") as out:
            while chunk := file.file.read(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Max {MAX_UPLOAD_MB} MB.",
                    )
                out.write(chunk)
    except HTTPException:
        shutil.rmtree(meeting_dir, ignore_errors=True)
        raise

    meetings[meeting_id] = {
        "status": "queued",
        "result": None,
        "error": None,
        "rag_chain": None,
    }

    background_tasks.add_task(
        process_meeting,
        meeting_id,
        save_path,
        language,
        meeting_dir,
    )

    return {"meeting_id": meeting_id, "status": "queued"}