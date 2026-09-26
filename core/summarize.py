import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from concurrent.futures import ThreadPoolExecutor

load_dotenv()
import os

MAX_SUMMARY_WORKERS = 8  # cap concurrent Gemini calls; tune based on your API rate limit


def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3,
    )


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )
    return splitter.split_text(transcript)


def _map_chunk_summaries(transcript: str) -> str:
    """Independent per-chunk summaries, run concurrently. Returns the combined text."""
    llm = get_llm()
    map_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Summarize this portion of a meeting transcript concisely."),
            ("human", "{text}"),
        ]
    )
    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)

    with ThreadPoolExecutor(max_workers=min(MAX_SUMMARY_WORKERS, len(chunks))) as executor:
        chunk_summaries = list(
            executor.map(lambda chunk: map_chain.invoke({"text": chunk}), chunks)
        )

    return "\n\n".join(chunk_summaries)


TITLE_AND_SUMMARY_SYSTEM_PROMPT = """You are an expert meeting summarizer. You are given partial
summaries of sequential portions of a meeting transcript.

Produce:
1. "title" — a short professional meeting title (max 8 words).
2. "summary" — one final professional meeting summary in bullet points, combining
   all the partial summaries into a single coherent overview.

Return ONLY valid JSON in exactly this shape, with no markdown fences and no extra text:
{{
  "title": "...",
  "summary": "..."
}}
"""


def generate_title_and_summary(transcript: str) -> dict:
    """
    Single-call replacement for the old generate_title() + the combine step of
    summarize(). The per-chunk map step still runs first (each chunk is
    genuinely independent work and is already parallelized), but the final
    title + summary generation — which both read the same combined text —
    now happens in one Gemini call instead of two.
    """
    combined = _map_chunk_summaries(transcript)

    llm = get_llm()
    chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages(
            [
                ("system", TITLE_AND_SUMMARY_SYSTEM_PROMPT),
                ("human", "{text}"),
            ]
        )
        | llm
        | StrOutputParser()
    )

    raw = chain.invoke(combined)
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback so a malformed response doesn't crash the pipeline —
        # summary falls back to the raw combined chunk text, title to a generic default.
        data = {"title": "Untitled Meeting", "summary": combined}

    return {
        "title": data.get("title", "Untitled Meeting"),
        "summary": data.get("summary", combined),
    }