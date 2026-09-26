import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os
from langchain_google_genai import ChatGoogleGenerativeAI


def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3,
    )

EXTRACTION_SYSTEM_PROMPT = """You are an expert meeting analyst. From the meeting transcript, extract:

1. action_items — a list of objects, each with:
   - "task": task description
   - "owner": who is responsible (or "Not specified")
   - "deadline": deadline if mentioned (or "Not specified")

2. key_decisions — a list of strings, each a key decision made in the meeting.

3. questions — a list of strings, each an unresolved question or topic needing follow-up.

If a category has nothing found, return an empty list for it.

Return ONLY valid JSON in exactly this shape, with no markdown fences and no extra text:
{{
  "action_items": [{{"task": "...", "owner": "...", "deadline": "..."}}],
  "key_decisions": ["..."],
  "questions": ["..."]
}}
"""


def _build_extraction_chain():
    llm = get_llm()
    return (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages(
            [
                ("system", EXTRACTION_SYSTEM_PROMPT),
                ("human", "{text}"),
            ]
        )
        | llm
        | StrOutputParser()
    )


def extract_all(transcript: str) -> dict:
    """Single-call replacement for extract_action_items + extract_key_decisions + extract_questions."""
    chain = _build_extraction_chain()
    raw = chain.invoke(transcript)

    # Defensive cleanup in case the model wraps output in ```json fences anyway
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback so a malformed response doesn't crash the whole pipeline
        data = {"action_items": [], "key_decisions": [], "questions": []}

    return {
        "action_items": data.get("action_items", []),
        "key_decisions": data.get("key_decisions", []),
        "questions": data.get("questions", []),
    }