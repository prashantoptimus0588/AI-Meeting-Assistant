import uuid
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_embeddings_instance = None


def get_embeddings():
    # Reuse one embedding model instance across calls in this process
    # instead of reloading it from disk every time (step 6's fix, applied early).
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"}
        )
    return _embeddings_instance


def build_vector_store(transcript: str) -> Chroma:
    """
    Build a fresh, isolated, in-memory vector store for THIS transcript only.
    No persist_directory — nothing is written to disk, and a new random
    collection name means this store can never merge with a previous
    video's embeddings or leak into a future one.
    """
    print("Building vector store (in-memory, isolated)")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_text(transcript)

    docs = [
        Document(page_content=chunk, metadata={'chunk_index': i})
        for i, chunk in enumerate(chunks)
    ]

    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=f"meeting_{uuid.uuid4().hex}",  # unique every call
        # no persist_directory → ephemeral, in-memory only
    )

    return vector_store


def get_retriever(vector_store: Chroma, k: int = 4):
    return vector_store.as_retriever(
        search_type='similarity',
        search_kwargs={"k": k}
    )