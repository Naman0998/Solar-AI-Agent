import os
import platform
import asyncio
import fitz  # PyMuPDF
import chromadb
from fastapi import FastAPI, Request
from pydantic import BaseModel
from dotenv import load_dotenv

from retriever import ingest_pdfs_and_store_chroma, retrieve_relevant_chunks_from_chroma
from ask_llm import ask_llm
from file_auth import list_pdf_files_in_folder, download_pdf, FOLDER_ID
from fastapi.responses import HTMLResponse




# ✅ Windows async fix
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load env
load_dotenv()

# Setup FastAPI
app = FastAPI(title="🔆 Solar Regulation & Finance Assistant")

# Setup persistent ChromaDB
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "solar_docs")
client = chromadb.PersistentClient(path=CHROMA_PATH)
chroma_collection = client.get_or_create_collection(COLLECTION_NAME)

# -----------------------------
# Request/Response Models
class QueryRequest(BaseModel):
    user_query: str

class WebhookMessage(BaseModel):
    message: str
    user: str

# -----------------------------

@app.get("/", response_class=HTMLResponse)
def home():
    return "<h1>🚀 VS Chatbot is running!</h1><p>Go to <a href='/docs'>/docs</a> to test the API.</p>"

@app.post("/ingest")
async def ingest_pdfs():
    """
    Fetch PDFs from Google Drive, embed, and store in ChromaDB.
    """
    pdf_dir = "./downloaded_pdfs"
    os.makedirs(pdf_dir, exist_ok=True)

    pdf_files = list_pdf_files_in_folder(FOLDER_ID)
    pdf_names, pdf_texts = [], []

    for file_id, file_name in pdf_files:
        dest_path = os.path.join(pdf_dir, file_name)
        pdf_names.append(file_name)
        if not os.path.exists(dest_path):
            download_pdf(file_id, dest_path)
        doc = fitz.open(dest_path)
        full_text = "\n".join([page.get_text() for page in doc])
        pdf_texts.append(full_text)

    if pdf_names:
        ingest_pdfs_and_store_chroma(pdf_texts, pdf_names, chroma_collection)
        return {"status": "success", "files": pdf_names}
    else:
        return {"status": "error", "message": "No PDFs found in Google Drive folder."}

# -----------------------------
@app.post("/query")
async def query_docs(req: QueryRequest):
    """
    Retrieve top chunks from ChromaDB and query OpenRouter LLM.
    """
    user_query = req.user_query
    top_chunks, _ = retrieve_relevant_chunks_from_chroma(user_query, chroma_collection)
    context = "\n\n".join(top_chunks)

    response = ask_llm(
        f"Based only on the following context:\n\n{context}\n\nAnswer this:\n{user_query}"
    )
    return {"answer": response, "context_chunks": top_chunks}

# -----------------------------
@app.post("/webhook")
async def google_chat_webhook(msg: WebhookMessage):
    """
    Google Chat bot integration – receives message and returns LLM answer.
    """
    query = msg.message
    top_chunks, _ = retrieve_relevant_chunks_from_chroma(query, chroma_collection)
    context = "\n\n".join(top_chunks)

    response = ask_llm(
        f"Based only on the following context:\n\n{context}\n\nAnswer this:\n{query}"
    )
    return {"reply": response}

# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
