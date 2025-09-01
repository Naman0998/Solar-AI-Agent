import os
import re
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

# ------------------------------
# Load model (cache if possible)
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# ------------------------------
def parse_checklist_sections(text):
    sections = {}
    current_section = None
    lines = text.split('\n')
    buffer = []
    for line in lines:
        line_strip = line.strip()
        if ":" in line_strip and line_strip.endswith(":"):
            # save finished section
            if current_section and buffer:
                sections[current_section] = "\n".join(buffer).strip()
            current_section = line_strip.rstrip(':')
            buffer = []
        else:
            buffer.append(line_strip)
    # save last
    if current_section and buffer:
        sections[current_section] = "\n".join(buffer).strip()
    return sections

# ------------------------------
# Text Chunking
def chunk_text(text, chunk_size=1000, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def initialize_chroma_collection(path, collection_name):
    client = chromadb.PersistentClient(path=path)
    return client.get_or_create_collection(collection_name)

# ------------------------------
# Ingest and embed all PDFs, parse sections, and store in ChromaDB
def ingest_pdfs_and_store_chroma(pdf_texts, filenames, chroma_collection, model=None):
    if model is None:
        model = load_model()
    all_chunks = []
    chunk_metadata = []
    for file_text, filename in zip(pdf_texts, filenames):
        sections = parse_checklist_sections(file_text)
        for section, section_text in sections.items():
            chunks = chunk_text(section_text)
            all_chunks.extend(chunks)
            chunk_metadata.extend([{"filename": filename, "section": section}] * len(chunks))
    emb_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    chroma_collection.add(
        documents=all_chunks,
        metadatas=chunk_metadata,
        ids=[f"{m['filename']}:{m['section']}:{i}" for i, m in enumerate(chunk_metadata)]
    )

# ------------------------------
# Retrieve top relevant chunks from ChromaDB given a query
def retrieve_relevant_chunks_from_chroma(query, chroma_collection, top_k=5):
    results = chroma_collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas"]  # Ensure metadata is included
    )
    documents = results['documents'][0] if results['documents'] else []
    metadatas = results['metadatas'][0] if results['metadatas'] else []
    return documents, metadatas



