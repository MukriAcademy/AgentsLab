# app_fixed.py
"""
RAG Chat App (Fixed for modern LlamaIndex v0.11+)
Features:
- Upload PDF / TXT -> text extraction -> chunking -> index (FAISS) persisted under INDEX_DIR
- Query documents -> retrieve top-k chunks -> pass context to LLM -> produce answer
- Calculator utility (safe eval)
- Optional SerpAPI web search tool (if SERPAPI_API_KEY set)
- Uses OpenAI for generation if OPENAI_API_KEY set, otherwise uses HF text-generation pipeline
"""

import os
import math
import faiss
import json
import gradio as gr
from typing import List, Tuple

# ---- LlamaIndex (v0.11+ correct imports) ----
from llama_index import SimpleDirectoryReader, StorageContext, ServiceContext
from llama_index import (
 
    Document,
)
from llama_index.indices.vector_store import VectorStoreIndex
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index import load_index_from_storage

# embedding wrapper base
from llama_index.embeddings.base import BaseEmbedding

# ---- sentence-transformers for embeddings ----
from sentence_transformers import SentenceTransformer

# ---- Transformers for local generation fallback ----
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

from llama_index.embeddings.huggingface import SentenceTransformerEmbedding

# ---- Optional OpenAI usage ----
try:
    import openai
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False

# ---- Optional SerpAPI ----
try:
    from serpapi import GoogleSearch
    HAS_SERPAPI = True
except Exception:
    HAS_SERPAPI = False

# ---------------- Config ----------------
DATA_DIR = "Uploads"
INDEX_DIR = "IndexDir"
DOCS_DIR = "Docs"  # used when building from existing docs folder
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

OPENAI_KEY = os.getenv("OPENAI_API_KEY", None)
SERPAPI_KEY = os.getenv("SERPAPI_API_KEY", None)

# sentence-transformers model (you can change)
ST_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# local HF model fallback for generation (small default)
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "gpt2")  # change if you have bigger model
DEVICE = "cuda" if (os.getenv("CUDA_VISIBLE_DEVICES") or os.getenv("CUDA_DEVICE_ORDER")) else "cpu"

# retrieval & chunking settings
CHUNK_SIZE = 800
TOP_K = 3

# ---------------- Embedding adapter for LlamaIndex ----------------
# class STEmbedding(BaseEmbedding):
#     """Wrap a sentence-transformers model for LlamaIndex."""
#     def __init__(self, model_name=ST_MODEL_NAME):
#         self.st = SentenceTransformer(model_name)

#     def get_text_embedding(self, text: str) -> List[float]:
#         return self.st.encode(text).tolist()

#     def get_query_embedding(self, text: str) -> List[float]:
#         return self.st.encode(text).tolist()

# # instantiate embedding adapter
# embed_model = STEmbedding(ST_MODEL_NAME)

#ST_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

embed_model = SentenceTransformerEmbedding(model_name=ST_MODEL_NAME)

# ---------------- Helper: safe math evaluator ----------------
import ast, operator as op

# allowed operators
_allowed_operators = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.Pow: op.pow, ast.USub: op.neg, ast.UAdd: op.pos, ast.Mod: op.mod,
    ast.FloorDiv: op.floordiv
}

def safe_eval(expr: str):
    """Safely evaluate a math expression using ast (no __import__ or names)."""
    def _eval(node):
        if isinstance(node, ast.Num):  # <number>
            return node.n
        if isinstance(node, ast.Constant):  # Python 3.8+
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Constants other than numbers are not allowed")
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            op_type = type(node.op)
            if op_type in _allowed_operators:
                return _allowed_operators[op_type](left, right)
            raise ValueError(f"Operator {op_type} not allowed")
        if isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            op_type = type(node.op)
            if op_type in _allowed_operators:
                return _allowed_operators[op_type](operand)
            raise ValueError(f"Operator {op_type} not allowed")
        raise ValueError(f"Unsupported expression: {ast.dump(node)}")

    node = ast.parse(expr, mode="eval").body
    return _eval(node)

# ---------------- Index building / loading ----------------
def index_exists(path: str) -> bool:
    """Check if essential index files exist in directory."""
    # LlamaIndex FAISS persistence may create vector_store.faiss and docstore.json (or similar)
    return os.path.exists(os.path.join(path, "vector_store.faiss")) and os.path.exists(os.path.join(path, "docstore.json"))

def build_index_from_docs(docs_dir: str = DOCS_DIR, persist_dir: str = INDEX_DIR):
    """Build index from all files in docs_dir and persist to persist_dir."""
    print("Building new index from docs folder...")
    # read documents
    reader = SimpleDirectoryReader(docs_dir)
    documents = reader.load_data()
    if not documents:
        return None

    # create a FAISS index instance with dimension inferred from embedding model
    # we need dim: run a sample embedding and get length
    sample_emb = embed_model.get_text_embedding("test")
    dim = len(sample_emb)
    faiss_index = faiss.IndexFlatL2(dim)
    vector_store = FaissVectorStore(faiss_index=faiss_index)

    storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=persist_dir)
    service_context = ServiceContext.from_defaults(embedding_model=embed_model)

    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, service_context=service_context)
    index.storage_context.persist(persist_dir=persist_dir)
    print("Index built and saved to", persist_dir)
    return index

def load_or_create_index(persist_dir: str = INDEX_DIR):
    """Load index if present else try to build from DOCS_DIR or return empty index object."""
    # attempt load
    try:
        if index_exists(persist_dir):
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            print("Loading index from", persist_dir)
            index = VectorStoreIndex.from_storage(storage_context)
            return index
    except Exception as e:
        print("Warning loading index:", e)

    # attempt to build from DOCS_DIR if files exist
    if os.listdir(DOCS_DIR):
        try:
            return build_index_from_docs(DOCS_DIR, persist_dir)
        except Exception as e:
            print("Error building from docs:", e)

    # else create empty index (no docs)
    # create empty faiss with dim from embed sample
    sample_emb = embed_model.get_text_embedding("sample")
    dim = len(sample_emb)
    faiss_index = faiss.IndexFlatL2(dim)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=persist_dir)
    service_context = ServiceContext.from_defaults(embedding_model=embed_model)
    index = VectorStoreIndex.from_documents([], storage_context=storage_context, service_context=service_context)
    index.storage_context.persist(persist_dir=persist_dir)
    print("Created empty index at", persist_dir)
    return index

# load or create index
INDEX = load_or_create_index()

# ---------------- Document handling ----------------
def save_uploaded_file(file_obj) -> str:
    """Save Gradio uploaded file to disk and return path."""
    if not file_obj:
        return None
    dest = os.path.join(DATA_DIR, os.path.basename(file_obj.name))
    with open(dest, "wb") as f:
        f.write(file_obj.read())
    return dest

def extract_text_from_file(path: str) -> str:
    """Extract text from pdf or txt."""
    text = ""
    if path.lower().endswith(".pdf"):
        try:
            import fitz
            doc = fitz.open(path)
            for page in doc:
                text += page.get_text()
        except Exception:
            from PyPDF2 import PdfReader
            reader = PdfReader(path)
            for p in reader.pages:
                txt = p.extract_text()
                if txt:
                    text += txt + "\n"
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    return text

def add_document_file(file_obj):
    """Save uploaded file, chunk it and insert into the index (persist)."""
    path = save_uploaded_file(file_obj)
    if not path:
        return "No file provided."
    text = extract_text_from_file(path)
    if not text.strip():
        return "No text extracted."

    # chunk text
    chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    docs = [Document(text=chunk) for chunk in chunks]

    # load index storage context and insert docs
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    try:
        index = VectorStoreIndex.from_storage(storage_context)
    except Exception:
        # create a fresh index object if loading failed
        sample_emb = embed_model.get_text_embedding("sample")
        dim = len(sample_emb)
        faiss_index = faiss.IndexFlatL2(dim)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=INDEX_DIR)
        service_context = ServiceContext.from_defaults(embedding_model=embed_model)
        index = VectorStoreIndex.from_documents([], storage_context=storage_context, service_context=service_context)

    # insert and persist
    index.insert_documents(docs)
    index.storage_context.persist(persist_dir=INDEX_DIR)
    return f"Added {len(docs)} chunks from {os.path.basename(path)}"

# ---------------- Retrieval & Generation ----------------
def retrieve_context(query: str, top_k: int = TOP_K) -> List[str]:
    """Return top-k document texts for the query."""
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    index = VectorStoreIndex.from_storage(storage_context)
    query_engine = index.as_query_engine()
    resp = query_engine.query(query, similarity_top_k=top_k)
    # resp may be a Response object; try to extract string
    text = str(resp)
    # LlamaIndex can produce formatted reply; we also return raw matched docs via query_engine.get_top_k? not stable
    return [text]

# ---- Generation: OpenAI or HF pipeline fallback ----
def generate_with_openai(prompt: str, max_tokens=256, temperature=0.0):
    if not HAS_OPENAI:
        raise RuntimeError("openai package not installed")
    if not OPENAI_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")
    openai.api_key = OPENAI_KEY
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini" if False else "gpt-4o-mini",  # replace model as needed
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp["choices"][0]["message"]["content"]

# HF fallback generator
HF_PIPELINE = None
def get_hf_pipeline():
    global HF_PIPELINE
    if HF_PIPELINE is not None:
        return HF_PIPELINE
    # small model generation pipeline
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(HF_MODEL_ID)
    HF_PIPELINE = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto" if DEVICE=="cuda" else None)
    return HF_PIPELINE

def generate_answer(question: str, top_k: int = TOP_K):
    # retrieve context
    contexts = retrieve_context(question, top_k=top_k)
    context_text = "\n\n".join(contexts)

    system_instruction = (
        "You are a helpful assistant that answers user questions using the provided context. "
        "If the answer is not contained in the context, be honest and say you don't know. Keep answers concise."
    )

    prompt = f"{system_instruction}\n\nContext:\n{context_text}\n\nQuestion: {question}\nAnswer:"

    # try OpenAI if available
    if HAS_OPENAI and OPENAI_KEY:
        try:
            return generate_with_openai(prompt, max_tokens=256, temperature=0.0)
        except Exception as e:
            print("OpenAI generation failed:", e)

    # fallback to HF pipeline (may be weak)
    pipe = get_hf_pipeline()
    out = pipe(prompt, max_new_tokens=200, do_sample=False)
    text = out[0]["generated_text"]
    # if tokenizer returns entire prompt+continuation, strip prompt if present
    if prompt in text:
        text = text.split(prompt, 1)[1].strip()
    return text.strip()

# ---------------- Web search tool ----------------
def web_search(query: str):
    if not HAS_SERPAPI or not SERPAPI_KEY:
        return "SerpAPI not available or SERPAPI_KEY not set."
    params = {"q": query, "api_key": SERPAPI_KEY, "engine": "google"}
    search = GoogleSearch(params)
    try:
        results = search.get_dict()
    except Exception as e:
        return f"Search failed: {e}"
    items = []
    for r in results.get("organic_results", [])[:5]:
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        link = r.get("link", "")
        items.append(f"{title}\n{snippet}\n{link}\n")
    return "\n\n".join(items) if items else "No search results."

# ---------------- Gradio handlers ----------------
def handle_ask(question: str, history):
    if not question or question.strip() == "":
        return history, "Please enter a question."

    # special-case calculator calls: if user starts with "calc:" treat as math
    if question.strip().lower().startswith("calc:"):
        expr = question.strip()[5:].strip()
        try:
            res = safe_eval(expr)
            answer = f"Result: {res}"
        except Exception as e:
            answer = f"Calculation error: {e}"
        history = history or []
        history.append((question, answer))
        return history, history

    # special-case websearch: starts with "search:"
    if question.strip().lower().startswith("search:"):
        q = question.strip()[7:].strip()
        answer = web_search(q)
        history = history or []
        history.append((question, answer))
        return history, history

    # default: RAG answer
    try:
        answer = generate_answer(question)
    except Exception as e:
        answer = f"Error generating answer: {e}"

    history = history or []
    history.append((question, answer))
    return history, history

def handle_list_docs():
    files = os.listdir(DATA_DIR)
    return "\n".join(files) if files else "No uploaded documents."

# ---------------- Gradio UI ----------------
with gr.Blocks(title="RAG Chat (fixed)") as demo:
    gr.Markdown("## 🧠 RAG Chat App — Upload docs, then ask questions\n\n"
                "- Prefix a question with `calc:` to use the calculator (e.g. `calc: 2+2*3`).\n"
                "- Prefix with `search:` to run a web search (if SerpAPI key is set).")

    with gr.Row():
        with gr.Column(scale=1):
            upload_file = gr.File(label="Upload Document (txt or pdf)", file_types=[".txt", ".pdf"])
            upload_btn = gr.Button("Upload & Index")
            upload_status = gr.Textbox(label="Upload status")
            refresh_btn = gr.Button("Refresh docs list")
            docs_list = gr.Textbox(label="Indexed documents", lines=10, interactive=False)
            refresh_btn.click(lambda: handle_list_docs(), None, docs_list)
            upload_btn.click(add_document_file, inputs=upload_file, outputs=upload_status)

        with gr.Column(scale=2):
            chat = gr.Chatbot()
            state = gr.State([])
            question = gr.Textbox(label="Question", placeholder="Ask about uploaded documents...")
            ask_btn = gr.Button("Ask")
            ask_btn.click(handle_ask, inputs=[question, state], outputs=[chat, state])
            question.submit(handle_ask, inputs=[question, state], outputs=[chat, state])
            clear_btn = gr.Button("Clear Chat")
            clear_btn.click(lambda: None, None, chat, queue=False)

    # bottom: show which services are configured
    info = "Services: "
    info += "OpenAI enabled. " if OPENAI_KEY else "OpenAI not configured. "
    info += "SerpAPI enabled. " if (HAS_SERPAPI and SERPAPI_KEY) else "SerpAPI not configured. "
    gr.Markdown(info)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
