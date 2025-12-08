import os
import tempfile
import json
from typing import List, Tuple

import gradio as gr

# LlamaIndex for ingest + retrieval
from langchain_google_genai import ChatGoogleGenerativeAI
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Document,
    Settings
)
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore
from langchain_openai import ChatOpenAI

from langgraph.prebuilt import create_react_agent

# LangChain for agent and tools
from langchain_core.tools import tool

# ---------- Configuration ----------
DATA_DIR = "Uploads"
INDEX_DIR = "IndexDir"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY",None)
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY",None)

# ---------- Utility functions ----------
def save_uploaded_file(file_obj) -> str:
    """Save uploaded file to DATA_DIR and return the file path."""
    file_path = os.path.join(DATA_DIR, file_obj.name)
    with open(file_path, "wb") as f:
        f.write(file_obj.read())
    return file_path

def extract_text_from_file(path: str) -> str:
    """Extract text from a file. Supports .txt and .pdf files."""
    text = ""
    if path.lower().endswith(".pdf"):
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(path)
            for page in doc:
                text += page.get_text()
        except Exception:
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(path)
                for page in reader.pages:
                    txt = page.extract_text()
                    if txt:
                        text += txt + "\n"
            except Exception as e:
                return f"Error reading PDF: {str(e)}"
    else:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"
    return text

# ---------- Index (LlamaIndex) ----------
def build_or_load_index():
    """Load index from disk if exists, otherwise create empty index."""
    storage_path = INDEX_DIR
    
    # Check if index exists
    if os.path.exists(os.path.join(storage_path, "docstore.json")):
        try:
            storage_context = StorageContext.from_defaults(persist_dir=storage_path)
            index = VectorStoreIndex([], storage_context=storage_context)
            print("✅ Index loaded from disk.")
            return index
        except Exception as e:
            print(f"⚠️ Error loading index: {e}")
    
    # Create new empty index
    print("📦 Creating new empty index...")
    storage_context = StorageContext.from_defaults()
    index = VectorStoreIndex([], storage_context=storage_context)
    index.storage_context.persist(persist_dir=storage_path)
    print("✅ New index created and persisted.")
    return index

# Initialize global index
INDEX = build_or_load_index()

# ---------- Add documents ----------
def add_document_file(file_obj):
    """Add a document file to the index."""
    global INDEX
    
    if file_obj is None:
        return "❌ No file provided!"
    
    try:
        path = save_uploaded_file(file_obj)
        text = extract_text_from_file(path)

        if not text.strip():
            return "❌ No text extracted from the file."
        
        # Split text into chunks
        chunk_size = 800
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        docs = [Document(text=chunk, metadata={"filename": file_obj.name}) for chunk in chunks]

        # Reload index
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        INDEX = VectorStoreIndex([], storage_context=storage_context)
        
        # Add documents
        for doc in docs:
            INDEX.insert(doc)
        
        # Persist
        INDEX.storage_context.persist(persist_dir=INDEX_DIR)
        
        return f"✅ Added {len(docs)} chunks from file: {file_obj.name}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ---------- Retrieval and QA Tools ----------
@tool
def document_qa_tool(query: str) -> str:
    """Search and answer questions from indexed documents."""
    try:
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        index = VectorStoreIndex([], storage_context=storage_context)
        
        # Query
        query_engine = index.as_query_engine(similarity_top_k=3)
        response = query_engine.query(query)
        
        return str(response)
    except Exception as e:
        return f"Error querying documents: {str(e)}"

# ---------- Calculator tool ----------
@tool
def calculator_tool(expression: str) -> str:
    """Perform mathematical calculations. Input should be a Python expression like '2+2' or '10*5'."""
    try:
        # Safe eval with limited scope
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error in calculation: {str(e)}"

# ---------- Web Search Tool ----------
@tool
def web_search_tool(query: str) -> str:
    """Search the web for current information."""
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        search = DuckDuckGoSearchRun()
        result = search.run(query)
        return result
    except Exception as e:
        return f"Error in web search: {str(e)}"

# ---------- Create Agent ----------
tools = [document_qa_tool, calculator_tool, web_search_tool]

#model="models/gemini-1.5-flash",
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
    convert_system_message_to_human=True,
)
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

agent_executor = create_react_agent(llm, tools)

# ---------- Gradio app functions ----------
def ask_agent(question: str, chat_history: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """User sends message -> agent answers using tools -> append to history."""
    if not question or question.strip() == "":
        return chat_history, chat_history
    
    try:
        response = agent_executor.invoke({
            "messages": [("user", question)]
        })
        
        # Extract final answer
        answer = ""
        for message in response["messages"]:
            if hasattr(message, 'content') and message.content:
                answer = message.content
        
        if not answer:
            answer = "No response from agent."
            
    except Exception as e:
        answer = f"❌ Error from agent: {str(e)}"
    
    chat_history = chat_history or []
    chat_history.append((question, answer))
    return chat_history, chat_history

def list_documents():
    """List uploaded documents."""
    try:
        files = os.listdir(DATA_DIR)
        return "\n".join(files) if files else "No documents uploaded."
    except Exception as e:
        return f"Error: {str(e)}"

# ---------- Gradio UI ----------
with gr.Blocks(title="RAG + Agent Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 RAG Agent Chatbot\nUpload documents (PDF/TXT), then ask questions. Agent has tools: DocumentQA, Calculator, WebSearch")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📂 Document Management")
            upload = gr.File(label="Upload PDF or Text File", file_types=[".pdf", ".txt"])
            upload_button = gr.Button("📤 Add Document", variant="primary")
            upload_output = gr.Textbox(label="Status", interactive=False, lines=3)
            
            gr.Markdown("---")
            refresh_button = gr.Button("🔄 Refresh Document List")
            docs_list = gr.Textbox(label="Uploaded Documents", value=list_documents(), interactive=False, lines=8)

        with gr.Column(scale=2):
            gr.Markdown("### 💬 Chat with Agent")
            chatbot = gr.Chatbot(label="Conversation", height=400)
            msg = gr.Textbox(label="Ask a question", placeholder="e.g., What is the main topic of the uploaded document?")
            with gr.Row():
                submit_btn = gr.Button("🚀 Send", variant="primary")
                clear = gr.Button("🗑️ Clear Chat")
    
    # State for chat history
    chat_history = gr.State([])
    
    # Event handlers
    upload_button.click(add_document_file, inputs=upload, outputs=upload_output)
    refresh_button.click(list_documents, inputs=None, outputs=docs_list)
    
    msg.submit(ask_agent, [msg, chat_history], [chatbot, chat_history])
    submit_btn.click(ask_agent, [msg, chat_history], [chatbot, chat_history])
    
    clear.click(lambda: ([], []), None, [chatbot, chat_history], queue=False)
    
    gr.Markdown("""
    ### 📝 Notes:
    - Set `GOOGLE_API_KEY` in your environment to use the Gemini model
    - Documents are stored in the 'Uploads' directory
    - Vector index is stored in 'IndexDir'
    - The agent can search documents, calculate, and search the web
    """)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7878, share=True)