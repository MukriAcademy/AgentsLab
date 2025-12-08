import os
from typing import List, Tuple
import gradio as gr
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# ---------- Configuration ----------
DATA_DIR = "Uploads"
INDEX_DIR = "IndexDir"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Global vectorstore
VECTORSTORE = None

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

# ---------- Load or create vectorstore ----------
def load_vectorstore():
    """Load existing vectorstore or return None."""
    global VECTORSTORE
    index_path = os.path.join(INDEX_DIR, "faiss_index")
    
    if os.path.exists(index_path):
        try:
            VECTORSTORE = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            print("✅ Vectorstore loaded from disk.")
            return True
        except Exception as e:
            print(f"⚠️ Error loading vectorstore: {e}")
            VECTORSTORE = None
            return False
    else:
        print("📦 No existing vectorstore found.")
        VECTORSTORE = None
        return False

# Load vectorstore on startup
load_vectorstore()

# ---------- Add documents ----------
def add_document_file(file_obj):
    """Add a document file to the vectorstore."""
    global VECTORSTORE
    
    if file_obj is None:
        return "❌ No file provided!"
    
    try:
        path = save_uploaded_file(file_obj)
        text = extract_text_from_file(path)

        if not text.strip():
            return "❌ No text extracted from the file."
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        # Create or update vectorstore
        if VECTORSTORE is None:
            VECTORSTORE = FAISS.from_texts(chunks, embeddings, metadatas=[{"source": file_obj.name}] * len(chunks))
        else:
            new_vectorstore = FAISS.from_texts(chunks, embeddings, metadatas=[{"source": file_obj.name}] * len(chunks))
            VECTORSTORE.merge_from(new_vectorstore)
        
        # Save to disk
        index_path = os.path.join(INDEX_DIR, "faiss_index")
        VECTORSTORE.save_local(index_path)
        
        return f"✅ Added {len(chunks)} chunks from file: {file_obj.name}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ---------- Tools ----------
@tool
def document_qa_tool(query: str) -> str:
    """Search and answer questions from indexed documents."""
    global VECTORSTORE
    
    if VECTORSTORE is None:
        return "No documents have been uploaded yet. Please upload documents first."
    
    try:
        docs = VECTORSTORE.similarity_search(query, k=3)
        
        if not docs:
            return "No relevant information found in the documents."
        
        # Combine results
        context = "\n\n".join([doc.page_content for doc in docs])
        sources = set([doc.metadata.get("source", "Unknown") for doc in docs])
        
        result = f"Found information from: {', '.join(sources)}\n\n{context[:1000]}"
        return result
    except Exception as e:
        return f"Error querying documents: {str(e)}"

@tool
def calculator_tool(expression: str) -> str:
    """Perform mathematical calculations. Input should be a Python expression like '2+2' or '10*5'."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error in calculation: {str(e)}"

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

# Initialize LLM - try Gemini, fallback to Ollama
from dotenv import load_dotenv
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

if google_api_key:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=google_api_key,
            temperature=0
        )
        print("✅ Using Google Gemini")
    except Exception as e:
        print(f"⚠️ Gemini failed: {e}")
        google_api_key = None

if not google_api_key:
    try:
        from langchain_community.llms import Ollama
        llm = Ollama(model="llama2", temperature=0)
        print("✅ Using Ollama (make sure Ollama is installed and running)")
        print("   Install from: https://ollama.ai")
        print("   Run: ollama pull llama2")
    except Exception as e:
        print(f"❌ Both Gemini and Ollama failed")
        print("Please either:")
        print("1. Set GOOGLE_API_KEY environment variable")
        print("2. Install Ollama from https://ollama.ai")
        exit(1)

agent_executor = create_react_agent(llm, tools)

# ---------- Gradio app functions ----------
def ask_agent(question: str, chat_history: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """User sends message -> agent answers using tools -> append to history."""
    if not question or question.strip() == "":
        return chat_history, chat_history
    
    try:
        if USE_AGENT and agent_executor:
            # Use the ReAct agent
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
        else:
            # Simple fallback: try document QA first, then other tools
            answer = "Let me search the documents...\n\n"
            
            # Try document QA
            doc_result = document_qa_tool.invoke(question)
            if "No documents" not in doc_result and "Error" not in doc_result:
                answer += f"📄 From documents:\n{doc_result}\n\n"
            
            # Check if it's a math question
            if any(op in question for op in ['+', '-', '*', '/', 'calculate', 'compute']):
                import re
                math_expr = re.search(r'[\d\+\-\*\/\(\)\s]+', question)
                if math_expr:
                    calc_result = calculator_tool.invoke(math_expr.group())
                    answer += f"🧮 Calculation: {calc_result}\n\n"
            
            # If no good answer yet, try web search
            if len(answer) < 100:
                web_result = web_search_tool.invoke(question)
                answer += f"🌐 From web:\n{web_result[:500]}..."
            
    except Exception as e:
        answer = f"❌ Error: {str(e)}"
    
    chat_history = chat_history or []
    chat_history.append((question, answer))
    return chat_history, chat_history

def list_documents():
    """List uploaded documents."""
    try:
        files = os.listdir(DATA_DIR)
        if not files:
            return "No documents uploaded."
        return "\n".join([f"📄 {f}" for f in files])
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
    - First time loading embeddings may take a moment
    """)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7878, share=True)