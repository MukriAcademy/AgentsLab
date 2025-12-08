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
    # Handle different types of file objects from Gradio
    if isinstance(file_obj, str):
        # file_obj is already a path
        file_path = file_obj
        filename = os.path.basename(file_path)
        dest_path = os.path.join(DATA_DIR, filename)
        
        # Copy to DATA_DIR if not already there
        if file_path != dest_path:
            import shutil
            shutil.copy(file_path, dest_path)
        return dest_path
    elif hasattr(file_obj, 'name'):
        # file_obj has a name attribute (path)
        file_path = file_obj.name if hasattr(file_obj, 'name') else file_obj
        filename = os.path.basename(file_path)
        dest_path = os.path.join(DATA_DIR, filename)
        
        if file_path != dest_path:
            import shutil
            shutil.copy(file_path, dest_path)
        return dest_path
    else:
        # Old handling (shouldn't reach here with new Gradio)
        filename = getattr(file_obj, 'name', 'uploaded_file.txt')
        file_path = os.path.join(DATA_DIR, filename)
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
        # Get the file path
        if isinstance(file_obj, str):
            path = file_obj
            filename = os.path.basename(path)
        elif hasattr(file_obj, 'name'):
            path = file_obj.name
            filename = os.path.basename(path)
        else:
            return "❌ Invalid file object!"
        
        # Copy to DATA_DIR
        dest_path = os.path.join(DATA_DIR, filename)
        if path != dest_path:
            import shutil
            shutil.copy(path, dest_path)
        
        # Extract text
        text = extract_text_from_file(dest_path)

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
            VECTORSTORE = FAISS.from_texts(chunks, embeddings, metadatas=[{"source": filename}] * len(chunks))
        else:
            new_vectorstore = FAISS.from_texts(chunks, embeddings, metadatas=[{"source": filename}] * len(chunks))
            VECTORSTORE.merge_from(new_vectorstore)
        
        # Save to disk
        index_path = os.path.join(INDEX_DIR, "faiss_index")
        VECTORSTORE.save_local(index_path)
        
        return f"✅ Added {len(chunks)} chunks from file: {filename}"
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error details:\n{error_details}")
        return f"❌ Error: {str(e)}"

# ---------- Tools ----------
@tool
def document_qa_tool(query: str) -> str:
    """Search and answer questions from indexed documents.
    
    Args:
        query: The question to search for in the documents
        
    Returns:
        str: The answer found in the documents
    """
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
    """Perform mathematical calculations.
    
    Args:
        expression: A Python math expression like '2+2' or '10*5'
        
    Returns:
        str: The result of the calculation
    """
    try:
        # Safe evaluation
        import ast
        import operator
        
        ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
        }
        
        def eval_expr(node):
            if isinstance(node, ast.Num):
                return node.n
            elif isinstance(node, ast.BinOp):
                return ops[type(node.op)](eval_expr(node.left), eval_expr(node.right))
            elif isinstance(node, ast.UnaryOp):
                return ops[type(node.op)](eval_expr(node.operand))
            else:
                raise TypeError(node)
        
        result = eval_expr(ast.parse(expression, mode='eval').body)
        return str(result)
    except Exception as e:
        return f"Error in calculation: {str(e)}"

@tool
def web_search_tool(query: str) -> str:
    """Search the web for current information.
    
    Args:
        query: The search query
        
    Returns:
        str: Search results from the web
    """
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        search = DuckDuckGoSearchRun()
        result = search.run(query)
        return result
    except Exception as e:
        return f"Error in web search: {str(e)}"

# ---------- Initialize LLM ----------
tools = [document_qa_tool, calculator_tool, web_search_tool]

from dotenv import load_dotenv
load_dotenv()
# Check which API keys are available
groq_key = os.getenv("GROQ_API_KEY")
anthropic_key = os.getenv("ANTHROPIC_API_KEY")
google_key = os.getenv("GOOGLE_API_KEY")

print("\n" + "="*60)
print("Checking API Keys...")
print(f"GROQ_API_KEY: {'✅ SET' if groq_key else '❌ NOT SET'}")
print(f"ANTHROPIC_API_KEY: {'✅ SET' if anthropic_key else '❌ NOT SET'}")
print(f"GOOGLE_API_KEY: {'✅ SET' if google_key else '❌ NOT SET'}")
print("="*60 + "\n")

llm = None
USE_AGENT = False

# Try Groq first (best option)
if groq_key:
    # Try multiple Groq models
    groq_models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it"
    ]
    
    for model in groq_models:
        try:
            from langchain_groq import ChatGroq
            llm = ChatGroq(
                model=model,
                api_key=groq_key,
                temperature=0
            )
            # Test it
            llm.invoke("Hi")
            print(f"✅ Using Groq ({model})")
            USE_AGENT = True
            break
        except Exception as e:
            print(f"⚠️ Groq {model} failed: {str(e)[:80]}")
            continue

# Try Anthropic
if not USE_AGENT and anthropic_key:
    try:
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            api_key=anthropic_key,
            temperature=0
        )
        llm.invoke("Hi")
        print("✅ Using Anthropic Claude")
        USE_AGENT = True
    except Exception as e:
        print(f"⚠️ Anthropic failed: {str(e)[:100]}")

# Try Google Gemini
if not USE_AGENT and google_key:
    gemini_models = [
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-pro"
    ]
    
    for model in gemini_models:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(
                model=model,
                google_api_key=google_key,
                temperature=0,
                convert_system_message_to_human=True
            )
            llm.invoke("Hi")
            print(f"✅ Using Google Gemini ({model})")
            USE_AGENT = True
            break
        except Exception as e:
            print(f"⚠️ Google {model} failed: {str(e)[:80]}")
            continue

# Create agent if LLM is available
if USE_AGENT and llm:
    try:
        # Method 1: Try simple langgraph agent
        from langgraph.prebuilt import create_react_agent
        
        agent_executor = create_react_agent(llm, tools)
        print("✅ Agent created successfully\n")
    except Exception as e:
        print(f"⚠️ LangGraph agent failed: {str(e)[:100]}")
        print("Trying direct LLM with manual tool routing...\n")
        
        # Method 2: Manual implementation without agent framework
        agent_executor = None
        USE_AGENT = True  # We still have LLM, just no agent framework
        print("✅ Using direct LLM with manual tool calling\n")
else:
    agent_executor = None
    USE_AGENT = False
    print("\n" + "="*60)
    print("❌ NO LLM AVAILABLE!")
    print("="*60)
    print("\nTo use this app, set ONE of these API keys:\n")
    print("1. GROQ (Recommended - Free & Fast):")
    print("   → Get key: https://console.groq.com/keys")
    print("   → In PowerShell: $env:GROQ_API_KEY='your_key_here'")
    print()
    print("2. ANTHROPIC (Claude):")
    print("   → Get key: https://console.anthropic.com/")
    print("   → In PowerShell: $env:ANTHROPIC_API_KEY='your_key_here'")
    print()
    print("3. GOOGLE (Gemini):")
    print("   → Get key: https://aistudio.google.com/app/apikey")
    print("   → In PowerShell: $env:GOOGLE_API_KEY='your_key_here'")
    print("="*60 + "\n")

# ---------- Gradio app functions ----------
def ask_agent(question: str, chat_history: List) -> Tuple[List, List]:
    """User sends message -> agent answers using tools -> append to history."""
    if not question or question.strip() == "":
        return chat_history, chat_history
    
    if not USE_AGENT:
        answer = "❌ No LLM configured. Please set an API key (see console output for instructions)."
        chat_history = chat_history or []
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": answer})
        return chat_history, chat_history
    
    try:
        if agent_executor:
            # Use agent if available
            response = agent_executor.invoke({
                "messages": [("user", question)]
            })
            
            # Extract final answer
            answer = ""
            for message in response.get("messages", []):
                if hasattr(message, 'content') and message.content:
                    answer = message.content
            
            if not answer:
                answer = "No response from agent."
        else:
            # Manual tool routing with LLM
            print(f"\n🤖 Processing: {question}")
            
            # First, ask LLM which tool to use
            tool_selection_prompt = f"""You have these tools available:
1. document_qa_tool - Search uploaded documents
2. calculator_tool - Perform math calculations  
3. web_search_tool - Search the web

Question: {question}

Which tool should be used? Reply with ONLY the tool name, or 'none' if no tool needed."""

            tool_choice = llm.invoke(tool_selection_prompt).content.lower().strip()
            print(f"🔧 Selected tool: {tool_choice}")
            
            # Execute the appropriate tool
            if "document" in tool_choice:
                tool_result = document_qa_tool.invoke(question)
                answer = f"📄 {tool_result}"
            elif "calculator" in tool_choice:
                # Extract math expression
                import re
                expr = re.search(r'[\d\+\-\*\/\(\)\s\.]+', question)
                if expr:
                    tool_result = calculator_tool.invoke(expr.group().strip())
                    answer = f"🧮 Calculation result: {tool_result}"
                else:
                    answer = "Could not extract a valid mathematical expression."
            elif "web" in tool_choice or "search" in tool_choice:
                tool_result = web_search_tool.invoke(question)
                answer = f"🌐 {tool_result}"
            else:
                # Direct LLM response
                answer = llm.invoke(question).content
            
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error details: {error_msg[:200]}")
        
        # Final fallback: try each tool based on keywords
        try:
            if any(word in question.lower() for word in ["document", "file", "upload", "pdf", "text"]):
                answer = document_qa_tool.invoke(question)
            elif any(char in question for char in "+-*/") or any(word in question.lower() for word in ["calculate", "compute", "math"]):
                import re
                expr = re.search(r'[\d\+\-\*\/\(\)\s\.]+', question)
                if expr:
                    answer = f"Result: {calculator_tool.invoke(expr.group().strip())}"
                else:
                    answer = web_search_tool.invoke(question)
            else:
                answer = web_search_tool.invoke(question)
        except Exception as e2:
            answer = f"❌ Error: {str(e)[:200]}"
    
    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": answer})
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
    
    if not USE_AGENT:
        gr.Markdown("## ⚠️ **NO LLM CONFIGURED** - Please set an API key (see console)")
    
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
            chatbot = gr.Chatbot(label="Conversation", height=400, type="messages")
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
    
    clear.click(lambda: [], None, [chatbot, chat_history], queue=False)
    
    gr.Markdown("""
    ### 📝 Setup Instructions:
    1. Get a FREE API key from Groq: https://console.groq.com/keys
    2. In PowerShell: `$env:GROQ_API_KEY='your_key_here'`
    3. Restart the app: `python demo-app1.py`
    """)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7878, share=True)