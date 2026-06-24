import os

# --- Directory and Database Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "my_vectordb")
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "my_knowledge")
DUCKDB_PATH = os.path.join(BASE_DIR, "chat_history.duckdb")

# --- Model Configurations ---
LLM_MODEL = "llama3"
LLM_TEMPERATURE_CHAT = 0.7
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- RAG Retrieval Parameters ---
RETRIEVER_K = 2
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def run_startup_checks():
    """
    Ensures necessary directories exist and performs a security audit 
    for Git repository protections.
    """
    # 1. Ensure target storage directories exist
    for directory in [DB_DIR, KNOWLEDGE_DIR]:
        os.makedirs(directory, exist_ok=True)
        
    # 2. Security Audit: Check for .gitignore protections
    gitignore_path = os.path.join(BASE_DIR, ".gitignore")
    
    # Files and folders that should NEVER be pushed to GitHub
    sensitive_items = [
        ".env", 
        "my_vectordb/", 
        "chat_history.duckdb", 
        "__pycache__/"
    ]
    
    if os.path.exists(gitignore_path):
        with open(gitignore_path, "r") as f:
            content = f.read()
            missing = [item for item in sensitive_items if item not in content]
            
            if missing:
                print(f"[SECURITY WARNING] Missing from .gitignore: {', '.join(missing)}")
            else:
                print("[INFO] Git protection active (Sensitive files hidden).")
    else:
        print("[WARNING] No .gitignore file found in the project root. Please create one.")

# Execute checks automatically when config is imported
run_startup_checks()