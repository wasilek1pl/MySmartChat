import os
import logging

# --- Directory and Database Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "my_vectordb")
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "my_knowledge")
DUCKDB_PATH = os.path.join(BASE_DIR, "chat_history.duckdb")

# --- Logging Setup ---
LOG_FILE = os.path.join(BASE_DIR, "app.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE), # Writes to the log file
        logging.StreamHandler()        # Prints to the terminal
    ]
)

# --- Model Configurations ---
LLM_MODEL = "llama3"
LLM_TEMPERATURE_CHAT = 0.7
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- RAG Retrieval Parameters ---
RETRIEVER_K = 2
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def run_startup_checks():
    logger = logging.getLogger("Startup")
    for directory in [DB_DIR, KNOWLEDGE_DIR]:
        os.makedirs(directory, exist_ok=True)
        
    gitignore_path = os.path.join(BASE_DIR, ".gitignore")
    sensitive_items = [".env", "my_vectordb/", "chat_history.duckdb", "__pycache__/"]
    
    if os.path.exists(gitignore_path):
        with open(gitignore_path, "r") as f:
            content = f.read()
            missing = [item for item in sensitive_items if item not in content]
            if missing:
                logger.warning(f"Missing from .gitignore: {', '.join(missing)}")
            else:
                logger.info("Git protection active (Sensitive files hidden).")
    else:
        logger.warning("No .gitignore file found in the project root.")

run_startup_checks()