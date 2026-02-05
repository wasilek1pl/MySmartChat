import os
import sys
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from config import setup_ai 

# --- 1. SECURITY & SETUP ---
# We call setup_ai to:
#   a) Load .env variables
#   b) Run the .gitignore security check
#   c) Verify the API key exists
# We set test_connection=False because we don't need the Chat Client here.
_ = setup_ai(test_connection=False)

# Now we can safely get the key (setup_ai already verified it exists)
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    # Double check just in case
    print("[CRITICAL] GEMINI_API_KEY missing. Exiting.")
    sys.exit(1)

# --- 2. DEFINE TOOLS ---
# The Librarian: Turns text into math vectors
# We use a SPECIFIC model for embeddings, distinct from the chat model in config.py
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", 
    google_api_key=api_key
)

# The Slicer: Breaks long notes into 500-character chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# --- 3. THE ETL PIPELINE ---
data_folder = "my_knowledge"
all_documents = []

print(f"\n[INFO] Starting ingestion from folder: '{data_folder}'...")

if not os.path.exists(data_folder):
    print(f"[ERROR] Folder '{data_folder}' not found. Please create it.")
    sys.exit(1)

# Step A: EXTRACT & TRANSFORM
for filename in os.listdir(data_folder):
    if filename.endswith(".txt") or filename.endswith(".md"):
        full_path = os.path.join(data_folder, filename)
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Create labeled chunks
                metadata = {"source": filename}
                doc_chunks = text_splitter.create_documents(
                    texts=[content], 
                    metadatas=[metadata]
                )
                
                all_documents.extend(doc_chunks)
                print(f"Processed {filename}: {len(doc_chunks)} chunks created.")
                
        except Exception as e:
            print(f"Error reading {filename}: {e}")

# Step B: LOAD
if not all_documents:
    print("\n[WARNING] No documents found to ingest.")
    sys.exit(0)

print(f"\n[INFO] Loading {len(all_documents)} total chunks into ChromaDB...")

try:
    # This creates the database folder
    vectorstore = Chroma.from_documents(
        documents=all_documents,
        embedding=embeddings,
        persist_directory="./my_vectordb"
    )
    print("--- SUCCESS ---")
    print("Database created and saved in './my_vectordb'")

except Exception as e:
    print(f"[ERROR] Failed to save database: {e}")