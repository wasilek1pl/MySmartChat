import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# 1. Connect to the Librarian
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# 2. Open the existing database (don't create a new one!)
db = Chroma(persist_directory="./my_vectordb", embedding_function=embeddings)

# 3. Ask a question
query = "What do I need to buy for a Danish apartment?"
results = db.similarity_search(query, k=2)

print("\n--- DATABASE SEARCH RESULTS ---")
for doc in results:
    print(f"\n[Source: {doc.metadata['source']}]")
    print(f"Content: {doc.page_content}")