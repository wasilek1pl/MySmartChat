import os
import config
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def build_vector_database():
    print(f"Scanning directory: {config.KNOWLEDGE_DIR}")
    
    loader = DirectoryLoader(config.KNOWLEDGE_DIR, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()

    if not documents:
        print("No source documents detected inside the knowledge directory.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)
    
    print(f"Divided {len(documents)} documents into {len(chunks)} text chunks.")
    print("Computing vector embeddings and updating ChromaDB store...")

    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=config.DB_DIR
    )
    
    print(f"Vector store generation completed successfully at: {config.DB_DIR}")

if __name__ == "__main__":
    build_vector_database()