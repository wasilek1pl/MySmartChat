# Local RAG Assistant with ChromaDB and Gemini

## Project Overview
This project implements a Retrieval-Augmented Generation (RAG) system designed to run locally on macOS. It leverages Google Gemini 1.5 Flash for language generation and ChromaDB as a vector store for persistent document memory. The system is designed to provide contextually aware responses by retrieving relevant information from a local knowledge base before querying the LLM.

## Technical Architecture
The system follows a standard RAG pipeline architecture:
1. Data Ingestion: A file-system crawler identifies and processes local text files.
2. Vectorization: Document chunks are converted into 768-dimensional embeddings.
3. Storage: Vectors and associated metadata are stored in a persistent ChromaDB instance.
4. Retrieval: User queries are vectorized and matched against the database using semantic similarity.
5. Generation: Retrieved context is injected into a system prompt for grounded AI responses.

## Project Roadmap

### Stage 1: Environment and Connectivity
- Configure Python 3.12 environment on Apple Silicon (M4).
- Secure Gemini API integration using environment variables.
- Validate basic text generation and API response latency.

### Stage 2: Similarity Logic
- Implementation of manual cosine similarity to validate embedding models.
- Prototype of the primary knowledge categories: Technical (C++/Python), Culinary, and Travel.
- Development of logic for handling unstructured text chunks.

### Stage 3: Persistent Memory (In Progress)
- File System Integration: Automated scanning of ./my_knowledge directory for new data.
- Metadata Engineering: Automatic extraction of category tags and source attributes from filenames.
- Persistence: Migration from in-memory lists to a disk-persistent PersistentClient in ChromaDB.

### Stage 4: Advanced Interaction
- Contextual Branching: Logic to detect topic shifts and suggest new chat sessions.
- Auto-Naming: Dynamic chat titles generated from the top-k retrieved metadata.
- System Prompt Optimization: Refinement of instructions to ensure high-fidelity retrieval usage.

## Technical Stack
- Language: Python 3.12+
- LLM: Google Gemini 1.5 Flash
- Vector Database: ChromaDB
- Embedding Model: Google Text-Embedding-004
- OS: macOS (Optimized for Apple Silicon M-series)

## Installation and Setup
1. Clone the repository to your local machine.
2. Create a virtual environment: python -m venv .venv
3. Activate the environment: source .venv/bin/activate
4. Install dependencies: pip install -r requirements.txt
5. Configure your .env file with your GOOGLE_API_KEY.
6. Place your text notes in the /my_knowledge folder for ingestion.

## Directory Structure
- /my_knowledge: Source text files for the knowledge base.
- /my_vectordb: Local persistent storage for ChromaDB (git-ignored).
- app.py: Main application entry point.
- ingest.py: Script for processing and uploading files to the vector store.
- .env: API keys and environment configuration (git-ignored).
- requirements.txt: List of required Python packages.