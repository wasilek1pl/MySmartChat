# MySmartChat Pro: Local RAG Assistant

## Project Overview
This project implements a fully localized Retrieval-Augmented Generation (RAG) system optimized for macOS Apple Silicon. It is designed for complete data privacy, operating entirely offline without external API dependencies. The system leverages Ollama (Llama 3) for inference, ChromaDB for vector storage, and DuckDB for relational conversation state management. It features persistent memory, automatic semantic topic extraction, and dynamic conversation branching.

## Technical Architecture
The system follows a highly decoupled data engineering architecture:
1. **Data Ingestion:** A document crawler (`ingest.py`) processes local text files, chunks them, and stores vector embeddings (via HuggingFace) in a local ChromaDB instance.
2. **Retrieval Pipeline:** User queries route through a LangChain retrieval chain, pulling top-k relevant contexts using semantic similarity.
3. **Relational State Management:** Conversation history is strictly managed using a local DuckDB analytical database, eliminating file-system traversal risks and allowing for robust SQL querying of historical data.
4. **Offline Inference:** Language generation is handled locally via Ollama, ensuring zero data egress.
5. **Topic Extraction & Branching:** An analytical extraction protocol uses strict Regex multi-line parsing to identify topic shifts within the conversation log, allowing users to fork contexts into new, isolated threads.

## Technical Stack
- **Language:** Python 3.10+
- **Frontend UI:** Streamlit
- **Orchestration:** LangChain
- **LLM Engine:** Ollama (model: `llama3`)
- **Vector Database:** ChromaDB
- **Embedding Model:** HuggingFace (`all-MiniLM-L6-v2`)
- **Relational Database:** DuckDB
- **OS:** macOS (Optimized for Apple Silicon M-series)

## Installation and Setup

1. **Environment Preparation**
   Clone the repository and initialize a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate

## Backlog & To-Do List

- **Research Mode Validation:** Conduct comprehensive testing of the RAG pipeline (`Research Mode`) against the ChromaDB vector store to verify accurate document retrieval and semantic matching.
- **Multi-Format Document Ingestion:** Expand the `ingest.py` parsing logic to handle structured formats, specifically targeting `.csv` and `.pdf` files, and tokenize them for the vector database.
- **Asynchronous Multithreading for UI:** Implement asynchronous UI streaming (e.g., `st.write_stream`) to render local LLM tokens to the Streamlit frontend in real-time, reducing perceived latency.
- **Dynamic Context Window Management:** Implement programmatic token counting algorithms to safely truncate historical context before reaching the model's memory threshold during long sessions.
- **DuckDB Analytics Dashboard:** Build a supplementary Streamlit dashboard designed to query the local `.duckdb` instance and display user metrics, frequently queried technical terms, and session length analytics.
