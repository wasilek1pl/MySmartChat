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

1. **Environment Preparation:** Clone the repository and initialize a Python virtual environment by running `python -m venv .venv` followed by `source .venv/bin/activate`.
2. **Dependency Installation:** Install the required Python packages into your isolated environment by executing `pip install -r requirements.txt`.
3. **Local LLM Provisioning:** Ensure Ollama is installed on your macOS system, then pull the foundational inference model by running `ollama pull llama3`.
4. **Security & Configuration Audit:** Run the configuration script via `python config.py` to initialize local storage directories and verify `.gitignore` protections.
5. **Knowledge Base Ingestion:** Place your source `.txt` files into the `my_knowledge/` directory, then build the initial ChromaDB vector index by executing `python ingest.py`.
6. **Application Launch:** Start the Streamlit frontend interface and relational database engine by running `streamlit run ui.py`.

## Backlog & To-Do List

## Active Development
- **Modular Architecture Refactoring:** Decoupling the monolithic Streamlit script into distinct system modules (`config.py`, `database.py`, `engine.py`, and `app.py`) for better separation of concerns.
- **Targeted Topic Branching:** Implementing semantic LLM filtering to extract specific themes from massive conversation logs and fork them into clean, isolated chat sessions.
- **Multi-Format Document Ingestion:** Expanding `ingest.py` to parse and tokenize structured formats (`.csv` and `.pdf`) into the ChromaDB vector database.

## Future Backlog
- **System Observability & Logging:** Implement Python's native `logging` module to track latency per inference, token counts per session, and gracefully capture JSON decoding exceptions into a local `app.log` file.
- **Asynchronous Multithreading:** Implement UI streaming using asynchronous generators to stream local LLM tokens to the Streamlit frontend, reducing perceived latency.
- **Dynamic Context Window Management:** Implement programmatic token counting algorithms to safely truncate historical context before reaching the model's memory threshold.
- **LLM Evaluation Pipeline:** Build an evaluation protocol using the "RAG Triad" (Context Relevance, Groundedness, Answer Relevance) to programmatically measure the accuracy of local model outputs.
- **Research Mode Validation:** Conduct comprehensive testing of the RAG pipeline (`Research Mode`) against the ChromaDB vector store to verify accurate document retrieval.
