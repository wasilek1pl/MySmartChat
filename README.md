# MySmartChat Pro: Local AI & Data Assistant

## Project Overview
This project is a fully local AI chat assistant built as a hands-on data engineering project. It is optimized for macOS Apple Silicon and designed for complete data privacy—everything runs offline on your machine with no external APIs. 

Instead of just a simple script, this is built with a decoupled data architecture. It uses **Ollama (Llama 3)** to generate text, **ChromaDB** to search through local documents, and **DuckDB** to safely save your chat history. 

## How It Works (System Architecture)
The code is broken down into separate, modular files so the frontend, backend, and databases don't get tangled up:
1. **Frontend (`app.py`):** The Streamlit user interface. It only handles what the user sees and clicks.
2. **The Brain (`engine.py`):** Uses LangChain to connect to the local Llama 3 model. It handles the logic for chatting, summarizing, and searching.
3. **Relational Database (`database.py`):** Saves all chat history locally using DuckDB. It uses explicit SQL "cursors" to safely write data without locking up or crashing the system.
4. **Data Ingestion (`ingest.py`):** A background script that reads local text files, chops them up, and saves them into ChromaDB so the AI can search through them later.
5. **System Safety (`config.py` & `error_handler.py`):** Centralized settings and a custom error catcher. If a database fails or Ollama is turned off, the app logs the error to a hidden `app.log` file instead of crashing the screen.

## Key Features
* **Standard Chat & Research Mode:** Chat normally, or ask the AI to search your local documents for answers (RAG).
* **Topic Extraction & Branching:** The AI can read a long chat, find specific topics, and "fork" them into a brand new, focused chat session.
* **Context Merging:** Select multiple historical chats and merge them together. The system creates a summarized "master context" so you can talk about all of them at once.
* **Auto-Renaming:** The system automatically generates short, clean titles for your chats based on the conversation context.
* **Crash Protection:** Background logging and error handlers keep the application stable.

## Technical Stack
- **Language:** Python 3.10+
- **Frontend UI:** Streamlit
- **Orchestration:** LangChain
- **LLM Engine:** Ollama (model: `llama3`)
- **Vector Database:** ChromaDB
- **Embedding Model:** HuggingFace (`all-MiniLM-L6-v2`)
- **Relational Database:** DuckDB

## Installation and Setup

1. **Environment Preparation:** Clone the repository and initialize a Python virtual environment by running `python -m venv .venv` followed by `source .venv/bin/activate`.
2. **Dependency Installation:** Install the required Python packages into your isolated environment by executing `pip install -r requirements.txt`.
3. **Local LLM Provisioning:** Ensure Ollama is installed on your macOS system, then pull the foundational inference model by running `ollama pull llama3`.
4. **Security & Configuration Audit:** Run the configuration script via `python config.py` to initialize local storage directories and verify `.gitignore` protections.
5. **Knowledge Base Ingestion:** Place your source `.txt` files into the `my_knowledge/` directory, then build the initial ChromaDB vector index by executing `python ingest.py`.
6. **Application Launch:** Start the Streamlit frontend interface and relational database engine by running `streamlit run ui.py`.

## Next on the Roadmap (Future Backlog)

* **Multi-Format Documents:** Update `ingest.py` so it can read PDFs and CSV files, not just text files.
* **Asynchronous Multithreading:** Stream the AI's text to the screen word-by-word so it feels faster.
* **Dynamic Context Limits:** Add a system to count words/tokens so the AI doesn't get overwhelmed and crash if a chat gets too long.
* **LLM Evaluation Pipeline:** Build a testing system to automatically score how accurate the AI's answers are.
* **Research Mode Validation:** Run detailed tests on the ChromaDB search to make sure it pulls the correct document paragraphs.
