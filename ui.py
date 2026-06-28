import streamlit as st
import os
import re
import uuid
import duckdb
from datetime import datetime
import json
import config

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- 1. Database Initialization ---
def init_database():
    """Initializes DuckDB tables with explicit cursors for state stability."""
    with duckdb.connect(config.DUCKDB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                chat_id VARCHAR PRIMARY KEY,
                title VARCHAR,
                created_at TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                message_id VARCHAR PRIMARY KEY,
                chat_id VARCHAR,
                role VARCHAR,
                content VARCHAR,
                timestamp TIMESTAMP
            )
        """)
        conn.commit()

init_database()

# --- 2. Database Transaction Layer ---
def save_message_to_db(chat_id, role, content):
    """Logs individual message entries using isolated cursor memory locks."""
    with duckdb.connect(config.DUCKDB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM chats WHERE chat_id = ?", [chat_id])
        if not cursor.fetchone():
            cursor.execute("INSERT INTO chats VALUES (?, ?, ?)", [chat_id, "General Discussion", datetime.now()])
        
        msg_id = str(uuid.uuid4())
        cursor.execute("INSERT INTO messages VALUES (?, ?, ?, ?, ?)", [msg_id, chat_id, role, content, datetime.now()])
        conn.commit()

def load_messages_from_db(chat_id):
    """Retrieves chronological array of message dictionaries matching a session token."""
    with duckdb.connect(config.DUCKDB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT role, content FROM messages WHERE chat_id = ? ORDER BY timestamp ASC", [chat_id])
        rows = cursor.fetchall()
        return [{"role": row[0], "content": row[1]} for row in rows] if rows else []

def update_chat_title(chat_id, new_title):
    """Commits title string updates to the target session row."""
    with duckdb.connect(config.DUCKDB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE chats SET title = ? WHERE chat_id = ?", [new_title, chat_id])
        conn.commit()

def get_chat_title(chat_id):
    """Fetches descriptive title for active container presentation safely."""
    if not chat_id:
        return "New Chat Session"
    try:
        with duckdb.connect(config.DUCKDB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT title FROM chats WHERE chat_id = ?", [chat_id])
            row = cursor.fetchone()
            return row[0] if row else "General Discussion"
    except Exception:
        return "General Discussion"

def delete_chat_from_db(chat_id):
    """Purges historical dependencies and parent records from database files."""
    with duckdb.connect(config.DUCKDB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages WHERE chat_id = ?", [chat_id])
        cursor.execute("DELETE FROM chats WHERE chat_id = ?", [chat_id])
        conn.commit()

# --- 3. Backend Model Engine ---
class SmartChatEngine:
    """Manages local model configuration, semantic retrievers, and parsing pipelines."""
    def __init__(self):
        self.llm = ChatOllama(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE_CHAT)
        self.embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
        
        self.vectorstore = Chroma(persist_directory=config.DB_DIR, embedding_function=self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": config.RETRIEVER_K})
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a data infrastructure assistant. Answer the query using the provided context logs. If the context does not contain the answer, use your local knowledge base.\n\nContext:\n{context}"), 
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        self.rag_chain = create_retrieval_chain(
            self.retriever, create_stuff_documents_chain(self.llm, self.prompt)
        )

    def analyze_chat(self, chat_text):
        """Generates thematic splits using resilient text segmentation logic."""
        prompt = f"""Read the following conversation log. Identify up to 3 distinct technical concepts or themes discussed.
        Write your answer in plain text. For each concept, use exactly this format on new lines:
        Topic: [Short Name]
        Summary: [One sentence description]
        
        Log:\n{chat_text}"""
        
        content = self.llm.invoke(prompt).content
        branches = []
        current_topic = None
        
        for line in content.split('\n'):
            clean_line = line.replace("*", "").strip()
            if clean_line.lower().startswith("topic:"):
                raw_topic = clean_line.split(":", 1)[1].strip()
                current_topic = re.sub(r'[^a-zA-Z0-9 ]', '', raw_topic).strip()
            elif clean_line.lower().startswith("summary:") and current_topic:
                summary = clean_line.split(":", 1)[1].strip()
                branches.append({'topic': current_topic[:35], 'summary': summary})
                current_topic = None
                
        if not branches:
            branches.append({'topic': "Consistent Theme", 'summary': "The conversation maintains a single consistent context without major deviations."})
            
        return branches[:3]

    def summarize_chat(self, chat_text):
        """Generates a dense, comprehensive summary of an entire conversation log for map-reduce processing."""
        prompt = f"""Review the following complete conversation history. Provide a dense, comprehensive technical summary that captures all key concepts, decisions, and data structures discussed throughout the entire log. Avoid generic filler or introductory text.
        
        Log:\n{chat_text}"""
        return self.llm.invoke(prompt).content.strip()

    def generate_short_preview(self, detailed_summary):
        """Compresses a detailed technical summary into a single plain-text sentence for the UI."""
        prompt = f"Summarize the following text into exactly one short, concise sentence. Do not include introductory filler. Output only the plain text sentence.\n\nText:\n{detailed_summary}"
        return self.llm.invoke(prompt).content.strip()

@st.cache_resource
def get_engine(version="2.0"): 
    return SmartChatEngine()

engine = get_engine(version="2.0")

# --- 4. Streamlit UI and State Layout ---
st.set_page_config(page_title="MySmartChat Pro", layout="wide")
for key in ["messages", "current_chat_id", "detected_branches", "suggested_title"]:
    if key not in st.session_state: 
        st.session_state[key] = [] if key == "messages" else None

current_chat_name = get_chat_title(st.session_state.current_chat_id)

with st.sidebar:
    if st.button("Start New Chat", use_container_width=True):
        st.session_state.update({"messages": [], "current_chat_id": None, "detected_branches": None, "suggested_title": None})
        st.rerun()
        
    st.markdown("### Operational Mode")
    chat_mode = st.radio("Route Execution Method:", ["Standard Chat", "Research Mode (RAG)"], index=0)
    
    # --- Session Management & Merging Interface ---
    if st.session_state.current_chat_id:
        st.markdown("### Session Management")
        with st.expander("Rename Chat Session", expanded=False):
            custom_title = st.text_input(
                "Edit title:", 
                value=current_chat_name, 
                key=f"rename_input_{st.session_state.current_chat_id}"
            )
            if st.button("Save Title", use_container_width=True):
                if custom_title.strip():
                    update_chat_title(st.session_state.current_chat_id, custom_title.strip())
                    st.rerun()

    with st.expander("Merge Historical Chats", expanded=False):
        with duckdb.connect(config.DUCKDB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT chat_id, title FROM chats ORDER BY created_at DESC")
            all_chats_for_merge = cursor.fetchall() or []
        
        merge_options = {f"{title} ({c_id[-4:]})": c_id for c_id, title in all_chats_for_merge}
        selected_chats = st.multiselect("Select chats to blend:", options=list(merge_options.keys()))
        
        if st.button("Merge Contexts", use_container_width=True):
            if len(selected_chats) > 1:
                new_id = "merged_" + datetime.now().strftime('%Y%m%d_%H%M%S')
                
                ui_parts = []
                context_parts = []
                
                with st.spinner("Synthesizing full chat histories (processing two plain-text passes)..."):
                    for chat_label in selected_chats:
                        c_id = merge_options[chat_label]
                        msgs = load_messages_from_db(c_id)
                        
                        full_chat_text = "\n".join([f"{m['role']}: {m['content']}" for m in msgs])
                        
                        detailed_summary = engine.summarize_chat(full_chat_text)
                        short_summary = engine.generate_short_preview(detailed_summary)
                        
                        clean_title = chat_label.rsplit(' (', 1)[0].strip()
                        
                        ui_parts.append(f"- **{clean_title}**: {short_summary}")
                        context_parts.append(f"--- Full Summary of [{clean_title}] ---\n{detailed_summary}")
                
                payload = {
                    "purpose": "multi_context_merge",
                    "ui_preview": "\n".join(ui_parts),
                    "detailed_context": "System Context: You are resuming a newly synthesized session built by blending the full contexts of multiple previous conversations. Here are the comprehensive summaries:\n\n" + "\n\n".join(context_parts) + "\n\nPlease analyze how these distinct domains intersect, look for structural patterns, and ask me how we should begin integrating them."
                }
                blended_text = json.dumps(payload)
                
                save_message_to_db(new_id, "user", blended_text)
                update_chat_title(new_id, "Synthesized Multi-Context")
                
                st.session_state.update({
                    "current_chat_id": new_id, 
                    "messages": [{"role": "user", "content": blended_text}], 
                    "detected_branches": None, 
                    "suggested_title": None
                })
                st.rerun()
            else:
                st.warning("Please select at least 2 sessions to merge.")

    # --- Analytical Utilities ---
    st.markdown("### Analytical Utilities")
    if len(st.session_state.messages) > 1:
        if st.button("Extract Conversations", use_container_width=True):
            chat_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
            try: 
                st.session_state.detected_branches = engine.analyze_chat(chat_text)
            except: 
                st.error("Analytical extraction halted.")
            st.rerun()

    if st.session_state.detected_branches:
        st.markdown("#### Isolate Thread Context:")
        for idx, b in enumerate(st.session_state.detected_branches):
            if st.button(f"Fork: {b['topic']}", key=f"fork_{idx}"):
                new_id = "fork_" + datetime.now().strftime('%Y%m%d_%H%M%S')
                
                pivot_msg = f"System Context from previous session regarding {b['topic']}: {b['summary']}\n\nLet's continue discussing this."
                save_message_to_db(new_id, "user", pivot_msg)
                
                update_chat_title(new_id, b['topic'])
                
                st.session_state.update({
                    "current_chat_id": new_id, 
                    "messages": [{"role": "user", "content": pivot_msg}], 
                    "detected_branches": None, 
                    "suggested_title": None
                })
                st.rerun()

    # --- Context-Aware Title Suggestion ---
    if st.session_state.current_chat_id and len(st.session_state.messages) > 0:
        if st.button("Generate Title Suggestion", use_container_width=True):
            title_context = " ".join([m['content'] for m in st.session_state.messages[:8]])
            
            title_prompt = (
                f"Analyze this chat. Identify the main distinct topics discussed. "
                f"Create a short title that captures them by combining their high-level categories "
                f"(e.g., 'History, Travel & Local Cuisine' or 'Python, RAG & Databases'). "
                f"Keep it under 6 words. Output ONLY the title inside square brackets. Context: {title_context}"
            )
            raw_output = engine.llm.invoke(title_prompt).content.strip()
            
            match = re.search(r'\[(.*?)\]', raw_output)
            if match:
                clean_name = match.group(1).strip()
            else:
                clean_name = raw_output.replace("*", "").replace('Title:', '').strip()
                
            clean_name = re.sub(r'[^a-zA-Z0-9 &,\-]', '', clean_name).strip()
            
            st.session_state.suggested_title = " ".join(clean_name.split()[:6])
            st.rerun()
        
        if st.session_state.suggested_title:
            st.info(f"Suggested: {st.session_state.suggested_title}")
            col1, col2 = st.columns(2)
            if col1.button("Confirm", type="primary", use_container_width=True):
                update_chat_title(st.session_state.current_chat_id, st.session_state.suggested_title)
                st.session_state.suggested_title = None
                st.rerun()
            if col2.button("Keep Old", use_container_width=True):
                st.session_state.suggested_title = None
                st.rerun()

    # --- Historical Sessions ---
    st.markdown("### Historical Sessions")
    with duckdb.connect(config.DUCKDB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT chat_id, title FROM chats ORDER BY created_at DESC")
        saved_chats = cursor.fetchall() or []
    
    for c_id, title in saved_chats:
        col1, col2 = st.columns([4, 1])
        
        is_active = (c_id == st.session_state.current_chat_id)
        display_name = title.replace("Fork: ", "").strip()
        btn_type = "primary" if is_active else "secondary"
        
        if col1.button(display_name, key=f"load_{c_id}", help=display_name, use_container_width=True, type=btn_type):
            st.session_state.update({
                "messages": load_messages_from_db(c_id), 
                "current_chat_id": c_id, 
                "detected_branches": None, 
                "suggested_title": None
            })
            st.rerun()
            
        if col2.button("X", key=f"del_{c_id}"):
            delete_chat_from_db(c_id)
            if st.session_state.current_chat_id == c_id:
                st.session_state.update({"messages": [], "current_chat_id": None, "detected_branches": None, "suggested_title": None})
            st.rerun()

# --- 5. Execution Interface ---
st.subheader(f"Session: {current_chat_name}")
st.markdown("---")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): 
        if msg["content"].startswith('{"purpose": "multi_context_merge"'):
            try:
                data = json.loads(msg["content"])
                st.info(f"**Synthesized Multi-Context Session Active**\n\nThe following summary outlines the merged domains:\n\n{data['ui_preview']}")
            except json.JSONDecodeError:
                st.markdown(msg["content"])
        else:
            st.markdown(msg["content"])

if prompt := st.chat_input("Enter text..."):
    if not st.session_state.current_chat_id:
        st.session_state.current_chat_id = "chat_" + datetime.now().strftime('%Y%m%d_%H%M%S')
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_message_to_db(st.session_state.current_chat_id, "user", prompt)
    with st.chat_message("user"): 
        st.markdown(prompt)
    
    chat_history = []
    for m in st.session_state.messages[:-1]:
        content_text = m["content"]
        if content_text.startswith('{"purpose": "multi_context_merge"'):
            try:
                data = json.loads(content_text)
                content_text = data["detailed_context"]
            except json.JSONDecodeError:
                pass
        
        if m["role"] == "user":
            chat_history.append(HumanMessage(content=content_text))
        else:
            chat_history.append(AIMessage(content=content_text))
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing context and generating response..."):
            if chat_mode == "Research Mode (RAG)":
                res = engine.rag_chain.invoke({
                    "input": prompt,
                    "chat_history": chat_history
                })
                ans = res.get("answer", str(res)) 
            else:
                full_history = chat_history + [HumanMessage(content=prompt)]
                ans = engine.llm.invoke(full_history).content
            
        st.session_state.messages.append({"role": "assistant", "content": ans})
        save_message_to_db(st.session_state.current_chat_id, "assistant", ans)
        st.markdown(ans)