import streamlit as st
import os
import re
from datetime import datetime
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 1. Global Helpers ---
def append_to_log(filepath, role, content):
    """Safely appends a message to the active chat log."""
    if not filepath: return
    os.makedirs("chat_logs", exist_ok=True)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(f"{'You' if role == 'user' else 'AI'}: {content}\n")

def load_chat_history(path):
    """Bulletproof multi-line parser for chat logs."""
    msgs = []
    if not os.path.exists(path): return msgs
    
    with open(path, 'r', encoding='utf-8') as file:
        content = file.read()
        
    parts = re.split(r'(?m)^(You:\s|AI:\s)', content)
    current_role = None
    current_text = ""
    
    for part in parts:
        if part == "You: ":
            if current_role: msgs.append({"role": current_role, "content": current_text.strip()})
            current_role = "user"
            current_text = ""
        elif part == "AI: ":
            if current_role: msgs.append({"role": current_role, "content": current_text.strip()})
            current_role = "assistant"
            current_text = ""
        else:
            if current_role: current_text += part
            
    if current_role and current_text.strip():
        msgs.append({"role": current_role, "content": current_text.strip()})
        
    return msgs

# --- 2. Backend Engine ---
class SmartChatEngine:
    def __init__(self):
        self.llm = ChatOllama(model="llama3", temperature=0.7)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = Chroma(persist_directory="./my_vectordb", embedding_function=self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "Use context:\n{context}"), ("human", "{input}")
        ])
        self.rag_chain = create_retrieval_chain(
            self.retriever, create_stuff_documents_chain(self.llm, self.prompt)
        )

    def analyze_chat(self, chat_text):
        """Lets the LLM speak naturally, then uses Python to structure the data."""
        # Very relaxed prompt - no JSON or XML required.
        prompt = f"""Review this chat and identify 3 distinct topics discussed. 
        Write them as a simple list where each item looks exactly like this:
        Topic: [Title]
        Summary: [One sentence description]
        
        Chat:
        {chat_text}"""
        
        res = self.llm.invoke(prompt)
        content = res.content
        
        branches = []
        current_topic = "General Chat"
        
        # Pure Python parsing: tolerant and crash-proof
        for line in content.split('\n'):
            line = line.strip()
            if line.lower().startswith("topic:"):
                current_topic = line.split(":", 1)[1].strip().replace('*', '')
            elif line.lower().startswith("summary:"):
                summary = line.split(":", 1)[1].strip().replace('*', '')
                branches.append({'topic': current_topic, 'summary': summary})
                
        # If the LLM completely ignored the format, return a safe default instead of crashing
        if not branches:
            branches.append({'topic': "Conversation Summary", 'summary': content[:150] + "..."})
            
        return branches[:3]

@st.cache_resource
def get_engine(): return SmartChatEngine()
engine = get_engine()

# --- 3. UI and State Management ---
st.set_page_config(page_title="MySmartChat Pro", layout="wide")

if "messages" not in st.session_state: st.session_state.messages = []
if "current_file" not in st.session_state: st.session_state.current_file = None
if "detected_branches" not in st.session_state: st.session_state.detected_branches = None
if "suggested_title" not in st.session_state: st.session_state.suggested_title = None

# --- 4. Sidebar Features ---
with st.sidebar:
    if st.button("Start New Chat", use_container_width=True):
        st.session_state.update({"messages": [], "current_file": None, "detected_branches": None, "suggested_title": None})
        st.rerun()
    
    st.markdown("### Analysis & Tools")
    
    # 1. Topic Extraction
    if len(st.session_state.messages) > 2:
        if st.button("Analyze Topics", use_container_width=True):
            chat_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
            try:
                st.session_state.detected_branches = engine.analyze_chat(chat_text)
            except Exception as e:
                st.error("Extraction failed, but the app is safe.")
            st.rerun()

    if st.session_state.detected_branches:
        st.markdown("#### Select Topic to Fork:")
        for idx, b in enumerate(st.session_state.detected_branches):
            if st.button(f"Fork: {b['topic']}", key=f"fork_btn_{idx}"):
                safe_name = re.sub(r'[^a-zA-Z0-9_]', '', b['topic'].replace(' ', '_'))
                new_f = os.path.join("chat_logs", f"branch_{safe_name}_{datetime.now().strftime('%H%M%S')}.txt")
                append_to_log(new_f, "user", f"Context: {b['summary']}")
                st.session_state.update({
                    "current_file": new_f, 
                    "messages": [{"role": "user", "content": f"Context: {b['summary']}"}], 
                    "detected_branches": None,
                    "suggested_title": None
                })
                st.rerun()

    # 2. Rename Workflow (Preview -> Confirm OR Cancel)
    if st.session_state.current_file and len(st.session_state.messages) > 0:
        st.markdown("----")
        if st.button("Generate Name Suggestion", use_container_width=True):
            title_context = " ".join([m['content'] for m in st.session_state.messages[:3]])
            
            # Strict prompt forcing bracketed output
            title_prompt = f"""Provide a 3-word title for this chat. 
            You MUST output ONLY the title inside square brackets, like [Data Analysis Project]. 
            Do not add any other text. 
            Chat: {title_context}"""
            
            raw_output = engine.llm.invoke(title_prompt).content
            
            # Extract only the text inside the brackets
            match = re.search(r'\[(.*?)\]', raw_output)
            if match:
                raw_name = match.group(1).strip()
            else:
                # Fallback: forcefully strip filler words and keep only the first 3 words
                stripped = re.sub(r'(?i)(here is|sure|i can|the title|note:)', '', raw_output).strip()
                raw_name = "_".join(stripped.split()[:3])
            
            # Sanitize the final string for the file system
            clean_name = re.sub(r'[^a-zA-Z0-9]', '_', raw_name)
            clean_name = re.sub(r'_+', '_', clean_name).strip('_') # Remove double underscores
            
            st.session_state.suggested_title = f"{clean_name}_{datetime.now().strftime('%H%M')}"
            st.rerun()
        
        if st.session_state.suggested_title:
            st.info(f"Preview: {st.session_state.suggested_title}.txt")
            
            col1, col2 = st.columns(2)
            if col1.button("Confirm", type="primary", use_container_width=True):
                new_path = os.path.join("chat_logs", f"{st.session_state.suggested_title}.txt")
                try:
                    os.rename(st.session_state.current_file, new_path)
                    st.session_state.current_file = new_path
                    st.session_state.suggested_title = None
                    st.rerun()
                except OSError as e:
                    st.error(f"Rename failed: {e}")
            
            if col2.button("Keep Old", use_container_width=True):
                st.session_state.suggested_title = None
                st.rerun()

    # 3. Saved Sessions & Deletion
    st.markdown("### Saved Sessions")
    os.makedirs("chat_logs", exist_ok=True)
    log_files = [f for f in os.listdir("chat_logs") if f.endswith('.txt')]
    
    for f in sorted(log_files, reverse=True):
        col1, col2 = st.columns([4, 1])
        path = os.path.join("chat_logs", f)
        
        if col1.button(f"{f[:15]}...", key=f"load_{f}", help=f):
            st.session_state.update({
                "messages": load_chat_history(path), 
                "current_file": path, 
                "detected_branches": None,
                "suggested_title": None
            })
            st.rerun()
            
        if col2.button("X", key=f"del_{f}"):
            if os.path.exists(path):
                os.remove(path)
            if st.session_state.current_file == path:
                st.session_state.update({"messages": [], "current_file": None, "detected_branches": None, "suggested_title": None})
            st.rerun()

# --- 5. Main Chat Interface ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("Type..."):
    if not st.session_state.current_file:
        st.session_state.current_file = os.path.join("chat_logs", f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    append_to_log(st.session_state.current_file, "user", prompt)
    with st.chat_message("user"): st.markdown(prompt)
    
    with st.chat_message("assistant"):
        res = engine.rag_chain.invoke({"input": prompt})
        ans = res.get("answer", str(res)) 
        
        st.session_state.messages.append({"role": "assistant", "content": ans})
        append_to_log(st.session_state.current_file, "assistant", ans)
        st.markdown(ans)