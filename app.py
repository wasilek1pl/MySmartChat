import streamlit as st
import json
import re
from datetime import datetime

import config
import database
from engine import SmartChatEngine
from error_handler import safe_local_execution
from langchain_core.messages import HumanMessage, AIMessage

# --- 1. Initialization & Helpers ---
database.init_database()

@st.cache_resource
def get_engine(version="2.0"): 
    return SmartChatEngine()

engine = get_engine(version="2.0")

def reset_session(new_chat_id=None, initial_messages=None):
    """Centralized state management to keep code DRY."""
    st.session_state.update({
        "messages": initial_messages or [],
        "current_chat_id": new_chat_id,
        "detected_branches": None,
        "suggested_title": None
    })

# --- 2. Safe Backend Execution Wrappers ---
@safe_local_execution
def generate_ai_response(prompt, chat_history, chat_mode):
    """Safely executes LLM generation catching connection errors."""
    if chat_mode == "Research Mode (RAG)":
        res = engine.rag_chain.invoke({"input": prompt, "chat_history": chat_history})
        return res.get("answer", str(res)) 
    else:
        full_history = chat_history + [HumanMessage(content=prompt)]
        return engine.llm.invoke(full_history).content

@safe_local_execution
def generate_auto_title(messages):
    title_context = " ".join([m['content'] for m in messages[:8]])
    title_prompt = f"Analyze this chat. Create a short title combining high-level categories (e.g., 'History & Travel'). Keep it under 6 words. Output ONLY the title inside square brackets. Context: {title_context}"
    raw_output = engine.llm.invoke(title_prompt).content.strip()
    match = re.search(r'\[(.*?)\]', raw_output)
    clean_name = match.group(1).strip() if match else raw_output.replace("*", "").replace('Title:', '').strip()
    return re.sub(r'[^a-zA-Z0-9 &,\-]', '', clean_name).strip()

@safe_local_execution
def run_topic_extraction(chat_text, target_theme):
    return engine.extract_specific_topic(chat_text, target_theme)

@safe_local_execution
def run_auto_branching(chat_text):
    return engine.analyze_chat(chat_text)

# --- 3. State Layout ---
st.set_page_config(page_title="MySmartChat Pro", layout="wide")
if "messages" not in st.session_state: 
    reset_session()

current_chat_name = database.get_chat_title(st.session_state.current_chat_id)

# --- 4. Sidebar UI ---
with st.sidebar:
    if st.button("Start New Chat", use_container_width=True):
        reset_session()
        st.rerun()
        
    st.markdown("### Operational Mode")
    chat_mode = st.radio("Route Execution Method:", ["Standard Chat", "Research Mode (RAG)"], index=0)
    
    if st.session_state.current_chat_id:
        st.markdown("### Session Management")
        with st.expander("Rename Chat Session", expanded=False):
            custom_title = st.text_input("Edit title:", value=current_chat_name, key="rename_input")
            if st.button("Save Title", use_container_width=True) and custom_title.strip():
                database.update_chat_title(st.session_state.current_chat_id, custom_title.strip())
                st.rerun()

    with st.expander("Merge Historical Chats", expanded=False):
        all_chats_for_merge = database.get_all_chats()
        merge_options = {f"{title} ({c_id[-4:]})": c_id for c_id, title in all_chats_for_merge}
        selected_chats = st.multiselect("Select chats to blend:", options=list(merge_options.keys()))
        
        if st.button("Merge Contexts", use_container_width=True):
            if len(selected_chats) > 1:
                new_id = "merged_" + datetime.now().strftime('%Y%m%d_%H%M%S')
                ui_parts, context_parts = [], []
                
                with st.spinner("Synthesizing full chat histories..."):
                    for chat_label in selected_chats:
                        c_id = merge_options[chat_label]
                        msgs = database.load_messages_from_db(c_id)
                        full_chat_text = "\n".join([f"{m['role']}: {m['content']}" for m in msgs])
                        detailed_summary = engine.summarize_chat(full_chat_text)
                        short_summary = engine.generate_short_preview(detailed_summary)
                        
                        clean_title = chat_label.rsplit(' (', 1)[0].strip()
                        ui_parts.append(f"- **{clean_title}**: {short_summary}")
                        context_parts.append(f"--- Full Summary of [{clean_title}] ---\n{detailed_summary}")
                
                payload = {
                    "purpose": "multi_context_merge",
                    "ui_preview": "\n".join(ui_parts),
                    "detailed_context": "System Context: Resuming synthesized session.\n\n" + "\n\n".join(context_parts)
                }
                blended_text = json.dumps(payload)
                database.save_message_to_db(new_id, "user", blended_text)
                database.update_chat_title(new_id, "Synthesized Multi-Context")
                reset_session(new_chat_id=new_id, initial_messages=[{"role": "user", "content": blended_text}])
                st.rerun()
            else:
                st.warning("Please select at least 2 sessions to merge.")

    st.markdown("### Analytical Utilities")
    if len(st.session_state.messages) > 1:
        # Targeted Extraction
        st.markdown("#### Targeted Extraction")
        target_theme = st.text_input("Enter specific theme to extract:", key="manual_fork_input")
        if st.button("Fork Specific Theme", use_container_width=True) and target_theme.strip():
            chat_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
            with st.spinner(f"Extracting context regarding '{target_theme}'..."):
                extracted_summary = run_topic_extraction(chat_text, target_theme.strip())
                if extracted_summary:
                    new_id = "fork_" + datetime.now().strftime('%Y%m%d_%H%M%S')
                    pivot_msg = f"System Context extracted regarding '{target_theme.strip()}':\n\n{extracted_summary}\n\nPlease acknowledge."
                    database.save_message_to_db(new_id, "user", pivot_msg)
                    database.update_chat_title(new_id, f"Fork: {target_theme.strip()[:25]}")
                    reset_session(new_chat_id=new_id, initial_messages=[{"role": "user", "content": pivot_msg}])
                    st.rerun()

        # Auto-Detect Branches
        st.markdown("#### Auto-Detect Branches")
        if st.button("Extract Conversations", use_container_width=True):
            chat_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
            branches = run_auto_branching(chat_text)
            if branches:
                st.session_state.detected_branches = branches
            st.rerun()

    if st.session_state.detected_branches:
        st.markdown("#### Isolate Thread Context:")
        for idx, b in enumerate(st.session_state.detected_branches):
            if st.button(f"Fork: {b['topic']}", key=f"fork_{idx}"):
                new_id = "fork_" + datetime.now().strftime('%Y%m%d_%H%M%S')
                pivot_msg = f"System Context from previous session regarding {b['topic']}: {b['summary']}\n\nLet's continue discussing this."
                database.save_message_to_db(new_id, "user", pivot_msg)
                database.update_chat_title(new_id, b['topic'])
                reset_session(new_chat_id=new_id, initial_messages=[{"role": "user", "content": pivot_msg}])
                st.rerun()

    if st.session_state.current_chat_id and len(st.session_state.messages) > 0:
        if st.button("Generate Title Suggestion", use_container_width=True):
            suggested = generate_auto_title(st.session_state.messages)
            if suggested:
                st.session_state.suggested_title = " ".join(suggested.split()[:6])
            st.rerun()
        
        if st.session_state.suggested_title:
            st.info(f"Suggested: {st.session_state.suggested_title}")
            col1, col2 = st.columns(2)
            if col1.button("Confirm", type="primary", use_container_width=True):
                database.update_chat_title(st.session_state.current_chat_id, st.session_state.suggested_title)
                st.session_state.suggested_title = None
                st.rerun()
            if col2.button("Keep Old", use_container_width=True):
                st.session_state.suggested_title = None
                st.rerun()

    st.markdown("### Historical Sessions")
    for c_id, title in database.get_all_chats():
        col1, col2 = st.columns([4, 1])
        is_active = (c_id == st.session_state.current_chat_id)
        display_name = title.replace("Fork: ", "").strip()
        
        if col1.button(display_name, key=f"load_{c_id}", help=display_name, use_container_width=True, type="primary" if is_active else "secondary"):
            reset_session(new_chat_id=c_id, initial_messages=database.load_messages_from_db(c_id))
            st.rerun()
            
        if col2.button("X", key=f"del_{c_id}"):
            database.delete_chat_from_db(c_id)
            if is_active:
                reset_session()
            st.rerun()

# --- 5. Main Chat Interface ---
st.subheader(f"Session: {current_chat_name}")
st.markdown("---")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): 
        if msg["content"].startswith('{"purpose": "multi_context_merge"'):
            try:
                data = json.loads(msg["content"])
                st.info(f"**Synthesized Multi-Context Session Active**\n\n{data['ui_preview']}")
            except json.JSONDecodeError:
                st.markdown(msg["content"])
        else:
            st.markdown(msg["content"])

if prompt := st.chat_input("Enter text..."):
    if not st.session_state.current_chat_id:
        st.session_state.current_chat_id = "chat_" + datetime.now().strftime('%Y%m%d_%H%M%S')
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    database.save_message_to_db(st.session_state.current_chat_id, "user", prompt)
    
    with st.chat_message("user"): 
        st.markdown(prompt)
    
    chat_history = []
    for m in st.session_state.messages[:-1]:
        content_text = m["content"]
        if content_text.startswith('{"purpose": "multi_context_merge"'):
            try:
                content_text = json.loads(content_text)["detailed_context"]
            except json.JSONDecodeError:
                pass
        chat_history.append(HumanMessage(content=content_text) if m["role"] == "user" else AIMessage(content=content_text))
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing context and generating response..."):
            ans = generate_ai_response(prompt, chat_history, chat_mode)
            
        if ans: # Only append if the LLM didn't fail and return None
            st.session_state.messages.append({"role": "assistant", "content": ans})
            database.save_message_to_db(st.session_state.current_chat_id, "assistant", ans)
            st.markdown(ans)