import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
import json

# 1. Load Keys
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# 2. Connect to the Memory (The exact same code from the test)
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=api_key)
vectorstore = Chroma(persist_directory="./my_vectordb", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. Connect to the Voice 
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", google_api_key=api_key)

# 4. Give the AI its instructions
system_prompt = (
    "You are a highly intelligent assistant with access to the user's personal notes. "
    "Context from the user's notes: {context}\n\n"
    "Instructions: "
    "1. If the user's question can be answered using the provided context, prioritize that information. "
    "2. If the context does not contain the answer, explicitly state 'I don't have this in your notes, but...' and then answer the question fully using your general knowledge."
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 5. Connect the Memory and the Voice together
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 6. The Chat Loop
print("\n" + "="*40)
print(" MySmartChat is Online ")
print(" Type 'quit' to exit.")
print("="*40 + "\n")

# Create a unique filename based on the current date and time
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f"chat_logs/chat_{timestamp}.txt"

print(f"[SYSTEM] Session logging to: {log_filename}")

# Add a flag to track the first message
is_first_message = True
message_counter = 0

def extract_chat_branches(log_filepath):
    print("\n[SYSTEM] Analyzing conversation history for branching topics...")
    try:
        with open(log_filepath, "r", encoding="utf-8") as f:
            history_content = f.read()
            
        if not history_content.strip():
            print("[SYSTEM] Log file is empty. Nothing to branch.")
            return []

        branch_prompt = f"""
        You are a backend data extraction routine. Analyze the following chat log and identify 2 to 3 distinct, high-level topics or paths that the user could branch out into for a separate, focused conversation.
        You MUST respond ONLY with a valid JSON array of strings representing these topics. Do not include markdown formatting, markdown code blocks (like ```json), or any introductory text.
        
        Example Output:
        ["C++ Memory Management", "Danish Interior Design", "DuckDB Green Logistics Pipeline"]

        Chat Log to Analyze:
        {history_content}
        """
        response = llm.invoke(branch_prompt)
        
        if isinstance(response.content, list):
            raw_json = response.content[0].get("text", "").strip()
        else:
            raw_json = response.content.strip()
            
        topics = json.loads(raw_json)
        return topics
    except Exception as e:
        print(f"[SYSTEM] Failed to parse branches: {e}")
        return []
    
while True:
    user_input = input("\nYou: ").strip()
    
    if user_input.lower() in ['quit', 'exit']:
        print("Shutting down... Goodbye!")
        break
        
    # ==========================================
    # THE BRANCH COMMAND INTERCEPTOR
    # ==========================================
    if user_input.lower() == 'branch':
        topics = extract_chat_branches(log_filename)
        
        if topics:
            print("\n" + "="*40)
            print(" 🧠 Suggested Chat Branches ")
            print("="*40)
            for i, topic in enumerate(topics, 1):
                print(f"[{i}] {topic}")
            print("[C] Cancel and continue current chat")
            print("="*40)
            
            # The Final Branch Execution Logic
            choice = input("\nSelect a branch number (or C to cancel): ").strip().upper()
            if choice != 'C' and choice.isdigit() and 1 <= int(choice) <= len(topics):
                selected_topic = topics[int(choice)-1]
                print(f"\n[SYSTEM] Executing Branch: {selected_topic}...")
                
                # 1. Format the topic into a clean filename
                clean_topic = selected_topic.replace(" ", "_").replace("/", "-")
                new_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                # 2. Update the log filename so all future messages go here!
                log_filename = f"chat_logs/branch_{clean_topic}_{new_timestamp}.txt"
                
                # 3. Create the new file and write a clean header
                with open(log_filename, "w", encoding="utf-8") as f:
                    f.write(f"--- BRANCHED CHAT: {selected_topic} ---\n")
                    f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("-" * 40 + "\n")
                    
                print(f"[SYSTEM] Branch successful. Now logging to: {log_filename}")
                
                # 4. Reset the drift counter for this new branch
                message_counter = 0
                
            else:
                print("\n[SYSTEM] Resuming current chat...")
        
        # This 'continue' is crucial! It forces the loop to restart 
        # so the word "branch" NEVER reaches the AI database.
        continue 
        
    if not user_input:
        continue

    # Search the database and generate the answer safely
    try:
        response = rag_chain.invoke({"input": user_input})
        
        print(f"\nAI: {response['answer']}\n")
        print("-" * 40)

        # Append the exchange directly to the log file
        with open(log_filename, "a", encoding="utf-8") as f:
            f.write(f"User: {user_input}\n")
            f.write(f"AI: {response['answer']}\n")
            f.write("-" * 40 + "\n")
            
    except Exception as e:
        error_msg = str(e)
        if "503" in error_msg or "UNAVAILABLE" in error_msg:
            print("\n[SYSTEM WARNING] The Gemini API is currently experiencing high traffic. Please wait a few seconds and try again.")
            continue
        elif "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            print("\n[SYSTEM WARNING] You hit the free-tier speed limit (20 requests/minute). Please wait about 45 seconds and try again.")
            continue
        else:
            print(f"\n[SYSTEM ERROR] An unexpected error occurred: {e}")
            continue
    
    # Increment the message counter
    message_counter += 1
    
    # ==========================================
    # SEMANTIC DRIFT CHECK (Every 10 messages)
    # ==========================================
    if message_counter % 10 == 0:
        print("\n[SYSTEM] Running background topic check...")
        
        # Read the last ~2000 characters of the log to save token costs
        with open(log_filename, "r", encoding="utf-8") as f:
            chat_history = f.read()[-2000:] 
            
        drift_prompt = f"""
        Analyze this recent chat history. The current file name is '{log_filename}'.
        Has the primary topic of the conversation significantly changed from the title? 
        If YES, reply ONLY with a new 3-to-4 word title (no spaces, use underscores). 
        If NO, reply ONLY with the word 'NO'.
        
        Recent History:
        {chat_history}
        """
        
        drift_response = llm.invoke(drift_prompt)
        
        # Safely extract text
        if isinstance(drift_response.content, list):
            drift_evaluation = drift_response.content[0].get("text", "").strip()
        else:
            drift_evaluation = drift_response.content.strip()
            
        # If the AI detects a change, it triggers the suggestion
        if drift_evaluation != "NO" and drift_evaluation != "":
            # TERMINAL VERSION 
            user_choice = input(f"[SYSTEM] Topic drift detected! Rename file to '{drift_evaluation}'? (Y/N): ").strip().upper()
            
            if user_choice == 'Y':
                new_filename = f"chat_logs/{drift_evaluation}_{timestamp}.txt"
                os.rename(log_filename, new_filename)
                log_filename = new_filename
                print(f"[SYSTEM] Chat successfully renamed to: {log_filename}")
                
    # ==========================================
    # THE CREATIVE SECRETARY (First Loop Only)
    # ==========================================
    if is_first_message:
        print("[SYSTEM] Generating chat title suggestion...")
        
        # Ask the LLM to summarize the first prompt into a title
        title_prompt = f"Write a clean, 3-to-4 word title for a chat that starts with this message: '{user_input}'. Respond ONLY with the title. No quotes, no punctuation."
        response_msg = llm.invoke(title_prompt)
        
        # Safely extract the text whether LangChain returns a string or a list
        if isinstance(response_msg.content, list):
            raw_title = response_msg.content[0].get("text", "")
        else:
            raw_title = response_msg.content
            
        suggested_title = raw_title.strip().replace(" ", "_")
        
        # Present the suggestion to the user
        user_choice = input(f"[SYSTEM] Save chat as '{suggested_title}'? (Press ENTER to accept, or type your own): ").strip()
        
        # Determine final name and rename the file
        final_title = user_choice.replace(" ", "_") if user_choice else suggested_title
        new_filename = f"chat_logs/{final_title}_{timestamp}.txt" 
        
        os.rename(log_filename, new_filename)
        log_filename = new_filename # Update the variable so future messages save to the new file
        
        print(f"[SYSTEM] Chat successfully renamed to: {log_filename}")
        
        # Flip the flag so this only happens once per session
        is_first_message = False