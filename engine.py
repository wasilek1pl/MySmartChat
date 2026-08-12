import re
import config
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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

    def extract_specific_topic(self, chat_text, target_topic):
        """Extracts and summarizes information strictly related to a user-defined target topic."""
        prompt = f"""Review the following conversation log. Extract and summarize all technical details, concepts, and decisions specifically related to the topic: "{target_topic}". 
        Strictly ignore all unrelated information. Provide a dense, comprehensive summary of the findings.
        
        Log:\n{chat_text}"""
        return self.llm.invoke(prompt).content.strip()

    def summarize_chat(self, chat_text):
        """Generates a dense, comprehensive summary of an entire conversation log for map-reduce processing."""
        prompt = f"""Review the following complete conversation history. Provide a dense, comprehensive technical summary that captures all key concepts, decisions, and data structures discussed throughout the entire log. Avoid generic filler or introductory text.
        
        Log:\n{chat_text}"""
        return self.llm.invoke(prompt).content.strip()

    def generate_short_preview(self, detailed_summary):
        """Compresses a detailed technical summary into a single plain-text sentence for the UI."""
        prompt = f"Summarize the following text into exactly one short, concise sentence. Do not include introductory filler. Output only the plain text sentence.\n\nText:\n{detailed_summary}"
        return self.llm.invoke(prompt).content.strip()