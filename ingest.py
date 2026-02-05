import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

data_folder = "my_knowledge" 

files = os.listdir(data_folder)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
for i in files:
    if i.endswith(".txt") or i.endswith(".md"):
        full_path = os.path.join(data_folder, i)
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
            chunks = text_splitter.split_text(content)
            print(f"Processing {i}: split into {len(chunks)} chunks.")