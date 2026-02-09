import json
import os
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.graphs import NetworkxEntityGraph

class HelixAIEngine:
    def __init__(self, data_folder, process_json=None):
        self.llm = Ollama(model="llama3") # or your preferred model
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.process = self._load_process(process_json)
        
        # 1. Load Knowledge Graph (JSONs) and Docs (MD)
        self.kb_data = self._load_data(data_folder)
        
        # 2. Build RAG Index
        self.vector_db = FAISS.from_texts(self.kb_data['texts'], self.embeddings)

    def _load_data(self, folder):
        texts = []
        for file in os.listdir(folder):
            if file.endswith(".md"):
                with open(os.path.join(folder, file)) as f:
                    texts.append(f.read())
            elif file.endswith(".json"):
                # Handle Knowledge Graph JSON
                pass 
        return {"texts": texts}

    def query(self, user_input, context_code=""):
        # Retrieve relevant context
        docs = self.vector_db.similarity_search(user_input, k=3)
        context = "\n".join([d.page_content for d in docs])
        
        # Follow strict process if provided
        system_prompt = f"Follow this process: {self.process}\nContext: {context}"
        
        full_prompt = f"{system_prompt}\n\nCode context:\n{context_code}\n\nUser: {user_input}"
        return self.llm.invoke(full_prompt)
