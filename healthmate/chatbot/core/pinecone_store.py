# chatbot/chatbot_logic/pinecone_store.py
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
import os

class PineconeStore:
    def __init__(self, api_key, index_name):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.index = self.pc.Index(index_name)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = PineconeVectorStore(index=self.index, embedding=self.embeddings)

    def search(self, query):
        return self.vector_store.similarity_search(query)
