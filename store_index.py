from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.helper import load_documents, split_documents
import os

INDEX_FILE = "faiss_index"

def create_faiss_index(embeddings):
    """Create FAISS vectorstore index from documents."""
    documents = load_documents()
    if not documents:
        print("No documents found.")
        return None
    
    splits = split_documents(documents)
    print(f"Adding {len(splits)} document chunks to FAISS index...")
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local(INDEX_FILE)
    print("FAISS index saved locally.")
    return vectorstore

def load_vectorstore(embeddings):
    """Load existing FAISS index or create a new one if not found."""
    if os.path.exists(INDEX_FILE):
        print("Loading existing FAISS index...")
        return FAISS.load_local(INDEX_FILE, embeddings)
    else:
        print("No FAISS index found. Creating new one...")
        return create_faiss_index(embeddings)
