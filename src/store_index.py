from langchain_community.vectorstores import Pinecone
import pinecone
from src.helper import load_documents, split_documents

PINECONE_API_KEY = "pcsk_5JX5ns_PnWDiUXNMjdi3c3uVKyPwpKAUyARwk5orKdm9h8zGawwxvZiQqaxkcxydJFEpXw"
INDEX_NAME = "medic-bot"

def create_pinecone_index(index_name=INDEX_NAME):
    """Create Pinecone index if it doesn't exist"""
    pinecone.init(api_key=PINECONE_API_KEY, environment='us-east-1-aws')
    
    # Get list of existing index names
    existing_indexes = pinecone.list_indexes()
    
    if index_name not in existing_indexes:
        print(f"Creating new index: {index_name}")
        pinecone.create_index(
            name=index_name,
            dimension=384,
            metric='cosine'
        )
        print(f"Index '{index_name}' created successfully")
    else:
        print(f"Index '{index_name}' already exists")

def load_vectorstore(index_name, embeddings):
    """Load or create vectorstore with documents"""
    # Ensure index exists
    create_pinecone_index(index_name)
    
    # Connect to existing index
    vectorstore = Pinecone.from_existing_index(index_name, embeddings)
    
    # Check if we need to load documents
    try:
        # Try a simple search to see if there are any vectors
        test_results = vectorstore.similarity_search("test", k=1)
        if not test_results:
            raise Exception("No vectors found")
        else:
            print(f"Index contains existing vectors")
    except:
        print("Index appears to be empty, loading documents...")
        documents = load_documents()
        if documents:
            split_docs = split_documents(documents)
            print(f"Adding {len(split_docs)} document chunks to vectorstore...")
            vectorstore.add_documents(split_docs)
            print("Documents added successfully!")
        else:
            print("No documents found in data folder")
    
    return vectorstore

def create_and_populate_vectorstore(index_name, embeddings, documents):
    """Create vectorstore and populate with documents"""
    create_pinecone_index(index_name)
    split_docs = split_documents(documents)
    vectorstore = Pinecone.from_documents(split_docs, embeddings, index_name=index_name)
    return vectorstore