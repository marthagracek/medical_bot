import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_documents(folder_path="data"):
    """Load documents from the specified folder"""
    documents = []
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Warning: Folder '{folder_path}' does not exist")
        return documents
    
    # Check if folder is empty
    files = os.listdir(folder_path)
    if not files:
        print(f"Warning: Folder '{folder_path}' is empty")
        return documents
    
    print(f"Loading documents from '{folder_path}'...")
    
    for file_name in files:
        full_path = os.path.join(folder_path, file_name)
        
        # Skip directories
        if os.path.isdir(full_path):
            continue
            
        try:
            print(f"Loading: {file_name}")
            
            if file_name.endswith(".txt"):
                loader = TextLoader(full_path, encoding='utf-8')
            elif file_name.endswith(".pdf"):
                loader = PyPDFLoader(full_path)
            else:
                # Try with UnstructuredFileLoader for other file types
                loader = UnstructuredFileLoader(full_path)
            
            docs = loader.load()
            documents.extend(docs)
            print(f"Loaded {len(docs)} documents from {file_name}")
            
        except Exception as e:
            print(f"Error loading {file_name}: {str(e)}")
            continue
    
    print(f"Total documents loaded: {len(documents)}")
    return documents

def split_documents(documents, chunk_size=500, chunk_overlap=100):
    """Split documents into smaller chunks"""
    if not documents:
        print("No documents to split")
        return []
    
    print(f"Splitting {len(documents)} documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    split_docs = splitter.split_documents(documents)
    print(f"Created {len(split_docs)} document chunks")
    return split_docs