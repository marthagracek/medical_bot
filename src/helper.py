import os
import pickle
import hashlib
import time
from typing import List, Optional
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Cache directory for processed documents
CACHE_DIR = "cache"
CACHE_FILE = os.path.join(CACHE_DIR, "processed_documents.pkl")
HASH_FILE = os.path.join(CACHE_DIR, "documents_hash.txt")

def get_folder_hash(folder_path: str) -> str:
    """Generate hash of all files in folder for cache invalidation"""
    hash_md5 = hashlib.md5()
    
    if not os.path.exists(folder_path):
        return ""
    
    for root, dirs, files in os.walk(folder_path):
        # Sort for consistent hashing
        for file in sorted(files):
            file_path = os.path.join(root, file)
            try:
                # Hash file name and modification time
                hash_md5.update(file.encode())
                hash_md5.update(str(os.path.getmtime(file_path)).encode())
            except (OSError, IOError):
                continue
                
    return hash_md5.hexdigest()

def load_from_cache() -> Optional[List[Document]]:
    """Load processed documents from cache if valid"""
    try:
        if not os.path.exists(CACHE_FILE) or not os.path.exists(HASH_FILE):
            return None
            
        # Check if cache is still valid
        with open(HASH_FILE, 'r') as f:
            cached_hash = f.read().strip()
            
        current_hash = get_folder_hash("data")
        if cached_hash != current_hash:
            print("📂 Documents changed, cache invalid")
            return None
            
        # Load cached documents
        with open(CACHE_FILE, 'rb') as f:
            cached_docs = pickle.load(f)
            
        print(f"⚡ Loaded {len(cached_docs)} documents from cache")
        return cached_docs
        
    except Exception as e:
        print(f"⚠️ Cache load failed: {e}")
        return None

def save_to_cache(documents: List[Document]) -> None:
    """Save processed documents to cache"""
    try:
        # Create cache directory
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        # Save documents
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(documents, f)
            
        # Save hash
        with open(HASH_FILE, 'w') as f:
            f.write(get_folder_hash("data"))
            
        print(f"💾 Cached {len(documents)} processed documents")
        
    except Exception as e:
        print(f"⚠️ Cache save failed: {e}")

def load_single_document(file_path: str, file_name: str) -> List[Document]:
    """Load a single document with optimized settings"""
    try:
        print(f"📄 Loading: {file_name}")
        
        if file_name.endswith(".txt"):
            loader = TextLoader(file_path, encoding='utf-8')
        elif file_name.endswith(".pdf"):
            # Optimized PDF loading
            loader = PyPDFLoader(file_path)
        elif file_name.endswith((".docx", ".doc")):
            # Handle Word documents
            loader = UnstructuredFileLoader(file_path)
        elif file_name.endswith((".md", ".markdown")):
            # Handle Markdown files
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            # Try with UnstructuredFileLoader for other types
            loader = UnstructuredFileLoader(file_path)
        
        docs = loader.load()
        
        # Add metadata for better tracking
        for doc in docs:
            doc.metadata.update({
                'source_file': file_name,
                'file_path': file_path,
                'load_time': time.time()
            })
            
        print(f"✅ Loaded {len(docs)} documents from {file_name}")
        return docs
        
    except Exception as e:
        print(f"❌ Error loading {file_name}: {str(e)}")
        return []

def load_documents(folder_path: str = "data", use_cache: bool = True) -> List[Document]:
    """
    Load documents from folder with caching for better performance
    
    Args:
        folder_path: Path to documents folder
        use_cache: Whether to use caching (recommended for production)
    """
    start_time = time.time()
    
    # Try cache first if enabled
    if use_cache:
        cached_docs = load_from_cache()
        if cached_docs is not None:
            load_time = time.time() - start_time
            print(f"⚡ Cache load completed in {load_time:.2f}s")
            return cached_docs
    
    print(f"📂 Loading documents from '{folder_path}'...")
    documents = []
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"⚠️ Warning: Folder '{folder_path}' does not exist")
        print(f"💡 Create the folder and add your medical documents (PDF, TXT, DOCX)")
        return documents
    
    # Get all files
    try:
        files = [f for f in os.listdir(folder_path) 
                if os.path.isfile(os.path.join(folder_path, f))]
    except PermissionError:
        print(f"❌ Permission denied accessing '{folder_path}'")
        return documents
    
    if not files:
        print(f"⚠️ Warning: Folder '{folder_path}' is empty")
        print(f"💡 Add medical documents to enable document search")
        return documents
    
    # Filter supported file types for better performance
    supported_extensions = {'.txt', '.pdf', '.docx', '.doc', '.md', '.markdown'}
    valid_files = [f for f in files 
                   if any(f.lower().endswith(ext) for ext in supported_extensions)]
    
    if not valid_files:
        print(f"⚠️ No supported documents found in '{folder_path}'")
        print(f"💡 Supported: {', '.join(supported_extensions)}")
        return documents
        
    print(f"📄 Found {len(valid_files)} supported documents")
    
    # Load documents with progress tracking
    successful_loads = 0
    for i, file_name in enumerate(valid_files, 1):
        full_path = os.path.join(folder_path, file_name)
        
        print(f"[{i}/{len(valid_files)}] Processing {file_name}...")
        
        docs = load_single_document(full_path, file_name)
        if docs:
            documents.extend(docs)
            successful_loads += 1
    
    load_time = time.time() - start_time
    print(f"📊 Summary:")
    print(f"   ✅ Successfully loaded: {successful_loads}/{len(valid_files)} files")
    print(f"   📄 Total documents: {len(documents)}")
    print(f"   ⏱️  Load time: {load_time:.2f}s")
    
    # Save to cache for next time
    if use_cache and documents:
        save_to_cache(documents)
    
    return documents

def split_documents(documents: List[Document], 
                   chunk_size: int = 400,  # Reduced from 500 for faster processing
                   chunk_overlap: int = 50) -> List[Document]:  # Reduced from 100
    """
    Split documents into optimized chunks for faster retrieval
    
    Args:
        documents: List of documents to split
        chunk_size: Maximum characters per chunk (smaller = faster search)
        chunk_overlap: Character overlap between chunks
    """
    if not documents:
        print("⚠️ No documents to split")
        return []
    
    start_time = time.time()
    print(f"✂️  Splitting {len(documents)} documents into chunks...")
    print(f"📏 Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
    
    # Optimized splitter settings for medical content
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        # Separators optimized for medical text
        separators=[
            "\n\n",  # Paragraph breaks
            "\n",    # Line breaks  
            ". ",    # Sentence endings
            "? ",    # Question endings
            "! ",    # Exclamation endings
            "; ",    # Semicolons
            ", ",    # Commas
            " ",     # Spaces
            ""       # Character level (fallback)
        ],
        keep_separator=True
    )
    
    try:
        split_docs = splitter.split_documents(documents)
        
        # Add chunk metadata for better tracking
        for i, doc in enumerate(split_docs):
            doc.metadata.update({
                'chunk_id': i,
                'chunk_size': len(doc.page_content),
                'split_time': time.time()
            })
        
        split_time = time.time() - start_time
        avg_chunk_size = sum(len(doc.page_content) for doc in split_docs) / len(split_docs)
        
        print(f"✅ Split completed:")
        print(f"   📄 Created: {len(split_docs)} chunks")
        print(f"   📏 Average chunk size: {avg_chunk_size:.0f} characters")
        print(f"   ⏱️  Split time: {split_time:.2f}s")
        
        return split_docs
        
    except Exception as e:
        print(f"❌ Error splitting documents: {e}")
        return documents  # Return original documents as fallback

def get_document_stats(documents: List[Document]) -> dict:
    """Get statistics about loaded documents"""
    if not documents:
        return {"total_docs": 0, "total_chunks": 0, "avg_size": 0}
    
    total_chars = sum(len(doc.page_content) for doc in documents)
    avg_size = total_chars / len(documents)
    
    # Count unique source files
    source_files = set()
    for doc in documents:
        if 'source_file' in doc.metadata:
            source_files.add(doc.metadata['source_file'])
    
    return {
        "total_docs": len(documents),
        "total_chunks": len(documents),
        "source_files": len(source_files),
        "total_characters": total_chars,
        "avg_chunk_size": avg_size,
        "files": list(source_files)
    }

def clear_cache():
    """Clear the document cache"""
    try:
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
        if os.path.exists(HASH_FILE):
            os.remove(HASH_FILE)
        if os.path.exists(CACHE_DIR) and not os.listdir(CACHE_DIR):
            os.rmdir(CACHE_DIR)
        print("🗑️ Cache cleared successfully")
    except Exception as e:
        print(f"❌ Error clearing cache: {e}")

# Convenience function for quick testing
def quick_load_test(folder_path: str = "data"):
    """Quick test of document loading performance"""
    print("🚀 Quick Load Test")
    print("-" * 30)
    
    start_time = time.time()
    docs = load_documents(folder_path, use_cache=True)
    load_time = time.time() - start_time
    
    if docs:
        split_start = time.time()
        split_docs = split_documents(docs)
        split_time = time.time() - split_start
        
        stats = get_document_stats(split_docs)
        
        print(f"\n📊 Performance Results:")
        print(f"   📂 Load time: {load_time:.2f}s")
        print(f"   ✂️  Split time: {split_time:.2f}s")
        print(f"   📄 Total chunks: {stats['total_chunks']}")
        print(f"   📏 Avg chunk size: {stats['avg_chunk_size']:.0f} chars")
        
        # Performance rating
        total_time = load_time + split_time
        if total_time < 2:
            print("🟢 EXCELLENT performance")
        elif total_time < 5:
            print("🟡 GOOD performance")
        elif total_time < 10:
            print("🟠 ACCEPTABLE performance")
        else:
            print("🔴 SLOW performance - consider fewer/smaller documents")
            
    else:
        print("❌ No documents loaded")
    
    return docs

if __name__ == "__main__":
    # Quick test when run directly
    quick_load_test()