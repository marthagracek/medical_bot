# from flask import Flask, render_template, request, jsonify
# from langchain_community.vectorstores import Pinecone
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_ollama import OllamaLLM
# from src.helper import load_documents
# from src.store_index import load_vectorstore
# import traceback

# app = Flask(__name__)

# # Initialize embeddings
# print("Initializing embeddings...")
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# # Load Pinecone vectorstore
# print("Loading vectorstore...")
# index_name = "medic-bot"
# try:
#     vectorstore = load_vectorstore(index_name, embeddings)
#     print("Vectorstore loaded successfully!")
# except Exception as e:
#     print(f"Error loading vectorstore: {e}")
#     vectorstore = None

# # Initialize Ollama LLM
# print("Initializing Ollama LLM...")
# try:
#     llm = OllamaLLM(model="gemma:4b", base_url="http://localhost:11434")
#     print("Ollama LLM initialized successfully!")
# except Exception as e:
#     print(f"Error initializing Ollama: {e}")
#     llm = None

# @app.route("/")
# def home():
#     return render_template("chat.html")

# @app.route("/get", methods=["POST"])
# def chatbot_response():
#     try:
#         user_input = request.form["msg"]
        
#         if not user_input.strip():
#             return "Please enter a question."
        
#         if not vectorstore:
#             return "Sorry, the vectorstore is not available. Please check your setup."
        
#         if not llm:
#             return "Sorry, the language model is not available. Please ensure Ollama is running."
        
#         print(f"User question: {user_input}")
        
#         # Search for relevant documents
#         docs = vectorstore.similarity_search(user_input, k=3)
        
#         if not docs:
#             return "Sorry, I couldn't find any relevant information to answer your question."
        
#         # Create context from retrieved documents
#         context = "\n\n".join([doc.page_content for doc in docs])
        
#         # Create prompt for the LLM
#         prompt = f"""Based on the following medical context, please provide a helpful and accurate answer to the question. If the context doesn't contain enough information to answer the question, please say so.

# Context: {context}

# Question: {user_input}

# Answer:"""
        
#         print("Generating response...")
        
#         # Get response from LLM
#         response = llm.invoke(prompt)
        
#         print(f"Response generated: {response[:100]}...")
        
#         return str(response)
        
#     except Exception as e:
#         print(f"Error in chatbot_response: {e}")
#         print(traceback.format_exc())
#         return f"Sorry, an error occurred: {str(e)}"

# @app.route("/health")
# def health_check():
#     """Health check endpoint"""
#     status = {
#         "vectorstore": "OK" if vectorstore else "ERROR",
#         "llm": "OK" if llm else "ERROR",
#         "embeddings": "OK" if embeddings else "ERROR"
#     }
#     return jsonify(status)

# if __name__ == "__main__":
#     print("Starting Flask application...")
#     print("Make sure you have:")
#     print("1. Documents in the 'data' folder")
#     print("2. Ollama running with gemma:4b model")
#     print("3. Valid Pinecone API key")
#     print("-" * 50)
    
#     app.run(debug=True, host="0.0.0.0", port=5000)


from flask import Flask, render_template, request, jsonify
from langchain_community.vectorstores import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from src.helper import load_documents
from src.store_index import load_vectorstore
import traceback
import subprocess
import sys

app = Flask(__name__)

# Initialize embeddings
print("Initializing embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load Pinecone vectorstore
print("Loading vectorstore...")
index_name = "medic-bot"
try:
    vectorstore = load_vectorstore(index_name, embeddings)
    print("Vectorstore loaded successfully!")
except Exception as e:
    print(f"Error loading vectorstore: {e}")
    print("Running without vectorstore - will use basic LLM responses")
    vectorstore = None

# Initialize Ollama LLM
print("Initializing Ollama LLM...")
try:
    # Try different models in order of preference
    models_to_try = ["llama3.2:3b", "llama3.1:8b", "llama2:7b", "mistral:7b", "gemma:2b", "phi3:mini"]
    llm = None
    
    for model in models_to_try:
        try:
            print(f"Trying model: {model}")
            llm = OllamaLLM(model=model, base_url="http://localhost:11434")
            # Test the model with a simple query
            test_response = llm.invoke("Hello")
            print(f"Successfully initialized Ollama with model: {model}")
            break
        except Exception as e:
            print(f"Model {model} not available: {str(e)[:100]}")
            continue
    
    if not llm:
        print("No Ollama models available. Attempting to install llama3.2:3b...")
        try:
            result = subprocess.run(["ollama", "pull", "llama3.2:3b"], 
                                  capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print("Successfully installed llama3.2:3b")
                llm = OllamaLLM(model="llama3.2:3b", base_url="http://localhost:11434")
                print("Ollama LLM initialized with llama3.2:3b!")
            else:
                print(f"Failed to install model: {result.stderr}")
                llm = None
        except subprocess.TimeoutExpired:
            print("Model installation timed out")
            llm = None
        except FileNotFoundError:
            print("Ollama command not found. Please install Ollama first.")
            llm = None
        except Exception as e:
            print(f"Error installing model: {e}")
            llm = None
        
except Exception as e:
    print(f"Error connecting to Ollama: {e}")
    llm = None

@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    try:
        user_input = request.form["msg"]
        
        if not user_input.strip():
            return "Please enter a question."
        
        if not llm:
            return """I'm a medical assistant chatbot, but I'm currently not connected to the language model. 
            
Please install Ollama to enable AI responses:
1. Visit https://ollama.com/download
2. Install Ollama
3. Run: ollama pull llama3.2:3b
4. Restart this application

For now, I recommend consulting with healthcare professionals for medical questions."""
        
        print(f"User question: {user_input}")
        
        if vectorstore:
            # Use vectorstore if available
            docs = vectorstore.similarity_search(user_input, k=3)
            
            if not docs:
                context = "No specific medical documents found."
            else:
                context = "\n\n".join([doc.page_content for doc in docs])
            
            prompt = f"""Based on the following medical context, please provide a helpful and accurate answer to the question. If the context doesn't contain enough information to answer the question, please say so.

Context: {context}

Question: {user_input}

Answer:"""
        else:
            # Fallback: use LLM without vectorstore
            prompt = f"""You are a medical assistant. Please provide a helpful and informative answer to the following medical question. Always remind users to consult with healthcare professionals for serious medical concerns.

Question: {user_input}

Answer:"""
        
        print("Generating response...")
        
        # Get response from LLM
        response = llm.invoke(prompt)
        
        print(f"Response generated: {response[:100]}...")
        
        return str(response)
        
    except Exception as e:
        print(f"Error in chatbot_response: {e}")
        print(traceback.format_exc())
        return f"Sorry, an error occurred: {str(e)}"

@app.route("/health")
def health_check():
    """Health check endpoint"""
    status = {
        "vectorstore": "OK" if vectorstore else "ERROR",
        "llm": "OK" if llm else "ERROR",
        "embeddings": "OK" if embeddings else "ERROR"
    }
    return jsonify(status)

@app.route("/models")
def list_models():
    """List available Ollama models"""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode == 0:
            return f"<pre>{result.stdout}</pre>"
        else:
            return f"Error: {result.stderr}"
    except Exception as e:
        return f"Error checking models: {e}"

if __name__ == "__main__":
    print("Starting Flask application...")
    print("Status:")
    print(f"- Vectorstore: {'✅ OK' if vectorstore else '❌ ERROR (Pinecone limit reached)'}")
    print(f"- LLM: {'✅ OK' if llm else '❌ ERROR (No Ollama model)'}")
    print(f"- Embeddings: {'✅ OK' if embeddings else '❌ ERROR'}")
    print()
    print("Make sure you have:")
    print("1. Documents in the 'data' folder (optional)")
    print("2. Ollama running with any model")
    print("3. Valid Pinecone API key (optional for basic functionality)")
    print("-" * 50)
    print("Access your chatbot at: http://127.0.0.1:5000")
    print("Check available models at: http://127.0.0.1:5000/models")
    print("Check system health at: http://127.0.0.1:5000/health")
    print("-" * 50)
    
    app.run(debug=True, host="0.0.0.0", port=5000)