
import os

# Base directory for data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = "data"

# Weaviate Configuration
WEAVIATE_URL = "http://localhost:8080"
WEAVIATE_CLASS_NAME = "DocumentChunk"

# Ollama Configuration
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_GENERATION_MODEL = "tinyllama" # Or "mistral"

# Document Chunking Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retriever Configuration
RETRIEVER_K = 3 # Number of top documents to retrieve
