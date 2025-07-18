from typing import List
from langchain_ollama import OllamaEmbeddings # Updated import
from langchain.schema import Document
from config import OLLAMA_EMBEDDING_MODEL

def get_ollama_embeddings_model(model_name: str = OLLAMA_EMBEDDING_MODEL) -> OllamaEmbeddings:
    """
    Initializes and returns an OllamaEmbeddings instance.
    """
    try:
        embeddings = OllamaEmbeddings(model=model_name)
        # Test if the model is accessible
        embeddings.embed_query("test query")
        print(f"Successfully initialized OllamaEmbeddings with model: {model_name}")
        return embeddings
    except Exception as e:
        print(f"Error initializing OllamaEmbeddings: {e}")
        print(f"Please ensure Ollama is running and the model '{model_name}' is pulled (`ollama pull {model_name}`).")
        return None

def generate_embeddings_for_chunks(chunks: List[Document], embeddings_model: OllamaEmbeddings) -> List[Document]:
    """
    Generates embeddings for a list of document chunks and attaches them to the chunk metadata.
    The embedding is stored under the 'vector' key in metadata, as expected by Weaviate.
    """
    if not embeddings_model:
        print("Embeddings model not available. Cannot generate embeddings.")
        return chunks # Return original chunks if model is not available

    print(f"Generating embeddings for {len(chunks)} chunks...")
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embeddings_model.embed_documents(texts)
    print(f"Generated {len(embeddings)} embeddings.")

    # Attach embeddings to chunks' metadata under the 'vector' key
    for i, chunk in enumerate(chunks):
        chunk.metadata["vector"] = embeddings[i]

    return chunks # Return chunks with embeddings in metadata

if __name__ == "__main__":
    # Example usage when run directly
    print("--- Testing embedding_manager.py ---")
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    # Create a dummy chunk for testing
    dummy_doc = Document(page_content="This is a test document for embedding generation.", metadata={"source": "test.txt"})
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    dummy_chunks = text_splitter.split_documents([dummy_doc])

    ollama_embeddings = get_ollama_embeddings_model()
    if ollama_embeddings:
        chunks_with_embeddings = generate_embeddings_for_chunks(dummy_chunks, ollama_embeddings)
        if chunks_with_embeddings and "vector" in chunks_with_embeddings[0].metadata:
            print(f"First chunk's embedding vector (first 5 dimensions): {chunks_with_embeddings[0].metadata['vector'][:5]}")
            print(f"Dimension of embeddings: {len(chunks_with_embeddings[0].metadata['vector'])}")
    print("--- embedding_manager.py Test Complete ---")
