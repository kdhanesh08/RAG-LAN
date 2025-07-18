import weaviate
import weaviate.classes as wvc
from typing import List
from langchain_community.vectorstores import Weaviate
from langchain_community.embeddings import OllamaEmbeddings
from langchain.schema import Document
from config import WEAVIATE_URL, WEAVIATE_CLASS_NAME, RETRIEVER_K

def connect_to_weaviate() -> weaviate.WeaviateClient:
    """
    Connects to Weaviate using the v4 client and returns the client.
    """
    try:
        # v4 client uses weaviate.connect_to_local() or connect_to_custom()
        client = weaviate.connect_to_local()
        client.is_ready() # Check if the connection is successful
        print("Successfully connected to Weaviate.")
        return client
    except Exception as e:
        print(f"Error connecting to Weaviate: {e}")
        print("Please ensure your Weaviate Docker container is running (`docker-compose up -d`).")
        return None

def setup_weaviate_schema(client: weaviate.WeaviateClient, class_name: str):
    """
    Sets up the Weaviate schema using the v4 client API.
    """
    if client.collections.exists(class_name):
        print(f"Class '{class_name}' already exists in Weaviate schema. Skipping setup.")
        return

    print(f"Creating class '{class_name}' in Weaviate schema...")
    try:
        client.collections.create(
            name=class_name,
            properties=[
                wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="source", data_type=wvc.config.DataType.TEXT),
            ],
            # We provide vectors manually, so no vectorizer is needed.
            vectorizer_config=wvc.config.Configure.Vectorizer.none()
        )
        print(f"Class '{class_name}' created successfully.")
    except Exception as e:
        print(f"Error creating Weaviate schema: {e}")

def ingest_data_to_weaviate(client: weaviate.WeaviateClient, chunks_with_embeddings: List[Document], class_name: str):
    """
    Ingests data into Weaviate using the v4 client batching.
    """
    print(f"Ingesting {len(chunks_with_embeddings)} chunks into Weaviate class '{class_name}'...")
    try:
        collection = client.collections.get(class_name)
        
        # Use the context manager for automatic batching
        with collection.batch.dynamic() as batch:
            for chunk in chunks_with_embeddings:
                properties = {
                    "text": chunk.page_content,
                    "source": chunk.metadata.get("source", "Unknown")
                }
                batch.add_object(
                    properties=properties,
                    vector=chunk.metadata["vector"] # Assumes vector is in metadata
                )
        
        print(f"Successfully ingested {len(chunks_with_embeddings)} chunks.")
    except Exception as e:
        print(f"Error ingesting data to Weaviate: {e}")

def get_weaviate_retriever(client: weaviate.WeaviateClient, embeddings_model: OllamaEmbeddings, class_name: str) -> Weaviate:
    """
    Creates and returns a Weaviate retriever compatible with LangChain and v4 client.
    """
    print("Setting up Weaviate retriever...")
    try:
        # The LangChain Weaviate vector store directly accepts the v4 client
        vector_store = Weaviate(
            client=client,
            index_name=class_name,
            text_key="text",
            embedding=embeddings_model, # Used for querying
            by_text=False # We search by vector, not text
        )
        print("Weaviate retriever setup successful.")
        return vector_store.as_retriever(search_kwargs={'k': RETRIEVER_K})
    except Exception as e:
        print(f"Error setting up Weaviate retriever: {e}")
        return None

if __name__ == "__main__":
    # Example usage when run directly
    print("--- Testing weaviate_manager.py ---")
    from data_loader import load_documents, chunk_documents
    from embedding_manager import get_ollama_embeddings_model, generate_embeddings_for_chunks

    weaviate_client = connect_to_weaviate()
    ollama_embeddings = get_ollama_embeddings_model()

    if weaviate_client and ollama_embeddings:
        # Load, chunk, and embed dummy data
        raw_docs = load_documents()
        processed_chunks = chunk_documents(raw_docs)
        chunks_with_embeddings = generate_embeddings_for_chunks(processed_chunks, ollama_embeddings)

        setup_weaviate_schema(weaviate_client)
        ingest_data_to_weaviate(weaviate_client, chunks_with_embeddings, ollama_embeddings)

        # Verify data count
        try:
            count_result = weaviate_client.query.aggregate(WEAVIATE_CLASS_NAME).with_meta_count().do()
            print(f"Number of objects in Weaviate: {count_result['data']['Aggregate'][WEAVIATE_CLASS_NAME][0]['meta']['count']}")
        except Exception as e:
            print(f"Error verifying data count: {e}")

        # Test retrieval
        weaviate_retriever = get_weaviate_retriever(weaviate_client, ollama_embeddings)
        if weaviate_retriever:
            print("\n--- Test Retrieval ---")
            query = "What did Elara find?"
            relevant_docs = weaviate_retriever.invoke(query)
            print(f"Found {len(relevant_docs)} relevant documents for query: '{query}'")
            for i, doc in enumerate(relevant_docs):
                print(f"  Doc {i+1} (Source: {doc.metadata.get('source', 'N/A')}): '{doc.page_content[:50]}...'")
    print("--- weaviate_manager.py Test Complete ---")
