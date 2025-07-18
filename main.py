import os
from config import DATA_DIR, WEAVIATE_CLASS_NAME, OLLAMA_EMBEDDING_MODEL, OLLAMA_GENERATION_MODEL
from data_loader import load_documents, chunk_documents
from embedding_manager import get_ollama_embeddings_model, generate_embeddings_for_chunks
from weaviate_manager import connect_to_weaviate, setup_weaviate_schema, ingest_data_to_weaviate, get_weaviate_retriever
from agent_builder import build_langgraph_agent, AgentState,get_ollama_llm

def main():
    print("--- Starting RAG Agent Application ---")

    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' is empty or does not exist. Please create it and add data files.")
        return

    weaviate_client = None # Initialize to None
    try:
        # 2. Connect to Weaviate and Ollama models
        print("\n--- Initializing Connections ---")
        weaviate_client = connect_to_weaviate()
        ollama_embeddings = get_ollama_embeddings_model(OLLAMA_EMBEDDING_MODEL)
        ollama_llm = get_ollama_llm(OLLAMA_GENERATION_MODEL) # Get LLM for agent_builder

        if not all([weaviate_client, ollama_embeddings, ollama_llm]):
            print("One or more essential connections/models failed to initialize. Exiting.")
            return

        # 3. Load, Chunk, and Embed Data
        print("\n--- Data Processing ---")
        raw_docs = load_documents(DATA_DIR)
        if not raw_docs:
            print("No documents loaded. Please check data directory and files. Exiting.")
            return
        processed_chunks = chunk_documents(raw_docs)
        chunks_with_embeddings = generate_embeddings_for_chunks(processed_chunks, ollama_embeddings)
        if not chunks_with_embeddings:
            print("No embeddings generated. Exiting.")
            return

        # 4. Setup Weaviate Schema and Ingest Data
        print("\n--- Weaviate Data Management ---")
        setup_weaviate_schema(weaviate_client, WEAVIATE_CLASS_NAME)
        ingest_data_to_weaviate(weaviate_client, chunks_with_embeddings, WEAVIATE_CLASS_NAME)

        # 5. Get Retriever
        print("\n--- Retriever Setup ---")
        retriever_for_agent = get_weaviate_retriever(weaviate_client, ollama_embeddings, WEAVIATE_CLASS_NAME)
        if not retriever_for_agent:
            print("Retriever setup failed. Exiting.")
            return

        # 6. Build LangGraph Agent
        print("\n--- Building LangGraph Agent ---")
        agent_app = build_langgraph_agent(retriever_for_agent, ollama_llm) # Pass both retriever and LLM

        # --- Run Agent Tests ---
        print("\n--- Running Agent Tests ---")

        test_queries = [
            "What did Elara find in the wizard's tower?",
            "Tell me about the Roman Empire.",
            "Who is William Shakespeare?",
            "What is the capital of France?" # This should result in "don't know"
        ]

        for i, query in enumerate(test_queries):
            print(f"\n--- Agent Run {i+1} (Query: '{query}') ---")
            inputs = {"question": query, "chat_history": []}
            try:
                for s in agent_app.stream(inputs):
                    if "__end__" not in s: # LangGraph streams intermediate states and then the final state
                        print(s)
                    else:
                        print(f"Final Generation: {s['__end__']['generation']}")
                    print("---")
            except Exception as e:
                print(f"Error during agent run for query '{query}': {e}")

    finally:
        # This block will run whether the try block succeeds or fails
        if weaviate_client:
            weaviate_client.close()
            print("\nWeaviate client connection closed.")

    print("\n--- RAG Agent Application Complete ---")

if __name__ == "__main__":
    main()
