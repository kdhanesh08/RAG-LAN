from typing import TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama # Updated import
from langchain.schema import Document
from langgraph.graph import StateGraph, END
from langchain_core.retrievers import BaseRetriever # Import BaseRetriever for type hinting
from config import OLLAMA_GENERATION_MODEL

# Define the AgentState
class AgentState(TypedDict):
    """
    Represents the state of our agent.
    - `question`: The user's initial question.
    - `documents`: List of retrieved documents.
    - `chat_history`: List of chat messages (for conversational memory).
    - `generation`: The LLM's generated response.
    """
    question: str
    documents: List[Document]
    chat_history: List[BaseMessage]
    generation: str

# Define LLM for generation (using Ollama)
def get_ollama_llm(model_name: str = OLLAMA_GENERATION_MODEL) -> ChatOllama:
    """
    Initializes and returns a ChatOllama instance for generation.
    """
    try:
        llm = ChatOllama(model=model_name)
        llm.invoke("hello") # Test if the model is accessible
        print(f"Successfully initialized ChatOllama with model: {model_name}")
        return llm
    except Exception as e:
        print(f"Error initializing ChatOllama: {e}")
        print(f"Please ensure Ollama is running and the model '{model_name}' is pulled (`ollama pull {model_name}`).")
        return None

# Global placeholder for retriever, will be set during agent build
# This is a common pattern when passing dependencies into LangGraph nodes
_weaviate_retriever: BaseRetriever = None
_ollama_llm: ChatOllama = None

# --- Graph Nodes ---

def retrieve(state: AgentState) -> dict:
    """
    Retrieves documents from Weaviate based on the user's question.
    """
    print("---NODE: RETRIEVE---")
    question = state["question"]
    if _weaviate_retriever is None:
        print("Error: Retriever not initialized in agent_builder.")
        return {"documents": []} # Return empty documents if retriever is not set

    documents = _weaviate_retriever.invoke(question)
    print(f"Retrieved {len(documents)} documents.")
    return {"documents": documents, "question": question}

def generate(state: AgentState) -> dict:
    """
    Generates a response using the LLM based on the retrieved documents and question.
    """
    print("---NODE: GENERATE---")
    question = state["question"]
    documents = state["documents"]
    chat_history = state["chat_history"]

    if _ollama_llm is None:
        print("Error: LLM not initialized in agent_builder.")
        return {"generation": "Error: LLM not available."}

    # Construct the prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the user's question based *only* on the provided context. If you cannot find the answer in the context, state that you don't know."),
        ("user", "Context: {context}\nQuestion: {question}")
    ])

    # Format the context from retrieved documents
    context = "\n\n".join([doc.page_content for doc in documents])

    # Create the chain for generation
    rag_chain = prompt | _ollama_llm

    # Invoke the chain with the current state
    response = rag_chain.invoke({"context": context, "question": question})

    return {"generation": response.content, "question": question, "documents": documents, "chat_history": chat_history}

# --- Build the LangGraph ---

def build_langgraph_agent(retriever_instance: BaseRetriever, llm_instance: ChatOllama):
    """
    Builds and compiles the LangGraph agent.
    Initializes global retriever and LLM instances for use in nodes.
    """
    global _weaviate_retriever, _ollama_llm
    _weaviate_retriever = retriever_instance
    _ollama_llm = llm_instance

    workflow = StateGraph(AgentState)

    # Define the nodes
    workflow.add_node("retrieve", retrieve) # Node to retrieve documents
    workflow.add_node("generate", generate) # Node to generate response

    # Define the entry point and edges
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate") # After retrieval, always generate
    workflow.add_edge("generate", END)       # After generation, end the process

    # Compile the graph
    app = workflow.compile()
    print("LangGraph agent compiled successfully.")
    return app

if __name__ == "__main__":
    # Example usage when run directly (requires Weaviate and Ollama running)
    print("--- Testing agent_builder.py ---")
    from weaviate_manager import connect_to_weaviate, get_weaviate_retriever
    from embedding_manager import get_ollama_embeddings_model
    from data_loader import load_documents, chunk_documents
    from weaviate_manager import setup_weaviate_schema, ingest_data_to_weaviate

    # Simulate full setup for testing
    weaviate_client = connect_to_weaviate()
    ollama_embeddings = get_ollama_embeddings_model()
    ollama_llm = get_ollama_llm()

    if weaviate_client and ollama_embeddings and ollama_llm:
        print("Performing data ingestion for testing agent_builder...")
        raw_docs = load_documents()
        processed_chunks = chunk_documents(raw_docs)
        chunks_with_embeddings = generate_embeddings_for_chunks(processed_chunks, ollama_embeddings)
        setup_weaviate_schema(weaviate_client)
        ingest_data_to_weaviate(weaviate_client, chunks_with_embeddings, ollama_embeddings)
        print("Data ingestion complete for agent_builder test.")

        retriever_for_agent = get_weaviate_retriever(weaviate_client, ollama_embeddings)

        if retriever_for_agent:
            agent_app = build_langgraph_agent(retriever_for_agent, ollama_llm)

            print("\n--- Agent Run Test 1 ---")
            inputs = {"question": "What did Elara find in the wizard's tower?", "chat_history": []}
            for s in agent_app.stream(inputs):
                print(s)
                print("---")
    print("--- agent_builder.py Test Complete ---")
