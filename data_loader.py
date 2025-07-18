import os
import json
from langchain_community.document_loaders import TextLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from typing import List

def load_documents(data_dir: str = DATA_DIR) -> List[Document]:
    """
    Loads various document types from a specified directory.
    Supports .txt and .json files.
    """
    all_documents = []

    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        if filename.endswith(".txt"):
            print(f"Loading TXT file: {filepath}")
            loader = TextLoader(filepath)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = filename
            all_documents.extend(docs)
        elif filename.endswith(".json"):
            print(f"Loading JSON file: {filepath}")
            # For JSON, manually process to extract content and metadata
            with open(filepath, 'r') as f:
                data = json.load(f)
                for item in data:
                    doc = Document(
                        page_content=item.get("content", ""),
                        metadata={
                            "source": filename,
                            "id": item.get("id"),
                            "category": item.get("category")
                        }
                    )
                    all_documents.append(doc)
    return all_documents

def chunk_documents(documents: List[Document], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[Document]:
    """
    Splits documents into smaller chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} raw documents into {len(chunks)} chunks.")
    return chunks

if __name__ == "__main__":
    # Example usage when run directly
    print("--- Testing data_loader.py ---")
    raw_docs = load_documents()
    print(f"Loaded {len(raw_docs)} raw documents.")
    for doc in raw_docs:
        print(f"  - Content: '{doc.page_content[:50]}...' Metadata: {doc.metadata}")

    processed_chunks = chunk_documents(raw_docs)
    print(f"First chunk content: '{processed_chunks[0].page_content[:100]}...'")
    print(f"First chunk metadata: {processed_chunks[0].metadata}")
    print("--- data_loader.py Test Complete ---")
