import os
import shutil
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def save_documents_to_chroma(documents: list[Document], cfg: dict, force_recreate: bool = True):
    """
    Handles memory-safe batch ingestion into Chroma DB.
    Can be reused by any chunking strategy that outputs a list of LangChain Documents.
    """
    global batch_buffer
    if not documents:
        if not documents:
            print("⚠️ No documents provided for storage.")
            return

    db_path = cfg["persist_directory_path"]
    collection = cfg["collection_name"]
    batch_size = cfg["batch_size"]
    embeddings = HuggingFaceEmbeddings(model_name=cfg["embedding_model_name"])

    # Clear old database files if explicit clean start is requested
    if force_recreate and os.path.exists(db_path):
        print(f"Clearing older vector data at '{db_path}'...")
        shutil.rmtree(db_path)

    print(f"Beginning batched vector persistence into collection '{collection}'...")
    vector_store = None
    batch_buffer = []

    for doc in documents:
        batch_buffer.append(doc)

        # Stream into vector database in small chunks to protect 4GB RAM laptop
        if len(batch_buffer) >= batch_size:
            if vector_store is None:
                vector_store = Chroma.from_documents(
                    documents=batch_buffer,
                    embedding=embeddings,
                    persist_directory=db_path,
                    collection_name=collection
                )
            else:
                vector_store.add_documents(documents=batch_buffer)

            print(f"Committed {len(batch_buffer)} chunks to Vector store. Memory safe.")
            batch_buffer = []

    # Process remaining elements left in array
    if batch_buffer:
        if vector_store is None:
            vector_store = Chroma.from_documents(
                documents=batch_buffer,
                embedding=embeddings,
                persist_directory=db_path,
                collection_name=collection
            )
        else:
            vector_store.add_documents(documents=batch_buffer)
    print(f"Committed final {len(batch_buffer)} chunks to Vector store.")

    print(f"✅ Database generation complete! Saved to: {db_path}")
