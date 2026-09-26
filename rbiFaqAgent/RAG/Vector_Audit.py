import os
import shutil
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from Utils.config_loader import load_rag_config

def vector_audit(cfg:dict):
    #cfg = load_rag_config(strategy_name="markdown_fixed_size")

    print("\n--- Running Database Audit ---")
    db_path = cfg["persist_directory_path"]
    collection = cfg["collection_name"]
    batch_size = cfg["batch_size"]
    embeddings = HuggingFaceEmbeddings(model_name=cfg["embedding_model_name"])
    try:
        # Load the persisted store directly from disk
        db_audit = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings,
            collection_name=collection
        )

        # Retrieve all document structural components from the collection
        collection_data = db_audit.get()
        total_vectors = len(collection_data['ids'])

        print(f"✅ Success! Total vectors securely stored on disk: {total_vectors}")

        # Print a tiny sample layout snippet to verify structural data integrity
        if total_vectors > 0:
            print("\n--- Structural First Chunk Sample ---")
            print(f"Metadata State: {collection_data['metadatas'][0]}")
            print(f"Content Preview: {collection_data['documents'][0][:150]}...")

    except Exception as audit_error:
        print(f"⚠️ Could not audit database: {audit_error}")
