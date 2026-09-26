import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from Utils.config_loader import load_rag_config


def get_biencoder_retriever(cfg: dict):
    """
    Initializes and returns a Bi-Encoder vector store retriever.

    The embedding model converts the user's query into a vector.
    Chroma then performs similarity search against the stored
    document embeddings.
    """

    db_path = cfg["persist_directory_path"]
    collection = cfg["collection_name"]
    model_name = cfg["embedding_model_name"]

    # ---------------------------------------------------------
    # 1. Check that the vector database exists
    # ---------------------------------------------------------

    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"Vector Database not found at '{db_path}'. "
            f"Please run main.py first."
        )

    # ---------------------------------------------------------
    # 2. Load the same embedding model used during indexing
    # ---------------------------------------------------------

    print(f"Loading Bi-Encoder embedding model: {model_name}...")

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name
    )

    # ---------------------------------------------------------
    # 3. Connect to existing Chroma collection
    # ---------------------------------------------------------

    print(
        f"Connecting to Chroma collection: '{collection}'..."
    )

    vector_store = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
        collection_name=collection
    )

    # ---------------------------------------------------------
    # 4. Create similarity retriever
    # ---------------------------------------------------------

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 5
        }
    )

    return retriever


def query_vector_store(
    query_text: str,
    strategy_name: str = "markdown_fixed_size"
):
    """
    Executes semantic similarity search using the Bi-Encoder
    and prints the retrieved chunks with their metadata.
    """

    # ---------------------------------------------------------
    # 1. Load configuration
    # ---------------------------------------------------------

    cfg = load_rag_config(
        strategy_name=strategy_name
    )

    # ---------------------------------------------------------
    # 2. Initialize retriever
    # ---------------------------------------------------------

    retriever = get_biencoder_retriever(cfg)

    print(
        f"\n🔍 Executing Bi-Encoder Search for: "
        f"'{query_text}'"
    )

    # ---------------------------------------------------------
    # 3. Retrieve top-k documents
    # ---------------------------------------------------------

    matching_docs = retriever.invoke(query_text)

    # ---------------------------------------------------------
    # 4. Display results
    # ---------------------------------------------------------

    print(
        f"\n--- Found {len(matching_docs)} "
        f"Relevant Source Text Snippets ---"
    )

    for i, doc in enumerate(matching_docs, 1):

        source_file = doc.metadata.get(
            "source",
            "Unknown"
        )

        batch = doc.metadata.get(
            "batch",
            "Unknown"
        )

        print(
            f"\n[Match {i}] | "
            f"📄 PDF Batch: {batch} | "
            f"File: {source_file}"
        )

        print("-" * 70)

        print(
            doc.page_content.strip()
        )

        print("-" * 70)


if __name__ == "__main__":

    test_query = (
        "What is the bank credit growth trend "
        "for commercial banks?"
    )

    query_vector_store(
        query_text=test_query
    )