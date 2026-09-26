import os
from langchain_text_splitters import MarkdownTextSplitter
from langchain_core.documents import Document


def extract_chunks_from_files(cfg: dict)->list[Document]:
    """
        Strategy 1: Reads a Markdown file, tracks custom HTML page markers,
        and chunks text into structured LangChain Document objects.
    """
    file_path = cfg["raw_doc_path"]
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Source markdown file not found at: {file_path}")

    print(f'Reading {file_path} and generating markdown layout chunks...')
    splitter = MarkdownTextSplitter(
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"]
    )

    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read()
    raw_chunks = splitter.split_text(full_text)

    current_page = "Unknown"
    processed_documents = []
    vector_store = None
    print(f"Total raw text snippets generated: {len(raw_chunks)}")
    print("Converting snippets to structured LangChain Documents...")

    for chunk in raw_chunks:
        # Detect if this specific text snippet contains our visual page boundary marker
        # Example: <!-- PAGE 12 -->
        if "<!-- PAGE " in chunk:
            try:
                # Extract the page integer out of the string fragment
                parts = chunk.split("<!-- PAGE ")
                if len(parts) > 1:
                    current_page = parts[1].split(" -->")[0]
            except Exception:
                pass  # Fallback safely if string parsing experiences structural anomalies

        # Turn the string into a formal LangChain Document object
        doc = Document(
            page_content=chunk,
            metadata={"source": file_path, "page": current_page}
        )
        processed_documents.append(doc)

    return processed_documents


