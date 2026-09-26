import os
import re

from langchain_text_splitters import MarkdownTextSplitter
from langchain_core.documents import Document


def extract_chunks_from_files(cfg: dict) -> list[Document]:
    """
    Reads the consolidated Markdown file, splits it into chunks,
    and creates LangChain Document objects with source and
    PDF batch metadata.

    Note:
    The current ingestion pipeline does not reliably preserve
    individual PDF page numbers, so this code does NOT fabricate
    page numbers.
    """

    file_path = cfg["raw_doc_path"]

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Source markdown file not found at: {file_path}"
        )

    print(f"Reading {file_path}...")
    print("Generating Markdown chunks...")

    splitter = MarkdownTextSplitter(
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"]
    )

    # ---------------------------------------------------------
    # 1. Read complete Markdown file
    # ---------------------------------------------------------

    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    # ---------------------------------------------------------
    # 2. Split text into chunks
    # ---------------------------------------------------------

    raw_chunks = splitter.split_text(full_text)

    print(f"Total raw chunks generated: {len(raw_chunks)}")
    print("Converting chunks to LangChain Documents...")

    # ---------------------------------------------------------
    # 3. Convert chunks to LangChain Documents
    # ---------------------------------------------------------

    processed_documents = []

    current_batch = "Unknown"

    for chunk in raw_chunks:

        # -----------------------------------------------------
        # Look for the PDF batch marker
        #
        # Example:
        # <!-- PDF BATCH 21-40 -->
        # -----------------------------------------------------

        batch_match = re.search(
            r"<!--\s*PDF BATCH\s+(\d+)-(\d+)\s*-->",
            chunk
        )

        if batch_match:
            batch_start = batch_match.group(1)
            batch_end = batch_match.group(2)

            current_batch = f"{batch_start}-{batch_end}"

        # -----------------------------------------------------
        # Create LangChain Document
        # -----------------------------------------------------

        doc = Document(
            page_content=chunk,
            metadata={
                "source": file_path,
                "batch": current_batch
            }
        )

        processed_documents.append(doc)

    # ---------------------------------------------------------
    # 4. Print some information for verification
    # ---------------------------------------------------------

    print(
        f"Successfully created "
        f"{len(processed_documents)} LangChain Documents."
    )

    # Show first few chunks for verification
    print("\nSample metadata:")

    for i, doc in enumerate(processed_documents[:5]):
        print(f"\nChunk {i + 1}")
        print(f"Metadata: {doc.metadata}")
        print(f"Content preview: {doc.page_content[:200]}...")

    return processed_documents