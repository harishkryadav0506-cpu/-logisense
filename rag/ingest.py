"""
Document ingestion pipeline for the RAG system.

Reads PDF documents from data/policies/, chunks them using
RecursiveCharacterTextSplitter, embeds with HuggingFace model,
and stores in a ChromaDB vector store.

Usage:
    python -m rag.ingest
"""

import logging
import os
import sys
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from rag.embeddings import get_embedding_model

logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
POLICIES_DIR = PROJECT_ROOT / "data" / "policies"
VECTOR_STORE_DIR = PROJECT_ROOT / "rag" / "vector_store"

# Chunking configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
COLLECTION_NAME = "logisense_policies"


def load_pdfs(directory: Path) -> list[dict]:
    """
    Load all PDF files from the given directory.

    Args:
        directory: Path to the directory containing PDF files.

    Returns:
        List of dicts with 'text', 'source', and 'page' keys.
    """
    from pypdf import PdfReader

    documents = []
    pdf_files = sorted(directory.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in {directory}")
        return documents

    for pdf_path in pdf_files:
        logger.info(f"Loading: {pdf_path.name}")
        try:
            reader = PdfReader(str(pdf_path))
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text and text.strip():
                    documents.append({
                        "text": text.strip(),
                        "source": pdf_path.name,
                        "page": page_num,
                    })
        except Exception as e:
            logger.error(f"Error reading {pdf_path.name}: {e}")
            continue

    logger.info(f"Loaded {len(documents)} pages from {len(pdf_files)} PDFs")
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Split documents into smaller chunks for embedding.

    Args:
        documents: List of document dicts from load_pdfs().

    Returns:
        List of chunk dicts with 'text', 'source', 'page', and 'chunk_id'.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    chunk_id = 0

    for doc in documents:
        text_chunks = splitter.split_text(doc["text"])
        for chunk_text in text_chunks:
            chunks.append({
                "text": chunk_text,
                "source": doc["source"],
                "page": doc["page"],
                "chunk_id": chunk_id,
            })
            chunk_id += 1

    logger.info(f"Created {len(chunks)} chunks from {len(documents)} pages")
    return chunks


def create_vector_store(
    chunks: list[dict],
    persist_directory: str | None = None,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    """
    Create and persist a ChromaDB vector store from document chunks.

    Args:
        chunks: List of chunk dicts from chunk_documents().
        persist_directory: Directory to persist the vector store.
        collection_name: Name of the ChromaDB collection.

    Returns:
        Chroma vector store instance.
    """
    if persist_directory is None:
        persist_directory = str(VECTOR_STORE_DIR)

    # Prepare texts and metadata
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [
        {
            "source": chunk["source"],
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"],
        }
        for chunk in chunks
    ]

    # Get embedding model
    embedding_model = get_embedding_model()

    logger.info(f"Creating vector store with {len(texts)} chunks...")

    # Create ChromaDB vector store
    vector_store = Chroma.from_texts(
        texts=texts,
        embedding=embedding_model,
        metadatas=metadatas,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )

    logger.info(f"Vector store created and persisted at {persist_directory}")
    return vector_store


def ingest(
    policies_dir: Path | None = None,
    force: bool = False,
) -> Chroma:
    """
    Full ingestion pipeline: load PDFs → chunk → embed → store.

    Args:
        policies_dir: Directory containing policy PDFs.
        force: If True, recreate vector store even if it exists.

    Returns:
        Chroma vector store instance.
    """
    if policies_dir is None:
        policies_dir = POLICIES_DIR

    persist_dir = str(VECTOR_STORE_DIR)

    # Check if vector store already exists
    if not force and os.path.exists(persist_dir):
        chroma_files = [f for f in os.listdir(persist_dir) if f != ".gitkeep"]
        if chroma_files:
            logger.info("Vector store already exists. Use force=True to recreate.")
            embedding_model = get_embedding_model()
            return Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=embedding_model,
                persist_directory=persist_dir,
            )

    # Run full pipeline
    logger.info("Starting document ingestion pipeline...")

    documents = load_pdfs(policies_dir)
    if not documents:
        logger.error("No documents found. Aborting ingestion.")
        sys.exit(1)

    chunks = chunk_documents(documents)
    vector_store = create_vector_store(chunks, persist_dir)

    logger.info("Ingestion pipeline completed successfully!")
    return vector_store


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    import argparse

    parser = argparse.ArgumentParser(description="Ingest policy documents into vector store")
    parser.add_argument("--force", action="store_true", help="Force recreate vector store")
    args = parser.parse_args()

    vector_store = ingest(force=args.force)

    # Quick verification
    results = vector_store.similarity_search("refund policy", k=3)
    print(f"\nVerification — Top 3 results for 'refund policy':")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. [{doc.metadata.get('source', 'unknown')}] {doc.page_content[:100]}...")
