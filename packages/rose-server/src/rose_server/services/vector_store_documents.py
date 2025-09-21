"""Pure functions for vector store document operations."""

import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from rose_server.entities.files import UploadedFile
from rose_server.entities.vector_stores import Document


def prepare_documents_and_embeddings(
    uploaded_file: UploadedFile,
    vector_store_id: str,
    chunks: Sequence[Any],
    embeddings: List[np.ndarray],
    decode_errors: bool,
) -> Tuple[List[Document], List[Dict[str, Any]], int]:
    """Prepare documents and embedding data for storage.

    Returns:
        Tuple of (documents, embedding_data, created_at)
    """
    created_at = int(time.time())
    documents = []
    embedding_data = []

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        meta = {
            "file_id": uploaded_file.id,
            "filename": uploaded_file.filename,
            "total_chunks": len(chunks),
            "start_index": chunk.start_index,
            "end_index": chunk.end_index,
            "decode_errors": decode_errors,
        }

        doc = Document(
            id=f"{uploaded_file.id}#{i}",
            vector_store_id=vector_store_id,
            chunk_index=i,
            content=chunk.text,
            meta=meta,
            created_at=created_at,
        )
        documents.append(doc)

        embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
        embedding_data.append({"doc_id": doc.id, "embedding": embedding_blob})

    return documents, embedding_data, created_at


def prepare_embedding_deletion_params(doc_ids: List[str]) -> Tuple[str, Dict[str, str]]:
    """Prepare SQL placeholders and parameters for deleting embeddings.

    Returns:
        Tuple of (placeholders_string, params_dict)
    """
    placeholders = ", ".join([f":doc_id_{i}" for i in range(len(doc_ids))])
    params = {f"doc_id_{i}": doc_id for i, doc_id in enumerate(doc_ids)}
    return placeholders, params
