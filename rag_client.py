import os
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings
from openai import OpenAI


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available persistent ChromaDB collections."""

    backends: Dict[str, Dict[str, str]] = {}
    current_dir = Path(".")

    excluded_directories = {
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
    }

    chroma_dirs: List[Path] = []

    for path in current_dir.rglob("*"):
        if not path.is_dir():
            continue

        if any(
            excluded in path.parts
            for excluded in excluded_directories
        ):
            continue

        if (
            "chroma" in path.name.lower()
            or (path / "chroma.sqlite3").exists()
        ):
            chroma_dirs.append(path)

    chroma_dirs = list(
        dict.fromkeys(chroma_dirs)
    )

    for chroma_dir in chroma_dirs:
        try:
            client = chromadb.PersistentClient(
                path=str(chroma_dir),
                settings=Settings(
                    anonymized_telemetry=False
                ),
            )

            collections = client.list_collections()

            for collection in collections:
                if hasattr(collection, "name"):
                    collection_object = collection
                    collection_name = collection.name
                else:
                    collection_name = str(collection)
                    collection_object = client.get_collection(
                        name=collection_name
                    )

                backend_key = (
                    f"{chroma_dir}:{collection_name}"
                )

                try:
                    document_count = (
                        collection_object.count()
                    )
                except Exception:
                    document_count = -1

                display_name = (
                    f"{chroma_dir} / "
                    f"{collection_name} "
                    f"({document_count} documents)"
                )

                backends[backend_key] = {
                    "path": str(chroma_dir),
                    "collection": collection_name,
                    "display_name": display_name,
                    "document_count": str(
                        document_count
                    ),
                }

        except Exception as exc:
            error_message = str(exc)

            if len(error_message) > 100:
                error_message = (
                    error_message[:100] + "..."
                )

            backend_key = (
                f"{chroma_dir}:error"
            )

            backends[backend_key] = {
                "path": str(chroma_dir),
                "collection": "",
                "display_name": (
                    f"{chroma_dir} "
                    f"(Error: {error_message})"
                ),
                "document_count": "0",
            }

    return backends


def initialize_rag_system(
    chroma_dir: str,
    collection_name: str,
):
    """Initialize and return a persistent Chroma collection."""

    client = chromadb.PersistentClient(
        path=chroma_dir,
        settings=Settings(
            anonymized_telemetry=False
        ),
    )

    return client.get_collection(
        name=collection_name
    )


def retrieve_documents(
    collection,
    query: str,
    n_results: int = 3,
    mission_filter: Optional[str] = None,
    openai_key: Optional[str] = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> Optional[Dict]:
    """
    Retrieve documents using the same OpenAI embedding model
    used when the documents were stored.
    """

    if not query or not query.strip():
        return {
            "documents": [[]],
            "metadatas": [[]],
            "ids": [[]],
            "distances": [[]],
        }

    if n_results <= 0:
        raise ValueError(
            "n_results must be greater than 0."
        )

    api_key = (
        openai_key
        or os.getenv("OPENAI_API_KEY")
    )

    if not api_key:
        return None

    try:
        client = OpenAI(
            api_key=api_key
        )

        query_response = client.embeddings.create(
            model=embedding_model,
            input=query,
        )

        query_embedding = (
            query_response.data[0].embedding
        )

        where_filter = None

        if (
            mission_filter
            and mission_filter.lower()
            not in {"all", "none"}
        ):
            where_filter = {
                "mission": mission_filter
            }

        query_kwargs = {
            "query_embeddings": [
                query_embedding
            ],
            "n_results": n_results,
        }

        if where_filter:
            query_kwargs["where"] = where_filter

        return collection.query(
            **query_kwargs
        )

    except Exception:
        return None


def format_context(
    documents: List[str],
    metadatas: List[Dict],
) -> str:
    """Format retrieved documents for the LLM."""

    if not documents:
        return ""

    context_parts = [
        "RELEVANT NASA MISSION DOCUMENTS:"
    ]

    for index, document in enumerate(
        documents
    ):
        metadata = (
            metadatas[index]
            if index < len(metadatas)
            else {}
        )

        metadata = metadata or {}

        mission = str(
            metadata.get(
                "mission",
                "Unknown Mission",
            )
        ).replace("_", " ").title()

        category = str(
            metadata.get(
                "document_category",
                "Unknown Category",
            )
        ).replace("_", " ").title()

        source = metadata.get(
            "source",
            "Unknown Source",
        )

        source_header = (
            f"\n--- Source {index + 1} ---\n"
            f"Mission: {mission}\n"
            f"Category: {category}\n"
            f"Source: {source}\n"
        )

        context_parts.append(
            source_header
        )

        document = str(document)

        max_document_length = 3000

        if len(document) > max_document_length:
            document = (
                document[:max_document_length]
                + "\n[Document truncated...]"
            )

        context_parts.append(
            document
        )

    return "\n".join(
        context_parts
    )
