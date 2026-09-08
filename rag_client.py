import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional
from pathlib import Path


def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory."""

    backends = {}
    current_dir = Path(".")

    # Look for directories that appear to contain ChromaDB data.
    # ChromaDB persistent directories normally contain files/directories
    # such as chroma.sqlite3.
    chroma_dirs = []

    for path in current_dir.rglob("*"):
        if path.is_dir():
            # Avoid scanning common virtual-environment/cache directories.
            if any(
                excluded in path.parts
                for excluded in [".git", "__pycache__", ".venv", "venv", "node_modules"]
            ):
                continue

            if (
                "chroma" in path.name.lower()
                or (path / "chroma.sqlite3").exists()
            ):
                chroma_dirs.append(path)

    # Remove duplicate paths
    chroma_dirs = list(dict.fromkeys(chroma_dirs))

    # Loop through each discovered directory
    for chroma_dir in chroma_dirs:
        try:
            # Initialize database client with directory path
            client = chromadb.PersistentClient(
                path=str(chroma_dir),
                settings=Settings(anonymized_telemetry=False)
            )

            # Retrieve available collections
            collections = client.list_collections()

            for collection in collections:
                # Handle both collection objects and collection names
                collection_name = (
                    collection.name
                    if hasattr(collection, "name")
                    else str(collection)
                )

                # Create a unique identifier
                backend_key = f"{chroma_dir}:{collection_name}"

                # Get document count
                try:
                    document_count = collection.count()
                except Exception:
                    document_count = -1

                # Create a user-friendly display name
                display_name = (
                    f"{chroma_dir} / {collection_name} "
                    f"({document_count} documents)"
                )

                # Store collection information
                backends[backend_key] = {
                    "path": str(chroma_dir),
                    "collection": collection_name,
                    "display_name": display_name,
                    "document_count": str(document_count),
                }

        except Exception as e:
            # Handle inaccessible directories gracefully
            error_message = str(e)

            # Truncate long error messages
            if len(error_message) > 100:
                error_message = error_message[:100] + "..."

            # Create fallback entry
            backend_key = f"{chroma_dir}:error"

            backends[backend_key] = {
                "path": str(chroma_dir),
                "collection": "",
                "display_name": f"{chroma_dir} (Error: {error_message})",
                "document_count": "0",
            }

    return backends


def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend."""

    # Create a persistent ChromaDB client
    client = chromadb.PersistentClient(
        path=chroma_dir,
        settings=Settings(anonymized_telemetry=False)
    )

    # Return the requested collection
    return client.get_collection(name=collection_name)


def retrieve_documents(
    collection,
    query: str,
    n_results: int = 3,
    mission_filter: Optional[str] = None
) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering."""

    # No filter by default
    where_filter = None

    # Apply mission-specific filtering when requested
    if mission_filter and mission_filter.lower() not in {"all", "none"}:
        where_filter = {"mission": mission_filter}

    try:
        # Execute similarity search
        if where_filter:
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter
            )
        else:
            results = collection.query(
                query_texts=[query],
                n_results=n_results
            )

        return results

    except Exception:
        # Return None when retrieval fails
        return None


def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context for the LLM."""

    if not documents:
        return ""

    # Initialize context with a header
    context_parts = [
        "RELEVANT NASA MISSION DOCUMENTS:"
    ]

    # Loop through documents and metadata
    for index, document in enumerate(documents):

        # Get metadata safely
        metadata = metadatas[index] if index < len(metadatas) else {}

        # Extract mission
        mission = metadata.get("mission", "Unknown Mission")
        mission = str(mission).replace("_", " ").title()

        # Extract source
        source = metadata.get("source", "Unknown Source")

        # Extract category
        category = metadata.get("category", "Unknown Category")
        category = str(category).replace("_", " ").title()

        # Create source header
        source_header = (
            f"\n--- Source {index + 1} ---\n"
            f"Mission: {mission}\n"
            f"Category: {category}\n"
            f"Source: {source}\n"
        )

        context_parts.append(source_header)

        # Make sure the document is a string
        document = str(document)

        # Truncate very long documents
        max_document_length = 3000

        if len(document) > max_document_length:
            document = document[:max_document_length] + "\n[Document truncated...]"

        context_parts.append(document)

    # Join everything into a single context string
    return "\n".join(context_parts)