#!/usr/bin/env python3
"""
ChromaDB Embedding Pipeline for NASA Space Mission Data - Text Files Only

This script reads parsed text data from various NASA space mission folders and creates
a permanent ChromaDB collection with OpenAI embeddings for RAG applications.
Optimized to process only text files to avoid duplication with JSON versions.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
import openai
from openai import OpenAI
import hashlib
import time
from datetime import datetime
import argparse
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("chroma_embedding_text_only.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class ChromaEmbeddingPipelineTextOnly:
    """Pipeline for creating ChromaDB collections with OpenAI embeddings."""

    def __init__(
        self,
        openai_api_key: str,
        chroma_persist_directory: str = "./chroma_db",
        collection_name: str = "nasa_space_missions_text",
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize the embedding pipeline.

        Args:
            openai_api_key: OpenAI API key
            chroma_persist_directory: Directory to persist ChromaDB
            collection_name: Name of the ChromaDB collection
            embedding_model: OpenAI embedding model to use
            chunk_size: Maximum size of text chunks
            chunk_overlap: Overlap between chunks
        """

        # Validate configuration
        if not openai_api_key:
            raise ValueError("OpenAI API key is required.")

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        # Initialize OpenAI client
        self.openai_api_key = openai_api_key
        self.client = OpenAI(api_key=openai_api_key)

        # Store configuration parameters
        self.chroma_persist_directory = chroma_persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Create the ChromaDB directory if necessary
        Path(chroma_persist_directory).mkdir(
            parents=True,
            exist_ok=True
        )

        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name
        )

        logger.info(
            "Initialized ChromaDB collection '%s' at '%s'",
            collection_name,
            chroma_persist_directory
        )

    def chunk_text(
        self,
        text: str,
        metadata: Dict[str, Any]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Split text into chunks with metadata.

        Args:
            text: Text to chunk
            metadata: Base metadata for the text

        Returns:
            List of (chunk_text, chunk_metadata) tuples
        """

        text = text.strip()

        if not text:
            return []

        # Handle short texts that do not need chunking
        if len(text) <= self.chunk_size:
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = 0
            chunk_metadata["total_chunks"] = 1

            return [(text, chunk_metadata)]

        chunks = []

        start = 0
        text_length = len(text)
        chunk_index = 0

        while start < text_length:
            end = min(start + self.chunk_size, text_length)

            # Try to break at a sentence boundary
            if end < text_length:
                search_start = max(start, end - 200)
                section = text[search_start:end]

                sentence_breaks = [
                    section.rfind(". "),
                    section.rfind("? "),
                    section.rfind("! "),
                    section.rfind("\n")
                ]

                best_break = max(sentence_breaks)

                if best_break != -1:
                    end = search_start + best_break + 1

            chunk = text[start:end].strip()

            if chunk:
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = chunk_index

                chunks.append((chunk, chunk_metadata))
                chunk_index += 1

            # Move forward while maintaining overlap
            next_start = end - self.chunk_overlap

            # Make sure progress is always made
            if next_start <= start:
                next_start = end

            start = next_start

        # Add total chunk count to every metadata dictionary
        total_chunks = len(chunks)

        final_chunks = []

        for chunk_text, chunk_metadata in chunks:
            chunk_metadata["total_chunks"] = total_chunks
            final_chunks.append((chunk_text, chunk_metadata))

        return final_chunks

    def check_document_exists(self, doc_id: str) -> bool:
        """
        Check if a document with the given ID already exists in the collection.

        Args:
            doc_id: Document ID to check

        Returns:
            True if document exists, False otherwise
        """

        try:
            result = self.collection.get(ids=[doc_id])

            return bool(result.get("ids"))

        except Exception as e:
            logger.error(
                "Error checking document %s: %s",
                doc_id,
                e
            )
            return False

    def update_document(
        self,
        doc_id: str,
        text: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Update an existing document in the collection.

        Args:
            doc_id: Document ID to update
            text: New text content
            metadata: New metadata

        Returns:
            True if successful, False otherwise
        """

        try:
            # Get new embedding
            embedding = self.get_embedding(text)

            # Update the document
            self.collection.update(
                ids=[doc_id],
                documents=[text],
                metadatas=[metadata],
                embeddings=[embedding]
            )

            logger.debug("Updated document: %s", doc_id)
            return True

        except Exception as e:
            logger.error(
                "Error updating document %s: %s",
                doc_id,
                e
            )
            return False

    def delete_documents_by_source(self, source_pattern: str) -> int:
        """
        Delete all documents from a specific source.

        Args:
            source_pattern: Pattern to match source names

        Returns:
            Number of documents deleted
        """

        try:
            all_docs = self.collection.get()

            ids_to_delete = []

            for i, metadata in enumerate(all_docs.get("metadatas", [])):
                if source_pattern in metadata.get("source", ""):
                    ids_to_delete.append(all_docs["ids"][i])

            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)

                logger.info(
                    "Deleted %d documents matching source pattern: %s",
                    len(ids_to_delete),
                    source_pattern
                )

                return len(ids_to_delete)

            logger.info(
                "No documents found matching source pattern: %s",
                source_pattern
            )

            return 0

        except Exception as e:
            logger.error(
                "Error deleting documents by source: %s",
                e
            )
            return 0

    def get_file_documents(self, file_path: Path) -> List[str]:
        """
        Get all document IDs for a specific file.

        Args:
            file_path: Path to the file

        Returns:
            List of document IDs for the file
        """

        try:
            source = file_path.stem
            mission = self.extract_mission_from_path(file_path)

            all_docs = self.collection.get()

            file_doc_ids = []

            for i, metadata in enumerate(
                all_docs.get("metadatas", [])
            ):
                if (
                    metadata.get("source") == source
                    and metadata.get("mission") == mission
                ):
                    file_doc_ids.append(all_docs["ids"][i])

            return file_doc_ids

        except Exception as e:
            logger.error(
                "Error getting file documents: %s",
                e
            )
            return []

    def get_embedding(self, text: str) -> List[float]:
        """
        Get OpenAI embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """

        if not text or not text.strip():
            raise ValueError("Cannot create an embedding for empty text.")

        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )

            return response.data[0].embedding

        except Exception as e:
            logger.error("Error generating embedding: %s", e)
            raise

    def generate_document_id(
        self,
        file_path: Path,
        metadata: Dict[str, Any]
    ) -> str:
        """
        Generate a stable document ID based on file path and chunk position.

        Format:
            mission_source_chunk_0001
        """

        mission = metadata.get("mission", "unknown")
        source = metadata.get("source", file_path.stem)
        chunk_index = metadata.get("chunk_index", 0)

        # Clean values so they are safe for use in IDs
        mission = str(mission).replace(" ", "_")
        source = str(source).replace(" ", "_")

        # Create a stable identifier
        raw_id = f"{mission}_{source}_chunk_{int(chunk_index):04d}"

        # Remove problematic characters
        safe_id = "".join(
            char if char.isalnum() or char in "_-" else "_"
            for char in raw_id
        )

        return safe_id

    def process_text_file(
        self,
        file_path: Path
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Process plain text files with enhanced metadata extraction.

        Args:
            file_path: Path to text file

        Returns:
            List of (text, metadata) tuples
        """

        try:
            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as f:
                content = f.read()

            if not content.strip():
                logger.warning(
                    "Skipping empty file: %s",
                    file_path
                )
                return []

            metadata = {
                "source": file_path.stem,
                "file_path": str(file_path),
                "file_type": "text",
                "content_type": "full_text",
                "mission": self.extract_mission_from_path(file_path),
                "data_type": self.extract_data_type_from_path(file_path),
                "document_category": (
                    self.extract_document_category_from_filename(
                        file_path.name
                    )
                ),
                "file_size": len(content),
                "processed_timestamp": datetime.now().isoformat()
            }

            return self.chunk_text(content, metadata)

        except Exception as e:
            logger.error(
                "Error processing text file %s: %s",
                file_path,
                e
            )
            return []

    def extract_mission_from_path(
        self,
        file_path: Path
    ) -> str:
        """Extract mission name from file path."""

        path_str = str(file_path).lower()

        if "apollo11" in path_str or "apollo_11" in path_str:
            return "apollo_11"

        if "apollo13" in path_str or "apollo_13" in path_str:
            return "apollo_13"

        if "challenger" in path_str:
            return "challenger"

        return "unknown"

    def extract_data_type_from_path(
        self,
        file_path: Path
    ) -> str:
        """Extract data type from file path."""

        path_str = str(file_path).lower()

        if "transcript" in path_str:
            return "transcript"

        if "textract" in path_str:
            return "textract_extracted"

        if "audio" in path_str:
            return "audio_transcript"

        if "flight_plan" in path_str:
            return "flight_plan"

        return "document"

    def extract_document_category_from_filename(
        self,
        filename: str
    ) -> str:
        """Extract document category from filename."""

        filename_lower = filename.lower()

        if "pao" in filename_lower:
            return "public_affairs_officer"

        if "cm" in filename_lower:
            return "command_module"

        if "tec" in filename_lower:
            return "technical"

        if "flight_plan" in filename_lower:
            return "flight_plan"

        if "mission_audio" in filename_lower:
            return "mission_audio"

        if "ntrs" in filename_lower:
            return "nasa_archive"

        if "19900066485" in filename_lower:
            return "technical_report"

        if "19710015566" in filename_lower:
            return "mission_report"

        if "full_text" in filename_lower:
            return "complete_document"

        return "general_document"

    def scan_text_files_only(
        self,
        base_path: str
    ) -> List[Path]:
        """
        Scan data directories for text files only.

        Args:
            base_path: Base directory path

        Returns:
            List of text file paths to process
        """

        base_path = Path(base_path)
        files_to_process = []

        data_dirs = [
            "apollo11",
            "apollo13",
            "challenger"
        ]

        for data_dir in data_dirs:
            dir_path = base_path / data_dir

            if dir_path.exists():
                logger.info(
                    "Scanning directory: %s",
                    dir_path
                )

                text_files = list(
                    dir_path.glob("**/*.txt")
                )

                files_to_process.extend(text_files)

                logger.info(
                    "Found %d text files in %s",
                    len(text_files),
                    data_dir
                )

        # Filter unwanted files
        filtered_files = []

        for file_path in files_to_process:
            if (
                file_path.name.startswith(".")
                or "summary" in file_path.name.lower()
                or file_path.suffix.lower() != ".txt"
            ):
                continue

            filtered_files.append(file_path)

        logger.info(
            "Total text files to process: %d",
            len(filtered_files)
        )

        # Log file breakdown by mission
        mission_counts = {}

        for file_path in filtered_files:
            mission = self.extract_mission_from_path(file_path)
            mission_counts[mission] = (
                mission_counts.get(mission, 0) + 1
            )

        logger.info("Files by mission:")

        for mission, count in mission_counts.items():
            logger.info(
                "  %s: %d files",
                mission,
                count
            )

        return filtered_files

    def add_documents_to_collection(
        self,
        documents: List[Tuple[str, Dict[str, Any]]],
        file_path: Path,
        batch_size: int = 50,
        update_mode: str = "skip"
    ) -> Dict[str, int]:
        """
        Add documents to ChromaDB collection in batches.

        Args:
            documents: List of (text, metadata) tuples
            file_path: Source file
            batch_size: Number of documents per batch
            update_mode: skip, update, or replace

        Returns:
            Dictionary containing added, updated, and skipped counts.
        """

        if not documents:
            return {
                "added": 0,
                "updated": 0,
                "skipped": 0
            }

        if update_mode not in {"skip", "update", "replace"}:
            raise ValueError(
                "update_mode must be 'skip', 'update', or 'replace'."
            )

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")

        stats = {
            "added": 0,
            "updated": 0,
            "skipped": 0
        }

        # Replace mode: delete all existing chunks from this source
        if update_mode == "replace":
            existing_ids = self.get_file_documents(file_path)

            if existing_ids:
                self.collection.delete(ids=existing_ids)

                logger.info(
                    "Removed %d existing chunks from %s",
                    len(existing_ids),
                    file_path
                )

        # Prepare documents for processing
        pending_documents = []

        for text, metadata in documents:

            doc_id = self.generate_document_id(
                file_path,
                metadata
            )

            exists = self.check_document_exists(doc_id)

            if exists and update_mode == "skip":
                stats["skipped"] += 1
                continue

            if exists and update_mode == "update":
                if self.update_document(
                    doc_id,
                    text,
                    metadata
                ):
                    stats["updated"] += 1
                else:
                    logger.error(
                        "Failed to update document: %s",
                        doc_id
                    )

                continue

            pending_documents.append(
                (doc_id, text, metadata)
            )

        # Process new documents in batches
        for batch_start in range(
            0,
            len(pending_documents),
            batch_size
        ):
            batch = pending_documents[
                batch_start:batch_start + batch_size
            ]

            ids = []
            texts = []
            metadatas = []
            embeddings = []

            for doc_id, text, metadata in batch:
                try:
                    embedding = self.get_embedding(text)

                    ids.append(doc_id)
                    texts.append(text)
                    metadatas.append(metadata)
                    embeddings.append(embedding)

                except Exception as e:
                    logger.error(
                        "Error generating embedding for %s: %s",
                        doc_id,
                        e
                    )

            if ids:
                try:
                    self.collection.add(
                        ids=ids,
                        documents=texts,
                        metadatas=metadatas,
                        embeddings=embeddings
                    )

                    stats["added"] += len(ids)

                    logger.info(
                        "Added %d documents from %s",
                        len(ids),
                        file_path
                    )

                except Exception as e:
                    logger.error(
                        "Error adding batch to collection: %s",
                        e
                    )

        return stats

    def process_all_text_data(
        self,
        base_path: str,
        update_mode: str = "skip"
    ) -> Dict[str, int]:
        """
        Process all text files and add them to ChromaDB.

        Args:
            base_path: Base directory containing data folders
            update_mode: skip, update, or replace

        Returns:
            Statistics about processed files.
        """

        stats = {
            "files_processed": 0,
            "documents_added": 0,
            "documents_updated": 0,
            "documents_skipped": 0,
            "errors": 0,
            "total_chunks": 0,
            "missions": {}
        }

        # Get files to process
        files = self.scan_text_files_only(base_path)

        for file_path in files:
            try:
                logger.info(
                    "Processing file: %s",
                    file_path
                )

                documents = self.process_text_file(file_path)

                if not documents:
                    logger.warning(
                        "No documents extracted from %s",
                        file_path
                    )
                    continue

                # Add documents to ChromaDB
                file_stats = self.add_documents_to_collection(
                    documents,
                    file_path,
                    update_mode=update_mode
                )

                mission = self.extract_mission_from_path(
                    file_path
                )

                # Initialize mission statistics
                if mission not in stats["missions"]:
                    stats["missions"][mission] = {
                        "files": 0,
                        "chunks": 0,
                        "added": 0,
                        "updated": 0,
                        "skipped": 0
                    }

                stats["missions"][mission]["files"] += 1
                stats["missions"][mission]["chunks"] += len(documents)
                stats["missions"][mission]["added"] += (
                    file_stats["added"]
                )
                stats["missions"][mission]["updated"] += (
                    file_stats["updated"]
                )
                stats["missions"][mission]["skipped"] += (
                    file_stats["skipped"]
                )

                stats["files_processed"] += 1
                stats["total_chunks"] += len(documents)
                stats["documents_added"] += file_stats["added"]
                stats["documents_updated"] += file_stats["updated"]
                stats["documents_skipped"] += file_stats["skipped"]

            except Exception as e:
                stats["errors"] += 1

                logger.error(
                    "Error processing %s: %s",
                    file_path,
                    e
                )

        return stats

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the ChromaDB collection."""

        try:
            document_count = self.collection.count()

            metadata = getattr(
                self.collection,
                "metadata",
                None
            )

            return {
                "collection_name": self.collection.name,
                "document_count": document_count,
                "metadata": metadata
            }

        except Exception as e:
            logger.error(
                "Error getting collection information: %s",
                e
            )

            return {
                "collection_name": self.collection_name,
                "document_count": 0,
                "metadata": None,
                "error": str(e)
            }

    def query_collection(
        self,
        query_text: str,
        n_results: int = 5
    ) -> Dict[str, Any]:
        """
        Query the collection for testing.

        Args:
            query_text: Query text
            n_results: Number of results to return

        Returns:
            Query results
        """

        if not query_text.strip():
            return {
                "documents": [[]],
                "metadatas": [[]],
                "ids": [[]]
            }

        try:
            return self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )

        except Exception as e:
            logger.error(
                "Error querying collection: %s",
                e
            )

            return {
                "error": str(e),
                "documents": [[]],
                "metadatas": [[]],
                "ids": [[]]
            }

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get detailed statistics about the collection."""

        try:
            all_docs = self.collection.get()

            if not all_docs.get("metadatas"):
                return {
                    "error": "No documents in collection"
                }

            stats = {
                "total_documents": len(
                    all_docs["metadatas"]
                ),
                "missions": {},
                "data_types": {},
                "document_categories": {},
                "file_types": {}
            }

            for metadata in all_docs["metadatas"]:
                mission = metadata.get(
                    "mission",
                    "unknown"
                )

                data_type = metadata.get(
                    "data_type",
                    "unknown"
                )

                doc_category = metadata.get(
                    "document_category",
                    "unknown"
                )

                file_type = metadata.get(
                    "file_type",
                    "unknown"
                )

                stats["missions"][mission] = (
                    stats["missions"].get(mission, 0) + 1
                )

                stats["data_types"][data_type] = (
                    stats["data_types"].get(data_type, 0) + 1
                )

                stats["document_categories"][doc_category] = (
                    stats["document_categories"].get(
                        doc_category,
                        0
                    ) + 1
                )

                stats["file_types"][file_type] = (
                    stats["file_types"].get(file_type, 0) + 1
                )

            return stats

        except Exception as e:
            logger.error(
                "Error getting collection stats: %s",
                e
            )

            return {
                "error": str(e)
            }


def main():
    """Main function."""

    parser = argparse.ArgumentParser(
        description="ChromaDB Embedding Pipeline for NASA Data"
    )

    parser.add_argument(
        "--data-path",
        default=".",
        help="Path to data directories"
    )

    parser.add_argument(
        "--openai-key",
        required=True,
        help="OpenAI API key"
    )

    parser.add_argument(
        "--chroma-dir",
        default="./chroma_db_openai",
        help="ChromaDB persist directory"
    )

    parser.add_argument(
        "--collection-name",
        default="nasa_space_missions_text",
        help="Collection name"
    )

    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="OpenAI embedding model"
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Text chunk size"
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Chunk overlap size"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for processing"
    )

    parser.add_argument(
        "--update-mode",
        choices=["skip", "update", "replace"],
        default="skip",
        help="How to handle existing documents"
    )

    parser.add_argument(
        "--test-query",
        help="Test query after processing"
    )

    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show collection statistics"
    )

    parser.add_argument(
        "--delete-source",
        help="Delete all documents from a specific source pattern"
    )

    args = parser.parse_args()

    # Initialize pipeline
    logger.info(
        "Initializing ChromaDB Embedding Pipeline..."
    )

    pipeline = ChromaEmbeddingPipelineTextOnly(
        openai_api_key=args.openai_key,
        chroma_persist_directory=args.chroma_dir,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )

    # Handle delete source operation
    if args.delete_source:
        deleted_count = pipeline.delete_documents_by_source(
            args.delete_source
        )

        logger.info(
            "Deleted %d documents matching source pattern: %s",
            deleted_count,
            args.delete_source
        )

        return

    # Show statistics only
    if args.stats_only:
        logger.info("Collection Statistics:")

        stats = pipeline.get_collection_stats()

        for key, value in stats.items():
            logger.info(
                "%s: %s",
                key,
                value
            )

        return

    # Process all data
    logger.info(
        "Starting text data processing with update mode: %s",
        args.update_mode
    )

    start_time = time.time()

    stats = pipeline.process_all_text_data(
        args.data_path,
        update_mode=args.update_mode
    )

    end_time = time.time()
    processing_time = end_time - start_time

    # Print results
    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)

    logger.info(
        "Files processed: %d",
        stats["files_processed"]
    )

    logger.info(
        "Total chunks created: %d",
        stats["total_chunks"]
    )

    logger.info(
        "Documents added to collection: %d",
        stats["documents_added"]
    )

    logger.info(
        "Documents updated in collection: %d",
        stats["documents_updated"]
    )

    logger.info(
        "Documents skipped (already exist): %d",
        stats["documents_skipped"]
    )

    logger.info(
        "Errors: %d",
        stats["errors"]
    )

    logger.info(
        "Processing time: %.2f seconds",
        processing_time
    )

    # Mission breakdown
    logger.info("\nMission breakdown:")

    for mission, mission_stats in stats["missions"].items():
        logger.info(
            "  %s: %d files, %d chunks",
            mission,
            mission_stats["files"],
            mission_stats["chunks"]
        )

        logger.info(
            "    Added: %d, Updated: %d, Skipped: %d",
            mission_stats["added"],
            mission_stats["updated"],
            mission_stats["skipped"]
        )

    # Collection information
    collection_info = pipeline.get_collection_info()

    logger.info(
        "\nCollection: %s",
        collection_info.get(
            "collection_name",
            "N/A"
        )
    )

    logger.info(
        "Total documents in collection: %s",
        collection_info.get(
            "document_count",
            "N/A"
        )
    )

    # Test query if provided
    if args.test_query:
        logger.info(
            "\nTesting query: '%s'",
            args.test_query
        )

        results = pipeline.query_collection(
            args.test_query
        )

        if results and "documents" in results:
            result_documents = results["documents"][0]

            logger.info(
                "Found %d results:",
                len(result_documents)
            )

            for i, doc in enumerate(
                result_documents[:3]
            ):
                logger.info(
                    "Result %d: %s...",
                    i + 1,
                    doc[:200]
                )

    logger.info(
        "Pipeline completed successfully!"
    )


if __name__ == "__main__":
    main()
