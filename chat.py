#!/usr/bin/env python3
"""
NASA RAG Chat with RAGAS Evaluation Integration

Interactive Streamlit application for chatting with NASA mission documents
using Retrieval-Augmented Generation (RAG) with optional RAGAS evaluation.
"""

import os
from typing import Dict, List, Optional

import streamlit as st

import ragas_evaluator
import rag_client
import llm_client


# RAGAS availability
try:
    from ragas import SingleTurnSample 

    RAGAS_AVAILABLE = True

except ImportError:
    RAGAS_AVAILABLE = False


# Page configuration
st.set_page_config(
    page_title="NASA RAG Chat with Evaluation",
    page_icon="🚀",
    layout="wide"
)


def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends."""

    return rag_client.discover_chroma_backends()


@st.cache_resource
def initialize_rag_system(
    chroma_dir: str,
    collection_name: str
):
    """
    Initialize and cache a ChromaDB RAG collection.

    Args:
        chroma_dir: ChromaDB persistence directory.
        collection_name: Name of the collection.

    Returns:
        Initialized ChromaDB collection.
    """

    return rag_client.initialize_rag_system(
        chroma_dir,
        collection_name
    )


def retrieve_documents(
    collection,
    query: str,
    n_results: int = 3,
    mission_filter: Optional[str] = None
) -> Optional[Dict]:
    """
    Retrieve relevant documents from ChromaDB.

    Args:
        collection: ChromaDB collection.
        query: User's question.
        n_results: Number of documents to retrieve.
        mission_filter: Optional mission filter.

    Returns:
        ChromaDB query results or None if retrieval fails.
    """

    try:
        return rag_client.retrieve_documents(
            collection,
            query,
            n_results,
            mission_filter
        )

    except Exception as e:
        st.error(f"Error retrieving documents: {e}")
        return None


def format_context(
    documents: List[str],
    metadatas: List[Dict]
) -> str:
    """Format retrieved documents into LLM context."""

    return rag_client.format_context(
        documents,
        metadatas
    )


def generate_response(
    openai_key: str,
    user_message: str,
    context: str,
    conversation_history: List[Dict],
    model: str = "gpt-3.5-turbo"
) -> str:
    """Generate an answer using the OpenAI LLM."""

    try:
        return llm_client.generate_response(
            openai_key,
            user_message,
            context,
            conversation_history,
            model
        )

    except Exception as e:
        return f"Error generating response: {e}"


def evaluate_response_quality(
    question: str,
    answer: str,
    contexts: List[str]
) -> Dict[str, float]:
    """Evaluate response quality using RAGAS."""

    try:
        return ragas_evaluator.evaluate_response_quality(
            question,
            answer,
            contexts
        )

    except Exception as e:
        return {
            "error": f"Evaluation failed: {str(e)}"
        }


def display_evaluation_metrics(
    scores: Dict[str, float]
):
    """Display RAGAS evaluation metrics in the sidebar."""

    if not scores:
        return

    if "error" in scores:
        st.sidebar.error(
            f"Evaluation Error: {scores['error']}"
        )
        return

    st.sidebar.subheader("📊 Response Quality")

    for metric_name, score in scores.items():

        if not isinstance(score, (int, float)):
            continue

        # Keep score within progress-bar range
        progress_value = max(
            0.0,
            min(1.0, float(score))
        )

        st.sidebar.metric(
            label=metric_name.replace(
                "_",
                " "
            ).title(),
            value=f"{score:.3f}"
        )

        st.sidebar.progress(progress_value)


def main():
    """Run the NASA RAG Streamlit application."""

    st.title(
        "🚀 NASA Space Mission Chat with Evaluation"
    )

    st.markdown(
        "Chat with AI about NASA space missions "
        "using retrieved mission documents with "
        "real-time quality evaluation."
    )

    # ---------------------------------------------------------
    # Session State
    # ---------------------------------------------------------

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "current_backend" not in st.session_state:
        st.session_state.current_backend = None

    if "last_evaluation" not in st.session_state:
        st.session_state.last_evaluation = None

    if "last_contexts" not in st.session_state:
        st.session_state.last_contexts = []

    # ---------------------------------------------------------
    # Sidebar Configuration
    # ---------------------------------------------------------

    with st.sidebar:

        st.header("🔧 Configuration")

        # Discover ChromaDB backends
        with st.spinner(
            "Discovering ChromaDB backends..."
        ):
            available_backends = (
                discover_chroma_backends()
            )

        if not available_backends:

            st.error(
                "No ChromaDB backends found!"
            )

            st.info(
                "Please run the embedding pipeline first."
            )

            st.stop()

        # -----------------------------------------------------
        # Backend Selection
        # -----------------------------------------------------

        st.subheader("📊 ChromaDB Backend")

        backend_options = {
            key: value["display_name"]
            for key, value in available_backends.items()
            if value.get("collection")
        }

        if not backend_options:

            st.error(
                "No valid ChromaDB collections found."
            )

            st.info(
                "Run the embedding pipeline to create "
                "a collection."
            )

            st.stop()

        selected_backend_key = st.selectbox(
            "Select Document Collection",
            options=list(
                backend_options.keys()
            ),
            format_func=lambda key:
                backend_options[key],
            help=(
                "Choose which NASA document collection "
                "to use for retrieval."
            )
        )

        selected_backend = available_backends[
            selected_backend_key
        ]

        # -----------------------------------------------------
        # OpenAI Settings
        # -----------------------------------------------------

        st.subheader("🔑 OpenAI Settings")

        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv(
                "OPENAI_API_KEY",
                ""
            ),
            help=(
                "Enter your OpenAI API key."
            )
        )

        if not openai_key:
            st.warning(
                "Please enter your OpenAI API key."
            )
            st.stop()

        # Set environment variable for components
        os.environ["OPENAI_API_KEY"] = openai_key
        os.environ["CHROMA_OPENAI_API_KEY"] = openai_key

        # -----------------------------------------------------
        # Model Selection
        # -----------------------------------------------------

        model_choice = st.selectbox(
            "OpenAI Model",
            options=[
                "gpt-3.5-turbo",
                "gpt-4",
                "gpt-4-turbo-preview"
            ],
            help=(
                "Choose the OpenAI model used "
                "to generate responses."
            )
        )

        # -----------------------------------------------------
        # Retrieval Settings
        # -----------------------------------------------------

        st.subheader("🔍 Retrieval Settings")

        n_docs = st.slider(
            "Documents to retrieve",
            min_value=1,
            max_value=10,
            value=3
        )

        # -----------------------------------------------------
        # Mission Filter
        # -----------------------------------------------------

        mission_filter = st.selectbox(
            "Mission Filter",
            options=[
                "all",
                "apollo_11",
                "apollo_13",
                "challenger"
            ],
            format_func=lambda value:
                "All Missions"
                if value == "all"
                else value.replace(
                    "_",
                    " "
                ).title(),
            help=(
                "Restrict document retrieval "
                "to a specific NASA mission."
            )
        )

        # -----------------------------------------------------
        # Evaluation Settings
        # -----------------------------------------------------

        st.subheader("📊 Evaluation Settings")

        enable_evaluation = st.checkbox(
            "Enable RAGAS Evaluation",
            value=RAGAS_AVAILABLE
        )

        if enable_evaluation and not RAGAS_AVAILABLE:
            st.warning(
                "RAGAS is not available. "
                "Install it with: pip install ragas"
            )

    # ---------------------------------------------------------
    # Initialize RAG System
    # ---------------------------------------------------------

    try:

        with st.spinner(
            "Initializing RAG system..."
        ):

            collection = initialize_rag_system(
                selected_backend["path"],
                selected_backend["collection"]
            )

    except Exception as e:

        st.error(
            f"Failed to initialize RAG system: {e}"
        )

        st.stop()

    # ---------------------------------------------------------
    # Display Last Evaluation
    # ---------------------------------------------------------

    if (
        st.session_state.last_evaluation
        and enable_evaluation
    ):
        display_evaluation_metrics(
            st.session_state.last_evaluation
        )

    # ---------------------------------------------------------
    # Display Previous Chat Messages
    # ---------------------------------------------------------

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):
            st.markdown(
                message["content"]
            )

    # ---------------------------------------------------------
    # Chat Input
    # ---------------------------------------------------------

    prompt = st.chat_input(
        "Ask about NASA space missions..."
    )

    if prompt:

        # -----------------------------------------------------
        # Add User Message
        # -----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        # -----------------------------------------------------
        # Generate Assistant Response
        # -----------------------------------------------------

        with st.chat_message("assistant"):

            with st.spinner(
                "Searching documents and "
                "generating response..."
            ):

                # Retrieve relevant documents
                docs_result = retrieve_documents(
                    collection,
                    prompt,
                    n_docs,
                    mission_filter
                )

                context = ""
                contexts_list = []

                # -------------------------------------------------
                # Process Retrieval Results
                # -------------------------------------------------

                if (
                    docs_result
                    and docs_result.get("documents")
                ):

                    retrieved_documents = (
                        docs_result["documents"][0]
                    )

                    retrieved_metadatas = (
                        docs_result.get(
                            "metadatas",
                            [[]]
                        )[0]
                    )

                    context = format_context(
                        retrieved_documents,
                        retrieved_metadatas
                    )

                    contexts_list = (
                        retrieved_documents
                    )

                    st.session_state.last_contexts = (
                        contexts_list
                    )

                # -------------------------------------------------
                # Generate Response
                # -------------------------------------------------

                # Exclude the current question from history
                # because it is passed separately to the LLM.
                conversation_history = (
                    st.session_state.messages[:-1]
                )

                response = generate_response(
                    openai_key,
                    prompt,
                    context,
                    conversation_history,
                    model_choice
                )

                st.markdown(response)

                # -------------------------------------------------
                # RAGAS Evaluation
                # -------------------------------------------------

                if (
                    enable_evaluation
                    and RAGAS_AVAILABLE
                    and contexts_list
                ):

                    with st.spinner(
                        "Evaluating response quality..."
                    ):

                        evaluation_scores = (
                            evaluate_response_quality(
                                prompt,
                                response,
                                contexts_list
                            )
                        )

                        st.session_state.last_evaluation = (
                            evaluation_scores
                        )

        # -----------------------------------------------------
        # Save Assistant Message
        # -----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response
            }
        )

        # Refresh the application so that
        # evaluation metrics appear immediately.
        st.rerun()


if __name__ == "__main__":
    main()