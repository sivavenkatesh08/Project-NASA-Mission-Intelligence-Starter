from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List

# RAGAS imports
try:
    from ragas import SingleTurnSample
    from ragas.metrics import (
        BleuScore,
        NonLLMContextPrecisionWithReference,
        ResponseRelevancy,
        Faithfulness,
        RougeScore,
    )
    from ragas import evaluate

    RAGAS_AVAILABLE = True

except ImportError:
    RAGAS_AVAILABLE = False


def evaluate_response_quality(
    question: str,
    answer: str,
    contexts: List[str]
) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics."""

    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}

    try:
        # Create evaluator LLM
        evaluator_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0
            )
        )

        # Create evaluator embeddings
        evaluator_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model="text-embedding-3-small"
            )
        )

        # Define evaluation metrics
        metrics = [
            BleuScore(),
            NonLLMContextPrecisionWithReference(),
            ResponseRelevancy(
                llm=evaluator_llm,
                embeddings=evaluator_embeddings
            ),
            Faithfulness(
                llm=evaluator_llm
            ),
            RougeScore(),
        ]

        # Create a RAGAS sample
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
        )

        # Evaluate the response
        result = evaluate(
            dataset=[sample],
            metrics=metrics,
        )

        # Convert evaluation result to dictionary
        result_dict = dict(result)

        # Return numerical scores
        scores = {}

        for metric_name, score in result_dict.items():
            try:
                if score is not None:
                    scores[metric_name] = float(score)
            except (TypeError, ValueError):
                continue

        return scores

    except Exception as e:
        return {
            "error": str(e)
        }
