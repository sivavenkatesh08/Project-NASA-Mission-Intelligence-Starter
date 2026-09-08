import os
from typing import Dict, List, Optional

from openai import AsyncOpenAI

from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory

try:
    from ragas.metrics.collections import (
        AnswerRelevancy,
        Faithfulness,
        ContextPrecisionWithoutReference,
        BleuScore,
        RougeScore,
    )

    RAGAS_AVAILABLE = True

except ImportError:
    RAGAS_AVAILABLE = False


def _get_score_value(result) -> float:
    """Extract a numeric score from a RAGAS MetricResult."""

    value = getattr(
        result,
        "value",
        result,
    )

    return float(value)


def evaluate_response_quality(
    question: str,
    answer: str,
    contexts: List[str],
    reference: Optional[str] = None,
    openai_key: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluate a RAG response using RAGAS 0.4.x metrics.

    Without a reference answer, evaluates:
    - Faithfulness
    - Answer Relevancy
    - Context Precision

    When a reference answer is supplied, also evaluates:
    - BLEU
    - ROUGE
    """

    if not RAGAS_AVAILABLE:
        return {
            "error": "RAGAS is not available."
        }

    api_key = (
        openai_key
        or os.getenv("OPENAI_API_KEY")
    )

    if not api_key:
        return {
            "error": "OpenAI API key is required for RAGAS evaluation."
        }

    if not question.strip():
        return {
            "error": "Question cannot be empty."
        }

    if not answer.strip():
        return {
            "error": "Answer cannot be empty."
        }

    if not contexts:
        return {
            "error": "At least one retrieved context is required."
        }

    try:
        openai_client = AsyncOpenAI(
            api_key=api_key
        )

        evaluator_llm = llm_factory(
            "gpt-3.5-turbo",
            client=openai_client,
        )

        evaluator_embeddings = embedding_factory(
            provider="openai",
            model="text-embedding-3-small",
            client=openai_client,
        )

        scores: Dict[str, float] = {}

        # -------------------------------------------------------------
        # RAG-specific metrics
        # -------------------------------------------------------------

        faithfulness = Faithfulness(
            llm=evaluator_llm
        )

        answer_relevancy = AnswerRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        )

        context_precision = ContextPrecisionWithoutReference(
            llm=evaluator_llm
        )

        faithfulness_result = faithfulness.score(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
        )

        scores["faithfulness"] = _get_score_value(
            faithfulness_result
        )

        relevancy_result = answer_relevancy.score(
            user_input=question,
            response=answer,
        )

        scores["answer_relevancy"] = _get_score_value(
            relevancy_result
        )

        context_result = context_precision.score(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
        )

        scores["context_precision"] = _get_score_value(
            context_result
        )

        # -------------------------------------------------------------
        # Reference-based metrics
        # -------------------------------------------------------------

        if reference and reference.strip():
            bleu = BleuScore()

            rouge = RougeScore(
                rouge_type="rougeL",
                mode="fmeasure",
            )

            bleu_result = bleu.score(
                reference=reference,
                response=answer,
            )

            scores["bleu"] = _get_score_value(
                bleu_result
            )

            rouge_result = rouge.score(
                reference=reference,
                response=answer,
            )

            scores["rouge"] = _get_score_value(
                rouge_result
            )

        return scores

    except Exception as exc:
        return {
            "error": str(exc)
        }
