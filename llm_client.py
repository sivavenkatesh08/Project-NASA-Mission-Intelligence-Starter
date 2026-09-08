from typing import Dict, List

from openai import OpenAI


def generate_response(
    openai_key: str,
    user_message: str,
    context: str,
    conversation_history: List[Dict],
    model: str = "gpt-3.5-turbo",
) -> str:
    """Generate a response using OpenAI with RAG context and history."""

    system_prompt = """You are a NASA Mission Intelligence Assistant.

You are an expert in NASA missions, space exploration, spacecraft,
astronauts, mission procedures, and related technical information.

Answer the user's questions using the provided NASA document context
whenever possible.

If the answer cannot be found in the provided context, clearly state
that the information is not available in the provided documents.
Do not invent facts or unsupported information.

Be accurate, concise, and helpful.
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]

    if context:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Here is the relevant information retrieved "
                    "from the NASA mission documents:\n\n"
                    f"{context}\n\n"
                    "Use this information as the primary source "
                    "for answering the user's question."
                ),
            }
        )

    if conversation_history:
        messages.extend(
            conversation_history
        )

    messages.append(
        {
            "role": "user",
            "content": user_message,
        }
    )

    try:
        client = OpenAI(
            api_key=openai_key
        )

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=1000,
        )

        content = response.choices[0].message.content

        if not content:
            return "I could not generate a response."

        return content.strip()

    except Exception as exc:
        return (
            f"Error generating response: {exc}"
        )
