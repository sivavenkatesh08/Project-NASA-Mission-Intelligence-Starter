from typing import Dict, List
from openai import OpenAI


def generate_response(
    openai_key: str,
    user_message: str,
    context: str,
    conversation_history: List[Dict],
    model: str = "gpt-3.5-turbo"
) -> str:
    """Generate a response using OpenAI with retrieved context and conversation history."""

    # Define the system prompt
    system_prompt = """You are a NASA Mission Intelligence Assistant.
You are an expert in NASA missions, space exploration, spacecraft,
astronauts, mission procedures, and related technical information.

Answer the user's questions using the provided context whenever possible.
If the answer cannot be found in the provided context, clearly state that
the information is not available in the provided documents rather than
making up information.

Be accurate, concise, and helpful.
"""

    # Create the messages list starting with the system message
    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    # Add retrieved context to the conversation
    if context:
        messages.append(
            {
                "role": "system",
                "content": f"""Here is the relevant information retrieved
from the NASA mission documents:

{context}

Use this information as the primary source for answering the user's question."""
            }
        )

    # Add previous conversation history
    if conversation_history:
        messages.extend(conversation_history)

    # Add the current user message
    messages.append(
        {
            "role": "user",
            "content": user_message
        }
    )

    # Create the OpenAI client
    client = OpenAI(api_key=openai_key)

    try:
        # Send the request to OpenAI
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=1000
        )

        # Return the generated response
        return response.choices[0].message.content.strip()

    except Exception as e:
        # Handle API or connection errors gracefully
        return f"Error generating response: {str(e)}"
