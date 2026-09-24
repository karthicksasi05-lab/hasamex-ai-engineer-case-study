import os
from google import genai


def generate_answer(prompt):
    """
    Generate a grounded answer using Gemini.

    Handles:
    - Missing API key
    - Gemini rate limits (429)
    - Other API errors
    """

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable is not set."
        )

    try:
        client = genai.Client(api_key=api_key)

        interaction = client.interactions.create(
            model="gemini-3.6-flash",
            input=prompt
        )

        return interaction.output_text

    except Exception as error:

        error_message = str(error)

        # -----------------------------------------
        # GEMINI RATE LIMIT / QUOTA ERROR
        # -----------------------------------------

        if "429" in error_message or "Rate limit exceeded" in error_message:

            raise RuntimeError(
                "Gemini API rate limit reached.\n\n"
                "The Free Tier daily request limit has been reached. "
                "The transcript retrieval pipeline is still working "
                "correctly, but AI answer generation is temporarily "
                "unavailable.\n\n"
                "Please wait until the Gemini quota resets before "
                "trying AI answer generation again."
            )

        # -----------------------------------------
        # OTHER GEMINI API ERRORS
        # -----------------------------------------

        raise RuntimeError(
            f"Gemini API error: {error_message}"
        )

