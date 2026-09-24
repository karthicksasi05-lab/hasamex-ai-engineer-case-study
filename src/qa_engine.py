from src.evidence_retriever import search_evidence


QUESTION_KEYWORDS = {
    "adoption": [
        "adoption",
        "growing",
        "increasing",
        "advanced",
        "standard",
        "access"
    ],

    "barriers": [
        "barrier",
        "cost",
        "funding",
        "budget",
        "capital",
        "training capacity",
        "hospital finances"
    ],

    "roi": [
        "roi",
        "economic",
        "economics",
        "financial",
        "finance",
        "cost",
        "budget",
        "procedure volume",
        "utilisation",
        "utilization"
    ],

    "training": [
        "training",
        "trained",
        "surgeon",
        "theatre staff",
        "clinical outcomes",
        "patient outcomes"
    ],

    "trend": [
        "growth",
        "growing",
        "increasing",
        "accelerate",
        "gradual",
        "annually",
        "outlook"
    ],

    "timeline": [
        "months",
        "weeks",
        "timeline",
        "budget cycle",
        "decision-making",
        "decision",
        "purchasing"
    ]
}


def identify_question_type(question):
    """
    Identify the main topic of the user's question.
    """

    question = question.lower()

    if any(word in question for word in [
        "barrier",
        "barriers",
        "holding adoption back"
    ]):
        return "barriers"

    if any(word in question for word in [
        "roi",
        "return on investment",
        "budget",
        "economic",
        "economics"
    ]):
        return "roi"

    if any(word in question for word in [
        "training",
        "trained",
        "clinical outcome",
        "clinical outcomes"
    ]):
        return "training"

    if any(word in question for word in [
        "trend",
        "future",
        "outlook",
        "3-5 years",
        "three to five"
    ]):
        return "trend"

    if any(word in question for word in [
        "timeline",
        "how long",
        "decision-making",
        "purchasing"
    ]):
        return "timeline"

    if "adoption" in question:
        return "adoption"

    return "adoption"


def answer_question(question, transcripts, parse_function):

    question_type = identify_question_type(question)

    keywords = QUESTION_KEYWORDS[question_type]

    results = []

    for filename, content in transcripts.items():

        records = parse_function(content)

        evidence = search_evidence(
            records,
            keywords
        )

        for item in evidence:

            results.append({
                "country": filename.replace(
                    "Transcript_",
                    ""
                ).replace(
                    ".txt",
                    ""
                ),
                "timestamp": item["timestamp"],
                "speaker": item["speaker"],
                "text": item["text"]
            })

    return results


def build_answer(question, results):
    """
    Build a simple transcript-grounded answer.

    This function does not generate new information.
    It only organizes the evidence already retrieved
    from the transcripts.
    """

    if not results:

        return (
            "The transcripts do not provide enough "
            "information to answer this question."
        )

    grouped = {}

    for result in results:

        country = result["country"]

        if country not in grouped:
            grouped[country] = []

        grouped[country].append(result)


    answer_parts = []

    for country, evidence in grouped.items():

        # Use the first relevant evidence segment
        # as the country-level answer.
        main_evidence = evidence[0]

        answer_parts.append(
            f"{country}: {main_evidence['text']}"
        )

    return "\n\n".join(answer_parts)