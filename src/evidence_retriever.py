def get_expert_records(records):
    """
    Keep only expert responses.
    Removes interviewer questions.
    """

    expert_records = []

    for record in records:

        if record["speaker"].lower() != "interviewer":

            expert_records.append(record)

    return expert_records


def search_evidence(records, keywords):
    """
    Find transcript segments containing relevant keywords.
    """

    expert_records = get_expert_records(records)

    results = []

    for record in expert_records:

        text = record["text"].lower()

        matched_keywords = []

        for keyword in keywords:

            if keyword.lower() in text:
                matched_keywords.append(keyword)

        if matched_keywords:

            results.append({
                "timestamp": record["timestamp"],
                "speaker": record["speaker"],
                "text": record["text"],
                "matched_keywords": matched_keywords
            })

    return results
QUESTION_KEYWORDS = {

    1: [
        "adoption",
        "growing",
        "increasing",
        "advanced",
        "standard"
    ],

    2: [
        "barrier",
        "cost",
        "funding",
        "budget",
        "capital",
        "training"
    ],

    3: [
        "ROI",
        "economic",
        "economics",
        "cost",
        "budget",
        "utilisation",
        "procedure volume"
    ],

    4: [
        "training",
        "trained",
        "surgeon",
        "clinical outcomes",
        "outcomes"
    ],

    5: [
        "growth",
        "gradual",
        "accelerate",
        "outlook",
        "annually",
        "3–5"
    ],

    6: [
        "timeline",
        "months",
        "weeks",
        "decision",
        "purchasing"
    ]
}