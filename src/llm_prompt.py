def build_grounded_prompt(question, evidence):
    """
    Build a strict grounded prompt for Gemini.

    Gemini can summarize and compare the retrieved evidence,
    but it must not add information that is not present.
    """

    country_evidence = {
        "France": [],
        "Germany": [],
        "UK": []
    }

    for item in evidence:
        country = item.get("country", "Unknown")

        if country not in country_evidence:
            country_evidence[country] = []

        country_evidence[country].append(item)

    evidence_text = ""

    for country in ["France", "Germany", "UK"]:

        evidence_text += f"\n===== {country} =====\n"

        if not country_evidence[country]:
            evidence_text += (
                "NO SUFFICIENTLY RELEVANT EVIDENCE RETRIEVED "
                "FOR THIS COUNTRY.\n"
            )
            continue

        for item in country_evidence[country]:

            evidence_text += (
                f"Expert: {item.get('speaker', 'Unknown')}\n"
                f"Timestamp: {item.get('timestamp', 'Unknown')}\n"
                f"Evidence: {item.get('text', '')}\n\n"
            )

    prompt = f"""
You are an interview research assistant analyzing expert
interview transcripts.

USER QUESTION:
{question}

You must answer the question using ONLY the transcript evidence
provided below.

IMPORTANT GROUNDING RULES:

1. The transcript evidence is the source of truth.
2. Do not use outside knowledge.
3. Do not invent facts, numbers, opinions, trends, or conclusions.
4. Never transfer information from one country to another.
5. Analyze each country independently.
6. A missing country does NOT mean the entire answer is impossible.
7. If evidence exists for a country, answer using that evidence.
8. If no sufficiently relevant evidence exists for a country, say:
   "The retrieved evidence does not provide enough information
   for this country."
9. You may provide a partial answer when evidence exists for
   some countries but not others.
10. Preserve all numbers exactly as stated.
11. Do not make claims stronger than the evidence supports.
12. Only state comparisons when the provided evidence directly
    supports the comparison.

RESPONSE FORMAT:

### France

Provide the evidence-supported answer for France.

### Germany

Provide the evidence-supported answer for Germany.

### UK

Provide the evidence-supported answer for the UK.

### Key Differences

List only differences directly supported by the evidence.

If there is not enough evidence to identify a difference,
do not create one.

### Evidence References

For each important factual statement, provide:

Country | Expert | Timestamp

Do not create timestamps or evidence that are not provided.

IMPORTANT:

If one country has no relevant evidence, do NOT respond with:

"The transcripts do not provide enough information to answer this."

for the entire question.

Instead, answer the countries that have evidence and explicitly
mark only the unsupported country as insufficient.

RETRIEVED TRANSCRIPT EVIDENCE:
{evidence_text}
"""

    return prompt

