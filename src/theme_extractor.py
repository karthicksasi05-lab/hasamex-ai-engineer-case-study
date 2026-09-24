THEMES = {
    "Economics & Cost": [
        "cost",
        "economic",
        "economics",
        "roi",
        "budget",
        "funding",
        "capital",
        "finance",
        "financial",
        "maintenance",
        "procedure volume",
        "utilisation",
        "utilization"
    ],

    "Training & Skills": [
        "training",
        "trained",
        "surgeon",
        "theatre staff",
        "skills"
    ],

    "Clinical Outcomes": [
        "clinical outcomes",
        "patient outcomes",
        "outcomes",
        "length of stay"
    ],

    "Utilisation": [
        "utilisation",
        "utilization",
        "procedure volume",
        "used enough",
        "volume"
    ],

    "Adoption & Growth": [
        "adoption",
        "growing",
        "growth",
        "accelerate",
        "gradual",
        "standard",
        "increasing"
    ]
}


def extract_themes(records):
    """
    Identify themes from expert responses.

    Only transcript evidence is returned.
    No new information is generated.
    """

    theme_results = {}

    for theme, keywords in THEMES.items():

        matched_records = []

        for record in records:

            text = record["text"].lower()

            matched_keywords = []

            for keyword in keywords:

                if keyword.lower() in text:
                    matched_keywords.append(keyword)

            if matched_keywords:

                matched_records.append({
                    "timestamp": record["timestamp"],
                    "speaker": record["speaker"],
                    "text": record["text"],
                    "keywords": matched_keywords
                })

        if matched_records:

            theme_results[theme] = matched_records

    return theme_results


def build_cross_country_themes(
    transcripts,
    parse_function
):
    """
    Build theme evidence across all countries.

    Returns:
        {
            "Theme": {
                "countries": {
                    "France": [...],
                    "Germany": [...],
                    "UK": [...]
                },
                "country_count": 3
            }
        }

    This function does not generate conclusions.
    It only organizes transcript evidence.
    """

    cross_country = {}

    for filename, content in transcripts.items():

        records = parse_function(content)

        themes = extract_themes(records)

        country_name = (
            filename
            .replace("Transcript_", "")
            .replace(".txt", "")
        )

        for theme, evidence in themes.items():

            if theme not in cross_country:

                cross_country[theme] = {
                    "countries": {},
                    "country_count": 0
                }

            cross_country[theme]["countries"][
                country_name
            ] = evidence

    for theme, data in cross_country.items():

        data["country_count"] = len(
            data["countries"]
        )

    return cross_country