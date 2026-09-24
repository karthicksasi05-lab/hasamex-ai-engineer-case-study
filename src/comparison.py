from src.evidence_retriever import search_evidence


def build_country_evidence(
    transcripts,
    parse_function,
    search_function,
    keywords
):
    """
    Build evidence grouped by country.
    """

    country_evidence = {}

    for filename, content in transcripts.items():

        records = parse_function(content)

        evidence = search_function(
            records,
            keywords
        )

        country_name = (
            filename
            .replace("Transcript_", "")
            .replace(".txt", "")
        )

        country_evidence[country_name] = evidence

    return country_evidence


def get_country_summary(country_evidence):
    """
    Create a simple structured summary of the
    evidence available for each country.
    """

    summary = {}

    for country, evidence in country_evidence.items():

        if not evidence:

            summary[country] = []

            continue

        summary[country] = []

        for item in evidence:

            summary[country].append({
                "timestamp": item["timestamp"],
                "speaker": item["speaker"],
                "text": item["text"]
            })

    return summary


def find_common_evidence(country_evidence):
    """
    Identify broad common topics based on keywords
    appearing across multiple countries.

    This does not generate new information.
    """

    country_keywords = {}

    for country, evidence in country_evidence.items():

        keywords = set()

        for item in evidence:

            for keyword in item.get(
                "matched_keywords",
                []
            ):

                keywords.add(
                    keyword.lower()
                )

        country_keywords[country] = keywords


    all_countries = list(
        country_keywords.keys()
    )

    common_keywords = set()

    for keyword in set.union(
        *country_keywords.values()
    ) if country_keywords else set():

        count = sum(
            keyword in country_keywords[country]
            for country in all_countries
        )

        if count >= 2:

            common_keywords.add(
                keyword
            )

    return sorted(
        common_keywords
    )


def find_country_differences(country_evidence):
    """
    Return evidence grouped by country so that
    differences can be reviewed without inventing
    conclusions.
    """

    differences = {}

    for country, evidence in country_evidence.items():

        differences[country] = []

        for item in evidence:

            differences[country].append({
                "timestamp": item["timestamp"],
                "speaker": item["speaker"],
                "text": item["text"]
            })

    return differences


def find_contrasting_evidence(country_evidence):
    """
    Identify potentially contrasting evidence across countries.

    This function does not decide that experts disagree.
    It groups evidence so differences in wording,
    expectations, or emphasis can be reviewed.

    Returns evidence grouped by country.
    """

    countries = list(country_evidence.keys())

    contrasting_evidence = {}

    for country in countries:

        contrasting_evidence[country] = []

        for item in country_evidence[country]:

            contrasting_evidence[country].append({
                "timestamp": item["timestamp"],
                "speaker": item["speaker"],
                "text": item["text"]
            })

    return contrasting_evidence