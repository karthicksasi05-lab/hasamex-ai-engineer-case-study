import re


def parse_interview_guide(text):
    """
    Extract clean questions from the Interview Guide.
    """

    lines = text.splitlines()

    questions = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        if line.endswith("?"):

            # Remove numbering such as:
            # 1.
            # 2.
            # 3.
            clean_question = re.sub(
                r"^\d+[\.\)]\s*",
                "",
                line
            )

            questions.append(clean_question)

    return questions