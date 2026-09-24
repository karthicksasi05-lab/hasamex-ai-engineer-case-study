import re


def parse_transcript(text):
    """
    Convert transcript text into structured records.
    """

    lines = text.splitlines()

    records = []
    current_timestamp = None

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # Check whether the line is a timestamp
        timestamp_match = re.match(
            r"^\d{2}:\d{2}(?::\d{2})?$",
            line
        )

        if timestamp_match:
            current_timestamp = line
            continue

        # Check whether the line contains Speaker: text
        speaker_match = re.match(
            r"^([^:]+):\s*(.*)$",
            line
        )

        if speaker_match and current_timestamp:

            speaker = speaker_match.group(1).strip()
            text_content = speaker_match.group(2).strip()

            records.append({
                "timestamp": current_timestamp,
                "speaker": speaker,
                "text": text_content
            })

            current_timestamp = None

    return records