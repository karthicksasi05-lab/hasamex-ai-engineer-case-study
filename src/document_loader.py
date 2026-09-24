from pathlib import Path


def load_text_file(file_path):
    """
    Read one text file and return its content.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def load_transcripts(data_folder="data"):
    """
    Load all transcript text files from the data folder.
    """

    data_path = Path(data_folder)

    transcript_files = sorted(
        data_path.glob("Transcript_*.txt")
    )

    transcripts = {}

    for file_path in transcript_files:
        transcripts[file_path.name] = load_text_file(file_path)

    return transcripts