from src.document_loader import load_transcripts
from src.transcript_parser import parse_transcript
from src.semantic_retriever import (
    load_embedding_model,
    semantic_search
)


# Load transcripts
transcripts = load_transcripts("data")

# Load embedding model
model = load_embedding_model()

# Test question
question = "What are the main barriers to adoption?"

print("\nSemantic Search Results\n")
print("=" * 60)


for filename, content in transcripts.items():

    records = parse_transcript(content)

    results = semantic_search(
        question,
        records,
        model,
        top_k=3
    )

    country = filename.replace(
        "Transcript_", ""
    ).replace(
        ".txt", ""
    )

    print(f"\n{country}")
    print("-" * 60)

    for result in results:

        print(
            f"[{result['timestamp']}] "
            f"{result['speaker']}"
        )

        print(
            f"Score: {result['score']:.3f}"
        )

        print(
            result["text"]
        )

        print()