from src.document_loader import load_transcripts
from src.transcript_parser import parse_transcript
from src.semantic_retriever import (
    load_embedding_model,
    semantic_search
)
from src.llm_prompt import build_grounded_prompt
from src.llm_engine import generate_answer


# --------------------------------------------------
# 1. Load transcripts
# --------------------------------------------------

transcripts = load_transcripts("data")


# --------------------------------------------------
# 2. Load embedding model
# --------------------------------------------------

model = load_embedding_model()


# --------------------------------------------------
# 3. User question
# --------------------------------------------------

question = "What are the main barriers to adoption?"


# --------------------------------------------------
# 4. Retrieve semantic evidence
# --------------------------------------------------

all_evidence = []


for filename, content in transcripts.items():

    records = parse_transcript(content)

    results = semantic_search(
        question,
        records,
        model,
        top_k=3
    )

    country = filename.replace(
        "Transcript_",
        ""
    ).replace(
        ".txt",
        ""
    )


    for result in results:

        all_evidence.append({
            "country": country,
            "speaker": result["speaker"],
            "timestamp": result["timestamp"],
            "text": result["text"],
            "score": result["score"]
        })


# --------------------------------------------------
# 5. Sort evidence by semantic relevance
# --------------------------------------------------

all_evidence = sorted(
    all_evidence,
    key=lambda x: x["score"],
    reverse=True
)


# --------------------------------------------------
# 6. Keep top evidence
# --------------------------------------------------

top_evidence = all_evidence[:8]


print("\nRetrieved Evidence")
print("=" * 60)


for item in top_evidence:

    print(
        f"\n{item['country']} | "
        f"{item['timestamp']} | "
        f"{item['speaker']}"
    )

    print(
        f"Score: {item['score']:.3f}"
    )

    print(item["text"])


# --------------------------------------------------
# 7. Build grounded prompt
# --------------------------------------------------

prompt = build_grounded_prompt(
    question,
    top_evidence
)


# --------------------------------------------------
# 8. Send evidence to Gemini
# --------------------------------------------------

print("\nSending grounded evidence to Gemini...\n")

answer = generate_answer(prompt)


# --------------------------------------------------
# 9. Display final answer
# --------------------------------------------------

print("GROUNDED LLM ANSWER")
print("=" * 60)

print(answer)